//! `SchedulerCore` — persistent scheduler state in Rust (Phase 3).
//!
//! The core owns: the waiting/running queues, the in-flight chunked
//! request, the radix tree (from `sglang-radix`), the new-token-ratio
//! tracker, and per-request token storage. Python drives it:
//!
//! ```text
//! core.ingest(ingress)                      // enqueue new reqs
//! core.apply_result(results, kv_rows)       // accept/finish/stash (last-batch order)
//! (plan, events) = core.plan(env)           // next-batch decision + tree/lock bookkeeping
//! ```
//!
//! which mirrors `event_loop_normal` (ingress -> plan -> run -> result).
//! Scope: base MHA, single-node, non-PP/DP, non-spec, non-overlap.
//!
//! KV-index contract: tree values are global KV pool positions. Python
//! supplies them at insert time (`kv_rows` in the result snapshot) and
//! frees what the core emits as events. The core never allocates;
//! allocation stays in the Python paged allocator (`BatchPlan.alloc_*`).
//!
//! Plan-for-plan parity with the Python scheduler:
//! - the prefill/decode decision is the same pure engine as
//!   [`crate::planner::plan_next_batch`], with the live tree as the
//!   admission tree;
//! - the persistent admission lock (`_req_inc_lock_ref`) is taken at
//!   admission and released at stash/finish/retract;
//! - the `batch_is_full` flag follows the `get_next_batch_to_run`
//!   merge-shrink and `update_running_batch` shrink resets.

use std::collections::HashSet;

use sglang_radix::{EvictionPolicy, ROOT, RadixKey, RadixTree};

use crate::adder::AdmissionTree;
use crate::ntr::Ntr;
use crate::planner::plan_next_batch_with_tree;
use crate::policy;
use crate::spec::SpecCounters;
use crate::types::{BatchPlan, CHUNKED_IDX, Config, PlanReq, Policy, StepEnv};

/// One incoming request (ingress).
#[derive(Debug, Clone)]
pub struct IngressReq {
    pub rid: u64,
    pub pool_idx: u32,
    /// `origin_input_ids` (token ids; the tree key).
    pub origin: Vec<i64>,
    pub max_new_tokens: u32,
    pub priority: i32,
    pub arrival_seq: u64,
    pub routing_key: u64,
    pub ignore_eos: bool,
}

/// Spec-v2 (MTP/EAGLE/DFlash) bookkeeping for one result row (plan §9).
/// The grammar-truncated accepted run arrives as `ResultRow::accepted`;
/// this carries the pre-truncation counters Python settled.
#[derive(Debug, Clone)]
pub struct ResultSpec {
    /// `accept_lens[i]` — accepted length before grammar truncation
    /// (drafts + bonus).
    pub accept_len: u32,
    /// Python settled this req's spec counters this step (not retracted
    /// and not finished before the step).
    pub settled: bool,
    /// `block_accept_lens[i]` (None = the batch has no block lens).
    pub block_accept_len: Option<u32>,
    /// `cap_lens[i]` (None = the batch has no cap lens).
    pub cap_len: Option<u32>,
}

/// One last-batch request's result row.
#[derive(Debug, Clone)]
pub struct ResultRow {
    /// Accepted output tokens this step (decode: the sampled token; the
    /// full grammar-truncated run for spec-v2 rows).
    pub accepted: Vec<i64>,
    pub finished: bool,
    /// 0 = none, 1 = length, 2 = stop token, 3 = stop string, 4 = abort.
    pub finish_reason: u32,
    /// Spec-v2 bookkeeping (None = non-spec step).
    pub spec: Option<ResultSpec>,
}

/// A request's KV row (global KV pool positions), supplied by Python for
/// the tree insert at finish / chunk-stash time.
#[derive(Debug, Clone)]
pub struct KvRow {
    pub core_idx: u32,
    pub row: Vec<i64>,
}

/// Events Python must execute alongside (or instead of) default
/// bookkeeping, in emission order.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// Free these tree-evicted KV runs in the allocator
    /// (`free_segment(run, start_pos=0)`).
    Evict { values: Vec<Vec<i64>> },
    /// Free token-offset ranges `[start, end)` of this req's
    /// `req_to_token` row in the paged allocator.
    FreeSegments {
        pool_idx: u32,
        ranges: Vec<(u32, u32)>,
    },
    /// Rewrite `row[pool_idx, start..]` with `new_indices` (the
    /// `cache_unfinished_req` row rewrite that re-points the row at the
    /// tree-owned pool positions).
    StashRowWrite {
        pool_idx: u32,
        start: u32,
        new_indices: Vec<i64>,
    },
    /// A request finished or was aborted: Python streams/aborts its output.
    /// `out_len` is the final accepted output length.
    Finished {
        core_idx: u32,
        reason: u32,
        out_len: u32,
    },
}

/// One scheduler step's output.
#[derive(Debug)]
pub struct StepOut {
    pub plan: BatchPlan,
    pub events: Vec<Event>,
}

/// Per-request persistent state.
#[derive(Debug, Clone)]
struct CoreReq {
    rid: u64,
    pool_idx: u32,
    origin: Vec<i64>,
    out: Vec<i64>,
    max_new_tokens: u32,
    priority: i32,
    arrival_seq: u64,
    routing_key: u64,
    ignore_eos: bool,
    retracted_stain: bool,
    /// Terminal node of the current matched prefix (page-floored).
    last_node: u32,
    /// Matched prefix length (page-floored).
    prefix_len: u32,
    /// `cache_protected_len`: the tree-owned prefix of this req's KV row.
    protected_len: u32,
    finished: bool,
    finish_reason: u32,
    /// Committed KV row length (running / chunked).
    committed_len: u32,
    /// The persistent admission lock is held on `last_node` (taken at
    /// prefill admission, released at stash/finish/retract/drop). Waiting
    /// reqs carry a `last_node` from plan-time scoring but no lock.
    locked: bool,
    /// Spec-v2 counters (plan §9); live on spec-decode requests only.
    spec: SpecCounters,
}

impl CoreReq {
    fn fill_tokens(&self) -> Vec<i64> {
        let mut v = self.origin.clone();
        v.extend_from_slice(&self.out);
        v
    }

    fn fill_len(&self) -> u32 {
        self.origin.len() as u32 + self.out.len() as u32
    }

    fn snapshot(&self) -> PlanReq {
        PlanReq {
            pool_idx: self.pool_idx,
            origin_len: self.origin.len() as u32,
            out_len: self.out.len() as u32,
            committed_len: self.committed_len,
            prefix_len: self.prefix_len,
            last_node: self.last_node,
            priority: self.priority,
            arrival_seq: self.arrival_seq,
            max_new_tokens: self.max_new_tokens,
            routing_key: self.routing_key,
            ignore_eos: self.ignore_eos,
            finished: self.finished,
            retracted_stain: self.retracted_stain,
            host_hit_length: 0,
        }
    }
}

/// Live-tree adapter for the adder's temporary admission lock (Python's
/// `_lock_node`: inc before the budget checks, dec after).
struct TreeAdapter<'a> {
    tree: &'a mut RadixTree,
}

impl AdmissionTree for TreeAdapter<'_> {
    fn temp_lock(&mut self, node: u32) {
        if node != ROOT {
            self.tree.inc_lock_ref(node);
        }
    }
    fn temp_unlock(&mut self, node: u32) {
        if node != ROOT {
            self.tree.dec_lock_ref(node);
        }
    }
}

pub struct SchedulerCore {
    cfg: Config,
    ntr: Ntr,
    tree: RadixTree,
    /// All requests, indexed by `core_idx` (stable for the req's lifetime).
    reqs: Vec<CoreReq>,
    free: Vec<u32>,
    waiting: Vec<u32>,
    running: Vec<u32>,
    /// The last prefill batch, in execution order, awaiting the merge into
    /// running (Python's `last_batch` extend-merge at the top of
    /// `get_next_batch_to_run`). Excludes the still-parked chunked req.
    pending_merge: Vec<u32>,
    /// Execution order of the last executed batch (`apply_result` input).
    last_batch: Vec<u32>,
    /// True when the last executed batch was prefill (extend) mode.
    last_was_prefill: bool,
    /// Last-batch reqs that ran as decode inside a mixed batch: their
    /// accepted tokens commit KV like a decode step.
    last_decode_like: HashSet<u32>,
    /// The in-flight chunked request (parked between prefill passes).
    chunked: Option<u32>,
    batch_is_full: bool,
    iter: u64,
}

impl SchedulerCore {
    pub fn new(cfg: Config, tree_policy: EvictionPolicy) -> Self {
        Self {
            tree: RadixTree::new(cfg.page_size as usize, false, tree_policy),
            ntr: Ntr::from_config(&cfg),
            cfg,
            reqs: Vec::new(),
            free: Vec::new(),
            waiting: Vec::new(),
            running: Vec::new(),
            pending_merge: Vec::new(),
            last_batch: Vec::new(),
            last_was_prefill: false,
            last_decode_like: HashSet::new(),
            chunked: None,
            batch_is_full: false,
            iter: 0,
        }
    }

    pub fn tree(&self) -> &RadixTree {
        &self.tree
    }

    pub fn waiting(&self) -> &[u32] {
        &self.waiting
    }

    pub fn running(&self) -> &[u32] {
        &self.running
    }

    pub fn chunked_idx(&self) -> Option<u32> {
        self.chunked
    }

    /// The admitted-but-not-yet-merged set (joins `running` at the next
    /// plan). Disjoint from waiting/running/chunked.
    pub fn pending_merge(&self) -> &[u32] {
        &self.pending_merge
    }

    pub fn new_token_ratio(&self) -> f64 {
        self.ntr.current()
    }

    pub fn batch_is_full(&self) -> bool {
        self.batch_is_full
    }

    /// Execution order of the last executed batch (for the Python side to
    /// line up `results` / `kv_rows`).
    pub fn last_batch(&self) -> &[u32] {
        &self.last_batch
    }

    pub fn req_pool_idx(&self, core_idx: u32) -> u32 {
        self.reqs[core_idx as usize].pool_idx
    }

    pub fn req_rid(&self, core_idx: u32) -> u64 {
        self.reqs[core_idx as usize].rid
    }

    pub fn req_out_len(&self, core_idx: u32) -> u32 {
        self.reqs[core_idx as usize].out.len() as u32
    }

    /// Committed KV row length (`origin + accepted so far`, partial for a
    /// parked chunk).
    pub fn req_committed_len(&self, core_idx: u32) -> u32 {
        self.reqs[core_idx as usize].committed_len
    }

    pub fn req_finished(&self, core_idx: u32) -> bool {
        self.reqs[core_idx as usize].finished
    }

    /// Tree-backed prefix length after admission: the row's `[0, prefix_len)`
    /// views tree positions and is never row-allocated.
    pub fn req_prefix_len(&self, core_idx: u32) -> u32 {
        self.reqs[core_idx as usize].prefix_len
    }

    /// Page-aligned protected length (the row's `[0, protected_len)` is
    /// tree-backed and excluded from any row free).
    pub fn req_protected_len(&self, core_idx: u32) -> u32 {
        self.reqs[core_idx as usize].protected_len
    }

    /// Spec-v2 counters for a live request (plan §9).
    pub fn spec_counters(&self, core_idx: u32) -> Option<&SpecCounters> {
        self.reqs.get(core_idx as usize).map(|r| &r.spec)
    }

    pub fn req_retracted_stain(&self, core_idx: u32) -> bool {
        self.reqs[core_idx as usize].retracted_stain
    }

    /// Enqueue new requests (ingress). No tree lock is taken: the base
    /// `RadixCache` matches/locks only inside the admission loop.
    pub fn ingest(&mut self, reqs: Vec<IngressReq>) -> Vec<u32> {
        let mut out = Vec::with_capacity(reqs.len());
        for r in reqs {
            let idx = self.free.pop().unwrap_or(self.reqs.len() as u32);
            self.reqs.push(CoreReq {
                rid: r.rid,
                pool_idx: r.pool_idx,
                origin: r.origin,
                out: Vec::new(),
                max_new_tokens: r.max_new_tokens,
                priority: r.priority,
                arrival_seq: r.arrival_seq,
                routing_key: r.routing_key,
                ignore_eos: r.ignore_eos,
                retracted_stain: false,
                last_node: ROOT,
                prefix_len: 0,
                protected_len: 0,
                finished: false,
                finish_reason: 0,
                committed_len: 0,
                locked: false,
                spec: SpecCounters::default(),
            });
            if idx as usize != self.reqs.len() - 1 {
                // Slot recycle: the recycled index already holds a dead
                // request; move the fresh one into place.
                let fresh = self.reqs.pop().unwrap();
                self.reqs[idx as usize] = fresh;
            }
            out.push(idx);
            self.waiting.push(idx);
        }
        out
    }

    /// Apply the previous batch's results: accept outputs, process finishes
    /// (`release_kv_cache`), stash a chunked req's new KV
    /// (`cache_unfinished_req`). `results` is ordered as the previous
    /// executed batch (`last_batch` order).
    pub fn apply_result(&mut self, results: &[ResultRow], kv_rows: &[KvRow]) -> Vec<Event> {
        debug_assert_eq!(results.len(), self.last_batch.len());
        let mut events = Vec::new();
        let batch = self.last_batch.clone();
        for (core_idx, row) in batch.iter().zip(results) {
            let core_idx = *core_idx;
            {
                let r = &mut self.reqs[core_idx as usize];
                r.out.extend_from_slice(&row.accepted);
                // Decode steps (and the mixed-batch decode tail) commit KV
                // for the accepted token; prefill steps do not (the sampled
                // token's KV lands on the next decode allocation).
                if !self.last_was_prefill || self.last_decode_like.contains(&core_idx) {
                    r.committed_len += row.accepted.len() as u32;
                }
                // Spec-v2 counters (plan §9): settled only — a retracted or
                // pre-finished row committed nothing, matching the Python
                // gate at resolve time.
                if let Some(s) = row.spec.as_ref().filter(|x| x.settled) {
                    r.spec.update(
                        s.accept_len.saturating_sub(1),
                        s.block_accept_len,
                        s.cap_len,
                    );
                }
            }

            if row.finished {
                let kv = kv_rows
                    .iter()
                    .find(|k| k.core_idx == core_idx)
                    .map(|k| &k.row);
                self.release_req(core_idx, row.finish_reason, &mut events, kv);
            } else if self.last_was_prefill && !self.last_decode_like.contains(&core_idx) {
                // `maybe_cache_unfinished_req` on every completed prefill
                // req (parked chunks, the just-completed chunk, and regular
                // admits alike): new KV beyond the protected prefix goes
                // into the tree and the row is re-pointed at it.
                let committed = self.reqs[core_idx as usize].committed_len;
                let protected = self.reqs[core_idx as usize].protected_len;
                if committed > protected
                    && let Some(kv) = kv_rows.iter().find(|k| k.core_idx == core_idx)
                {
                    self.stash_chunk(core_idx, committed, &kv.row, &mut events);
                }
            }
        }
        self.last_batch.clear();
        self.last_was_prefill = false;
        self.last_decode_like.clear();
        events
    }

    /// `release_kv_cache` on the base path: insert `[0, committed)` into
    /// the tree, free the duplicate range `[protected, freed_end)` and the
    /// unaligned tail `[key_len, committed)`, release the prefix lock.
    fn release_req(
        &mut self,
        core_idx: u32,
        reason: u32,
        events: &mut Vec<Event>,
        kv: Option<&Vec<i64>>,
    ) {
        let committed = self.reqs[core_idx as usize].committed_len as usize;
        let fill = self.reqs[core_idx as usize].fill_tokens();
        let key_tokens = committed.min(fill.len());
        let key = RadixKey::new(&fill[..key_tokens]);
        let flat = key.flatten_page_aligned(self.cfg.page_size as usize);
        let (pool_idx, protected, last_node, priority, out_len) = {
            let r = &self.reqs[core_idx as usize];
            (
                r.pool_idx,
                r.protected_len,
                r.last_node,
                r.priority,
                r.out.len() as u32,
            )
        };

        let mut ranges: Vec<(u32, u32)> = Vec::new();
        if let Some(row) = kv {
            let values = &row[..flat.len().min(row.len())];
            let insert = self.tree.insert(&key, values, priority, false);
            let freed_end = insert.prefix_len as u32;
            if freed_end > protected {
                ranges.push((protected, freed_end));
            }
        }
        if committed as u32 > flat.len() as u32 {
            ranges.push((flat.len() as u32, committed as u32));
        }
        if !ranges.is_empty() {
            events.push(Event::FreeSegments { pool_idx, ranges });
        }
        if last_node != ROOT {
            self.tree.dec_lock_ref(last_node);
        }

        {
            let r = &mut self.reqs[core_idx as usize];
            r.finished = true;
            r.finish_reason = reason;
            r.locked = false;
        }
        events.push(Event::Finished {
            core_idx,
            reason,
            out_len,
        });
    }

    /// `cache_unfinished_req(req, chunked=True)` on the base path.
    fn stash_chunk(&mut self, core_idx: u32, committed: u32, row: &[i64], events: &mut Vec<Event>) {
        let fill = self.reqs[core_idx as usize].fill_tokens();
        let key_tokens = (committed as usize).min(fill.len());
        let key = RadixKey::new(&fill[..key_tokens]);
        let flat = key.flatten_page_aligned(self.cfg.page_size as usize);
        let (pool_idx, protected, old_last, priority) = {
            let r = &self.reqs[core_idx as usize];
            (r.pool_idx, r.protected_len, r.last_node, r.priority)
        };
        let values = &row[..flat.len().min(row.len())];
        let insert = self.tree.insert(&key, values, priority, true);
        let new_prefix = insert.prefix_len as u32;
        if new_prefix > protected {
            events.push(Event::FreeSegments {
                pool_idx,
                ranges: vec![(protected, new_prefix)],
            });
        }

        // Re-match: the full key must be resident now.
        let matched = self.tree.match_prefix(&key);
        debug_assert_eq!(
            matched.indices.len(),
            flat.len(),
            "stash must re-match fully"
        );
        let new_protected = matched.indices.len() as u32;
        if new_protected > protected {
            events.push(Event::StashRowWrite {
                pool_idx,
                start: protected,
                new_indices: matched.indices[protected as usize..].to_vec(),
            });
        }

        if old_last != ROOT {
            self.tree.dec_lock_ref(old_last);
        }
        let new_last = matched.last_node;
        if new_last != ROOT {
            self.tree.inc_lock_ref(new_last);
        }
        let r = &mut self.reqs[core_idx as usize];
        r.last_node = new_last;
        r.prefix_len = new_protected;
        r.protected_len = new_protected;
        r.locked = new_last != ROOT;
    }

    /// User-initiated abort (Python `abort_request`): drop the request from
    /// waiting, running, or the pending last-batch merge, release its KV row
    /// and prefix lock, and recycle the slot. Fresh waiting reqs carry no
    /// resources (the release steps are no-ops for them). Returns the events
    /// Python must execute (KV frees + the abort stream notice).
    pub fn drop_request(&mut self, core_idx: u32) -> Vec<Event> {
        let mut events = Vec::new();
        let core = core_idx as usize;
        if core >= self.reqs.len() {
            return events;
        }
        let queued = self.waiting.contains(&core_idx)
            || self.running.contains(&core_idx)
            || self.pending_merge.contains(&core_idx);
        if !queued {
            return events; // finished or unknown
        }
        self.waiting.retain(|&i| i != core_idx);
        self.running.retain(|&i| i != core_idx);
        self.pending_merge.retain(|&i| i != core_idx);
        if self.chunked == Some(core_idx) {
            self.chunked = None;
        }
        let (pool_idx, committed, protected, last_node, locked) = {
            let r = &self.reqs[core];
            (r.pool_idx, r.committed_len, r.protected_len, r.last_node, r.locked)
        };
        // Only admitted reqs hold the lock; waiting reqs carry a scored
        // `last_node` without it (Python's queue abort touches no tree lock).
        if locked && last_node != ROOT {
            self.tree.dec_lock_ref(last_node);
        }
        // Same ownership split as retraction: the tree keeps [0, protected).
        if committed > protected {
            events.push(Event::FreeSegments {
                pool_idx,
                ranges: vec![(protected, committed)],
            });
        }
        let out_len = {
            let r = &mut self.reqs[core];
            let n = r.out.len() as u32;
            r.finished = true;
            r.finish_reason = 4; // abort
            n
        };
        events.push(Event::Finished {
            core_idx,
            reason: 4,
            out_len,
        });
        self.free.push(core_idx);
        events
    }

    /// The next-batch decision (ingress + results already applied).
    pub fn plan(&mut self, env: &StepEnv) -> StepOut {
        let mut events = Vec::new();

        // 0. `filter_batch()`: finished reqs leave the running batch after
        //    every result pass, before any plan (Python drops them in
        //    `process_batch_result`, not on the next decode plan).
        if self.running.iter().any(|&i| self.reqs[i as usize].finished) {
            let mut kept: Vec<u32> = Vec::new();
            for i in self.running.drain(..) {
                if self.reqs[i as usize].finished {
                    self.free.push(i);
                } else {
                    kept.push(i);
                }
            }
            self.running = kept;
        }

        // 1. Merge the last prefill batch into running (Python: the
        //    `last_batch` extend-merge). Finished reqs never join; the
        //    still-parked chunked req stays parked; a shrunken merge resets
        //    `batch_is_full`.
        if !self.pending_merge.is_empty() {
            let merged = self
                .pending_merge
                .iter()
                .filter(|&&i| !self.reqs[i as usize].finished && self.chunked != Some(i))
                .count();
            if merged < self.pending_merge.len() {
                self.batch_is_full = false;
            }
            for i in self.pending_merge.drain(..) {
                if self.reqs[i as usize].finished {
                    self.free.push(i);
                } else if self.chunked != Some(i) {
                    self.running.push(i);
                }
            }
        }

        // 2. Scoring: re-match every waiting req against the live tree
        //    (Python's `init_next_round_input` / schedule-time match). LPM
        //    additionally runs the in-batch dedup on a scratch tree
        //    (Python's `waiting_queue_radix_tree`).
        let n_waiting = self.waiting.len();
        let mut scores = vec![0u32; n_waiting];
        let mut deprio = vec![false; n_waiting];
        {
            let page = self.cfg.page_size as usize;
            let lpm = self.cfg.active_policy() == Policy::Lpm
                && n_waiting as u32 <= self.cfg.lpm_queue_degrade_at;
            let mut scratch = lpm.then(|| RadixTree::new(page, false, EvictionPolicy::Lru));
            for (pos, &core_idx) in self.waiting.iter().enumerate() {
                let fill = self.reqs[core_idx as usize].fill_tokens();
                let key = RadixKey::new(&fill);
                let matched = self.tree.match_prefix(&key);
                self.reqs[core_idx as usize].prefix_len = matched.indices.len() as u32;
                self.reqs[core_idx as usize].last_node = matched.last_node;
                scores[pos] = matched.indices.len() as u32;
                if let Some(scratch) = scratch.as_mut()
                    && matched.indices.len() as u32 <= self.cfg.in_batch_check_threshold
                {
                    let dm = scratch.match_prefix(&key);
                    if dm.indices.len() as u32 >= self.cfg.in_batch_deprioritize_threshold {
                        deprio[pos] = true;
                    } else {
                        let flat = key.flatten_page_aligned(page);
                        scratch.insert(&key, &vec![0i64; flat.len()], 0, false);
                    }
                }
            }
        }

        // 3. Snapshots (after scoring: the adder budgets off the fresh
        //    prefix lengths).
        let mut waiting_snap: Vec<PlanReq> = self
            .waiting
            .iter()
            .map(|&i| self.reqs[i as usize].snapshot())
            .collect();
        let running_snap: Vec<PlanReq> = self
            .running
            .iter()
            .map(|&i| self.reqs[i as usize].snapshot())
            .collect();
        let mut waiting_cores: Vec<u32> = self.waiting.clone();
        let running_cores: Vec<u32> = self.running.clone();

        if self.cfg.active_policy() == Policy::DfsWeight {
            // Python reorders the queue in DFS order; the planner's DfsWeight
            // arm degrades to the stable in-order pass, so admission follows
            // this permuted order.
            let order = policy::order_dfs(&waiting_snap, &|n| self.tree.node_children(n));
            let snap: Vec<PlanReq> = order.iter().map(|&i| waiting_snap[i as usize]).collect();
            waiting_snap = snap;
            scores = order.iter().map(|&i| scores[i as usize]).collect();
            deprio = order.iter().map(|&i| deprio[i as usize]).collect();
            waiting_cores = order.iter().map(|&i| waiting_cores[i as usize]).collect();
        }

        let chunked_snap = self.chunked.map(|i| {
            let mut s = self.reqs[i as usize].snapshot();
            s.committed_len = self.reqs[i as usize].fill_len();
            s
        });
        let mut env2 = *env;
        env2.batch_is_full = self.batch_is_full;
        // The core owns the tree, so its evictable size is authoritative
        // (the env value is the shadow-mode input).
        env2.tree_evictable_tokens = self.tree.evictable_size() as u32;

        let mut adapter = TreeAdapter {
            tree: &mut self.tree,
        };
        let plan = plan_next_batch_with_tree(
            &self.cfg,
            &self.ntr,
            &waiting_snap,
            &running_snap,
            chunked_snap.as_ref(),
            &scores,
            &deprio,
            &env2,
            self.iter,
            &mut adapter,
        );

        // 4. Bookkeeping (snap -> core mapping happens BEFORE any queue
        //    mutation: `waiting_cores` / `running_cores` are stable).
        let mut last_batch: Vec<u32> = Vec::new();
        let mut decode_like: HashSet<u32> = HashSet::new();
        if let Some(p) = &plan.prefill {
            debug_assert!(!p.admitted.is_empty());
            let chunked_still = matches!(p.chunked, Some(CHUNKED_IDX));
            let new_chunked = p.chunked.filter(|&w| w != CHUNKED_IDX);
            let mut pending: Vec<u32> = Vec::new();

            // Remove the admitted reqs from `waiting` by *core index*, not
            // by snap position: under DfsWeight the snap (and
            // `waiting_cores`) is permuted, so `a.waiting_idx` is a position
            // in the permuted order, not in `self.waiting`.
            let admitted_cores: HashSet<u32> = p
                .admitted
                .iter()
                .filter(|a| a.waiting_idx != CHUNKED_IDX)
                .map(|a| waiting_cores[a.waiting_idx as usize])
                .collect();
            self.waiting.retain(|&c| !admitted_cores.contains(&c));

            for a in &p.admitted {
                if a.waiting_idx == CHUNKED_IDX {
                    let c = self.chunked.unwrap();
                    self.reqs[c as usize].committed_len = a.extend_end;
                    last_batch.push(c);
                    if chunked_still {
                        self.chunked = Some(c);
                    } else {
                        // Last chunk completed this pass: the req joins
                        // running through the next merge.
                        self.chunked = None;
                        pending.push(c);
                    }
                    continue;
                }
                let core = waiting_cores[a.waiting_idx as usize];
                {
                    let r = &mut self.reqs[core as usize];
                    r.committed_len = a.extend_end;
                    r.prefix_len = a.prefix_len;
                    r.protected_len = a.prefix_len;
                    // The persistent admission lock
                    // (Python `_req_inc_lock_ref`).
                    if r.last_node != ROOT {
                        self.tree.inc_lock_ref(r.last_node);
                    }
                    r.locked = r.last_node != ROOT;
                }
                last_batch.push(core);
                if Some(a.waiting_idx) == new_chunked {
                    self.chunked = Some(core);
                } else {
                    pending.push(core);
                }
            }

            if p.mixed {
                // Mixed-style chunked prefill: the running reqs join this
                // batch and running becomes empty (the flag is carried by
                // the plan).
                for &i in &running_cores {
                    if self.reqs[i as usize].finished {
                        self.free.push(i);
                    } else {
                        last_batch.push(i);
                        decode_like.insert(i);
                    }
                }
                self.running.clear();
            }

            self.pending_merge = pending;
            self.last_batch = last_batch;
            self.last_was_prefill = true;
            self.last_decode_like = decode_like;
            self.batch_is_full = plan.batch_is_full;
        } else if let Some(d) = &plan.decode {
            // Tree eviction first (Python: `check_decode_mem` evicts the
            // shortfall before the retraction frees land).
            if d.evict_tokens > 0 {
                let evicted = self.tree.evict(d.evict_tokens as usize);
                if !evicted.evicted_values.is_empty() {
                    events.push(Event::Evict {
                        values: evicted.evicted_values,
                    });
                }
            }

            // Retracted reqs: free the full row, release the prefix lock,
            // requeue stained (Python: `release_req` — no tree insert, the
            // space is needed instantly).
            for &i in &d.retract {
                let core = running_cores[i as usize];
                {
                    let r = &mut self.reqs[core as usize];
                    let pool_idx = r.pool_idx;
                    let committed = r.committed_len;
                    let protected = r.protected_len;
                    let last_node = r.last_node;
                    r.retracted_stain = true;
                    r.committed_len = 0;
                    r.prefix_len = 0;
                    r.last_node = ROOT;
                    r.protected_len = 0;
                    r.locked = false;
                    if last_node != ROOT {
                        self.tree.dec_lock_ref(last_node);
                    }
                    // Python `release_kv_cache(is_insert=False)` frees
                    // [cache_protected_len, committed) only: [0, protected)
                    // is tree-owned and stays in the tree.
                    if committed > protected {
                        events.push(Event::FreeSegments {
                            pool_idx,
                            ranges: vec![(protected, committed)],
                        });
                    }
                }
                self.waiting.push(core);
            }

            // Aborted reqs: free the row, release the lock, recycle the slot.
            for &i in &d.abort {
                let core = running_cores[i as usize];
                let (pool_idx, committed, protected, last_node, out_len) = {
                    let r = &self.reqs[core as usize];
                    (
                        r.pool_idx,
                        r.committed_len,
                        r.protected_len,
                        r.last_node,
                        r.out.len() as u32,
                    )
                };
                if last_node != ROOT {
                    self.tree.dec_lock_ref(last_node);
                }
                // Same ownership split as retraction: the tree keeps
                // [0, protected); the row frees the rest. (Python's abort
                // finish would additionally insert the prefix, but the core
                // has no row values at plan time, so it skips the insert.)
                if committed > protected {
                    events.push(Event::FreeSegments {
                        pool_idx,
                        ranges: vec![(protected, committed)],
                    });
                }
                {
                    let r = &mut self.reqs[core as usize];
                    r.finished = true;
                    r.finish_reason = 4; // abort
                    r.locked = false;
                }
                events.push(Event::Finished {
                    core_idx: core,
                    reason: 4,
                    out_len,
                });
                self.free.push(core);
            }

            // Finished-filtered reqs: released at result time; recycle now.
            for &i in &d.finished_removed {
                let core = running_cores[i as usize];
                self.free.push(core);
            }

            self.running = d
                .decode
                .iter()
                .map(|&i| running_cores[i as usize])
                .collect();
            if !d.decode.is_empty() {
                if d.retract.is_empty() {
                    self.ntr.decay_step();
                } else {
                    self.ntr.set_current(d.ntr);
                }
            }

            self.last_batch = d
                .decode
                .iter()
                .map(|&i| running_cores[i as usize])
                .collect();
            self.last_was_prefill = false;
            self.last_decode_like = decode_like;
            self.batch_is_full = plan.batch_is_full;
        } else {
            // Idle: nothing ran this iteration.
            self.last_batch.clear();
            self.last_was_prefill = false;
            self.last_decode_like = decode_like;
            self.batch_is_full = plan.batch_is_full;
        }
        self.iter += 1;
        StepOut { plan, events }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Plan §12: fuzz `core.step` with random ingress + result sequences.
    /// Invariants: no double-free of KV pages or row segments, tree lock
    /// counts stay consistent (a u32 underflow panics in debug), plan
    /// allocations never exceed the free pool, and identical op sequences
    /// replay deterministically.
    #[test]
    fn fuzz_core_step_invariants() {
        for &seed in &[0x5eed, 0xc0de] {
            for (policy, priority) in [
                (Policy::Fcfs, false),
                (Policy::Lpm, false),
                (Policy::DfsWeight, false),
                (Policy::Lof, true),
                (Policy::Fcfs, true),
                (Policy::Lpm, true),
            ] {
                let a = fuzz_once(seed, policy, priority);
                let b = fuzz_once(seed, policy, priority);
                assert_eq!(a, b, "seed {seed:#x} policy {policy:?}: deterministic replay diverged");
            }
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    struct FuzzSnap {
        plan: BatchPlan,
        events: Vec<Event>,
        waiting: Vec<u32>,
        running: Vec<u32>,
        chunked: Option<u32>,
        tree: (i64, i64, i64),
        ntr: f64,
    }

    fn fuzz_once(seed: u64, policy: Policy, priority: bool) -> Vec<FuzzSnap> {
        let page = 8u32;
        let cfg = Config {
            policy,
            page_size: page,
            max_prefill_tokens: 512,
            chunked_prefill_size: Some(128),
            mixed_chunk: true,
            priority_scheduling: priority,
            ..Config::default()
        };
        let mut core = SchedulerCore::new(cfg, EvictionPolicy::Lru);

        let mut rng = seed;
        let mut nx = |n: u64| -> u64 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng % n
        };

        let mut snaps: Vec<FuzzSnap> = Vec::new();
        // Pooled-KV model: `allocated` = live values, `evicted` = freed by
        // Evict events (each value may be evicted at most once), `owned` =
        // per-row token offsets backed by the req's own allocation (the
        // tree-backed prefix is tracked loosely).
        let mut next_val: i64 = 1;
        let mut next_tok: u64 = 0;
        let mut next_pool: u32 = 0;
        let mut next_arrival: u64 = 0;
        let mut allocated: HashSet<i64> = HashSet::new();
        let mut evicted: HashSet<i64> = HashSet::new();
        let mut vals: Vec<Vec<i64>> = Vec::new();
        let mut committed: Vec<u32> = Vec::new();
        let mut owned: Vec<HashSet<u32>> = Vec::new();
        let mut avail: i64 = 2048;
        let mut batch_is_full = false;
        let mut last_mode: u8 = 0;
        let mut last_decode_like: HashSet<u32> = HashSet::new();

        // Prefix banks: banks 0..=3 share a 32-token head, so repeated and
        // re-queued origins re-match the tree.
        let head: Vec<i64> = (0..32).map(|i| (1 + i * 7) as i64).collect();
        let banks: Vec<Vec<i64>> = (0..8)
            .map(|b| {
                let mut v = head.clone();
                for i in 0..(40 + b * 16) {
                    v.push(100 + b as i64 * 500 + i as i64);
                }
                v
            })
            .collect();

        for iter in 0..200 {
            // 1. Apply the previous batch's results (driver order: apply the
            //    last batch, then plan the next one).
            if !core.last_batch().is_empty() {
                let batch = core.last_batch().to_vec();
                let mut rows: Vec<ResultRow> = Vec::with_capacity(batch.len());
                for &c in &batch {
                    let is_decode =
                        last_mode == crate::types::MODE_DECODE || last_decode_like.contains(&c);
                    let mut accepted = Vec::new();
                    if is_decode {
                        accepted.push(2_000_000 + next_tok as i64);
                        next_tok += 1;
                        // The decode token's page was row-allocated at
                        // plan time (plan_decode reserved it), so the
                        // ownership lands before the apply's events are
                        // checked — the finish path may free the
                        // just-committed tail immediately.
                        let before = committed[c as usize];
                        committed[c as usize] = before + accepted.len() as u32;
                        let pool = core.req_pool_idx(c);
                        for off in before..committed[c as usize] {
                            owned[pool as usize].insert(off);
                            let v = next_val;
                            next_val += 1;
                            allocated.insert(v);
                            vals[c as usize].push(v);
                        }
                    }
                    let finished_row = is_decode && nx(8) == 0;
                    let spec = if is_decode && nx(5) == 0 {
                        Some(ResultSpec {
                            accept_len: 2 + nx(3) as u32,
                            settled: true,
                            block_accept_len: (nx(2) == 0).then(|| 1 + nx(3) as u32),
                            cap_len: (nx(2) == 0).then(|| 1 + nx(3) as u32),
                        })
                    } else {
                        None
                    };
                    rows.push(ResultRow {
                        accepted,
                        finished: finished_row,
                        finish_reason: if finished_row { 1 + nx(2) as u32 } else { 0 },
                        spec,
                    });
                }
                // KV rows come after the committed update: a finished row's
                // full row includes the just-accepted token's slot.
                let mut kv: Vec<KvRow> = Vec::new();
                for (i, &c) in batch.iter().enumerate() {
                    if rows[i].finished
                        || (last_mode == crate::types::MODE_PREFILL
                            && !last_decode_like.contains(&c)
                            && committed[c as usize] > 0)
                    {
                        kv.push(KvRow {
                            core_idx: c,
                            row: vals[c as usize][..committed[c as usize] as usize].to_vec(),
                        });
                    }
                }
                let events = core.apply_result(&rows, &kv);
                check_fuzz_events(&core, &events, &allocated, &mut evicted, &mut owned, &committed);
                // Drift guard: the fuzz's committed tracking must follow the
                // core's, else the ownership model goes blind.
                for &c in &batch {
                    if committed[c as usize] != core.req_committed_len(c) {
                        panic!(
                            "iter {iter}: committed drift on req {c}: fuzz {} core {}; last_mode={last_mode} decode_like={:?} accepted: {:?}",
                            committed[c as usize],
                            core.req_committed_len(c),
                            last_decode_like,
                            rows.iter().map(|r| r.accepted.len()).collect::<Vec<_>>()
                        );
                    }
                }
            }

            // 2. Ingress (0..=3 reqs).
            for _ in 0..(if nx(4) == 0 { 0 } else { 1 + nx(3) }) {
                let bank = &banks[nx(8) as usize];
                let min_n = 8u64.min(bank.len() as u64);
                let n = (min_n + nx(bank.len() as u64 - min_n + 1)).min(bank.len() as u64) as usize;
                let mut origin = bank[..n].to_vec();
                if nx(10) == 0 {
                    for _ in 0..(1 + nx(24)) {
                        origin.push(1_000_000 + next_tok as i64);
                        next_tok += 1;
                    }
                }
                let req = IngressReq {
                    rid: next_pool as u64 + 1,
                    pool_idx: next_pool,
                    origin,
                    max_new_tokens: 2 + nx(47) as u32,
                    priority: nx(4) as i32,
                    arrival_seq: next_arrival,
                    routing_key: nx(2),
                    ignore_eos: nx(20) == 0,
                };
                next_arrival += 1;
                next_pool += 1;
                let c = core.ingest(vec![req])[0];
                // Slot recycling: a recycled core idx keeps its stale
                // vectors — reset them. Pool idxs are never recycled, so
                // `owned` (keyed by pool) just grows.
                while vals.len() <= c as usize {
                    vals.push(Vec::new());
                    committed.push(0);
                }
                vals[c as usize].clear();
                committed[c as usize] = 0;
                owned.push(HashSet::new());
            }

            // 3. Random abort of a WAITING request — the driver's contract
            //    (Python aborts from the waiting queue; running reqs finish
            //    through results).
            if nx(10) == 0 {
                let w = core.waiting().to_vec();
                if !w.is_empty() {
                    let events = core.drop_request(w[nx(w.len() as u64) as usize]);
                    check_fuzz_events(&core, &events, &allocated, &mut evicted, &mut owned, &committed);
                }
            }

            // 4. Plan.
            let evictable_live = core.tree().evictable_size();
            let pre_waiting: Vec<u32> = core.waiting().to_vec();
            let pre_running: Vec<u32> = core.running().to_vec();
            let pre_chunked = core.chunked_idx();
            let committed_before: Vec<u32> =
                pre_running.iter().map(|&c| core.req_committed_len(c)).collect();
            avail = (avail + nx(601) as i64 - 300).clamp(0, 4096);
            if nx(7) == 0 {
                avail = 0; // memory pressure
            }
            let e = StepEnv {
                allocator_avail_tokens: avail as u32,
                tree_evictable_tokens: evictable_live.max(0) as u32,
                num_allocatable_reqs: if nx(10) == 0 { nx(3) as u32 } else { u32::MAX },
                batch_is_full,
                mixed_chunk_allowed: nx(7) != 0,
            };
            let out = core.plan(&e);
            batch_is_full = out.plan.batch_is_full;
            last_mode = out.plan.mode;
            last_decode_like = if out.plan.prefill.as_ref().is_some_and(|p| p.mixed) {
                pre_running
                    .iter()
                    .copied()
                    .filter(|c| !core.req_finished(*c))
                    .collect()
            } else {
                HashSet::new()
            };

            // 5. Plan invariants.
            let pool_budget = avail + evictable_live.max(0);
            if let Some(p) = &out.plan.prefill {
                assert!(
                    p.extend_tokens as i64 <= pool_budget,
                    "prefill extends {} tokens > free pool {pool_budget} (avail {avail}, evictable {evictable_live})",
                    p.extend_tokens
                );
            }
            if let Some(d) = &out.plan.decode {
                let freed: i64 = d
                    .retract
                    .iter()
                    .chain(d.abort.iter())
                    .map(|&i| committed_before[i as usize] as i64)
                    .sum();
                assert!(
                    d.alloc_decode_pages as i64 * page as i64 <= pool_budget + freed,
                    "decode alloc {} pages ({}) > free pool {pool_budget} + retracted {freed}",
                    d.alloc_decode_pages,
                    d.alloc_decode_pages as i64 * page as i64
                );
                assert!(
                    d.evict_tokens as i64 <= evictable_live.max(0),
                    "decode evicts {} tokens but only {evictable_live} are evictable",
                    d.evict_tokens
                );
            }

            // 6. Row-value bookkeeping (diff-based: policy reordering is
            //    opaque, so read committed lengths after the plan).
            for c in &pre_waiting {
                let after = core.req_committed_len(*c);
                if after > 0 {
                    let pool = core.req_pool_idx(*c);
                    for _ in vals[*c as usize].len() as u32..after {
                        let v = next_val;
                        next_val += 1;
                        allocated.insert(v);
                        vals[*c as usize].push(v);
                    }
                    // [0, prefix_len) is tree-backed (row views of tree
                    // positions); only the row-allocated tail is owned.
                    let prefix = core.req_prefix_len(*c);
                    for off in prefix.min(after)..after {
                        owned[pool as usize].insert(off);
                    }
                }
                committed[*c as usize] = after;
            }
            if let Some(c) = pre_chunked {
                let after = core.req_committed_len(c);
                let before = committed[c as usize];
                if after > before {
                    let pool = core.req_pool_idx(c);
                    for off in before..after {
                        owned[pool as usize].insert(off);
                        let v = next_val;
                        next_val += 1;
                        allocated.insert(v);
                        vals[c as usize].push(v);
                    }
                }
                committed[c as usize] = after;
            }

            // 7. Queue hygiene: no finished req queued anywhere; sets
            //    disjoint.
            let mut queued: Vec<u32> = core.waiting().to_vec();
            queued.extend_from_slice(core.running());
            queued.extend_from_slice(core.pending_merge());
            if let Some(c) = core.chunked_idx() {
                queued.push(c);
            }
            let mut seen: HashSet<u32> = HashSet::new();
            for c in &queued {
                assert!(!core.req_finished(*c), "finished req {c} still queued");
                if !seen.insert(*c) {
                    panic!(
                        "req {c} in two queues: waiting={} running={} pending={} chunked={:?}",
                        core.waiting().contains(c),
                        core.running().contains(c),
                        core.pending_merge().contains(c),
                        core.chunked_idx()
                    );
                }
            }

            check_fuzz_events(&core, &out.events, &allocated, &mut evicted, &mut owned, &committed);

            snaps.push(FuzzSnap {
                plan: out.plan,
                events: out.events,
                waiting: core.waiting().to_vec(),
                running: core.running().to_vec(),
                chunked: core.chunked_idx(),
                tree: (
                    core.tree().evictable_size(),
                    core.tree().protected_size(),
                    core.tree().total_size(),
                ),
                ntr: core.new_token_ratio(),
            });
        }
        snaps
    }

    /// Fuzz-side event model for `fuzz_once`: `allocated`/`evicted` track
    /// pooled KV values (each may be evicted at most once); `owned` tracks,
    /// per row, which token offsets are backed by the req's own allocation.
    fn check_fuzz_events(
        core: &SchedulerCore,
        events: &[Event],
        allocated: &HashSet<i64>,
        evicted: &mut HashSet<i64>,
        owned: &mut [HashSet<u32>],
        committed: &[u32],
    ) {
        let dump = |msg: &str| -> ! {
            let mut lines: Vec<String> = vec![msg.to_string()];
            lines.push(format!(
                "waiting={:?} running={:?} chunked={:?}",
                core.waiting(),
                core.running(),
                core.chunked_idx()
            ));
            for &c in core.waiting().iter().chain(core.running().iter()) {
                lines.push(format!(
                    "req {c}: pool={} core_committed={} fuzz_committed={} protected={} prefix={} finished={}",
                    core.req_pool_idx(c),
                    core.req_committed_len(c),
                    committed.get(c as usize).copied().unwrap_or(u32::MAX),
                    core.req_protected_len(c),
                    core.req_prefix_len(c),
                    core.req_finished(c)
                ));
            }
            panic!("{}", lines.join("\n"));
        };
        for ev in events {
            match ev {
                Event::Evict { values } => {
                    for run in values {
                        for &v in run {
                            assert!(
                                allocated.contains(&v),
                                "evicted a value that was never allocated: {v}"
                            );
                            assert!(
                                evicted.insert(v),
                                "double-free of KV page {v} via Evict"
                            );
                        }
                    }
                }
                Event::FreeSegments { pool_idx, ranges } => {
                    let o = &mut owned[*pool_idx as usize];
                    for (s, e) in ranges {
                        for off in *s..*e {
                            if !o.remove(&off) {
                                dump(&format!(
                                    "double-free of row segment [{s}, {e}) of pool {pool_idx}: offset {off} not row-owned"
                                ));
                            }
                        }
                    }
                }
                Event::StashRowWrite {
                    pool_idx,
                    start,
                    new_indices,
                } => {
                    // The row now views tree values over [start, start+len).
                    // The duplicate part [protected, new_prefix) was freed
                    // by the paired FreeSegments; the absorbed tail
                    // ([new_prefix, ...)) moved into the tree, so neither is
                    // row-owned any more.
                    let o = &mut owned[*pool_idx as usize];
                    for off in *start..*start + new_indices.len() as u32 {
                        o.remove(&off);
                    }
                }
                Event::Finished { .. } => {}
            }
        }
    }

    #[test]
    fn drop_waiting_matched_prefix_underflows_lock() {
        let mut core = SchedulerCore::new(cfg(), EvictionPolicy::Lru);
        let a = core.ingest(vec![IngressReq {
            rid: 1,
            pool_idx: 0,
            origin: origin(64, 1),
            max_new_tokens: 4,
            priority: 0,
            arrival_seq: 0,
            routing_key: 0,
            ignore_eos: false,
        }]);
        core.plan(&env());
        core.apply_result(
            &[ResultRow {
                accepted: vec![],
                finished: true,
                finish_reason: 1,
                spec: None,
            }],
            &[kv_row(a[0], 64, 400)],
        );
        // Second req shares the prefix; keep it WAITING (0 allocatable) so the
        // plan scores it (last_node = the shared node) but never locks it.
        let b = core.ingest(vec![IngressReq {
            rid: 2,
            pool_idx: 1,
            origin: origin(64, 1),
            max_new_tokens: 4,
            priority: 0,
            arrival_seq: 1,
            routing_key: 0,
            ignore_eos: false,
        }]);
        let e = StepEnv {
            num_allocatable_reqs: 0,
            ..env()
        };
        core.plan(&e);
        assert!(!core.waiting().is_empty());
        let _ = core.drop_request(b[0]);
    }

    fn cfg() -> Config {
        Config::default()
    }

    fn origin(n: u32, salt: u32) -> Vec<i64> {
        (0..n).map(|i| ((i * 31 + salt) % 1000) as i64).collect()
    }

    fn env() -> StepEnv {
        StepEnv {
            allocator_avail_tokens: 100_000,
            tree_evictable_tokens: 0,
            num_allocatable_reqs: u32::MAX,
            batch_is_full: false,
            mixed_chunk_allowed: true,
        }
    }

    fn kv_row(core_idx: u32, core_len: u32, salt: u64) -> KvRow {
        KvRow {
            core_idx,
            row: (0..core_len)
                .map(|i| salt as i64 * 1_000_000 + i as i64)
                .collect(),
        }
    }

    fn decode_result(len: usize, finished: bool) -> Vec<ResultRow> {
        (0..len)
            .map(|_| ResultRow {
                accepted: vec![7],
                finished,
                finish_reason: 0,
                spec: None,
            })
            .collect()
    }

    #[test]
    fn prefill_then_decode_flow() {
        let mut core = SchedulerCore::new(cfg(), EvictionPolicy::Lru);
        let idx = core.ingest(vec![
            IngressReq {
                rid: 1,
                pool_idx: 0,
                origin: origin(64, 1),
                max_new_tokens: 16,
                priority: 0,
                arrival_seq: 0,
                routing_key: 0,
                ignore_eos: false,
            },
            IngressReq {
                rid: 2,
                pool_idx: 1,
                origin: origin(32, 2),
                max_new_tokens: 16,
                priority: 0,
                arrival_seq: 1,
                routing_key: 0,
                ignore_eos: false,
            },
        ]);
        assert_eq!(core.waiting(), &idx[..]);

        // Prefill pass: both admitted, parked in last_batch.
        let out = core.plan(&env());
        assert_eq!(out.plan.mode, crate::types::MODE_PREFILL);
        let p = out.plan.prefill.as_ref().unwrap();
        assert_eq!(p.admitted.len(), 2);
        assert_eq!(core.last_batch(), &idx[..]);
        assert_eq!(core.waiting().len(), 0);
        assert_eq!(core.running().len(), 0); // merges on the next plan

        // Results: each req samples one token; no finishes. Both admits are
        // stashed into the tree at result time (Python
        // `maybe_cache_unfinished_req`): row rewrites, no frees yet
        // (fresh tree -> nothing to dedup).
        let events = core.apply_result(
            &decode_result(2, false),
            &[kv_row(idx[0], 64, 100), kv_row(idx[1], 32, 200)],
        );
        let stashes = events
            .iter()
            .filter(|e| matches!(e, Event::StashRowWrite { .. }))
            .count();
        assert_eq!(stashes, 2);
        assert_eq!(core.tree().total_size(), 96);

        // Next plan: merge into running, decode pass.
        let out = core.plan(&env());
        assert_eq!(out.plan.mode, crate::types::MODE_DECODE);
        assert_eq!(core.running(), &idx[..]);
        let d = out.plan.decode.as_ref().unwrap();
        assert_eq!(d.decode.len(), 2);
        assert!(d.retract.is_empty());

        // One decode step; req 1 finishes.
        let events = core.apply_result(
            &[
                ResultRow {
                    accepted: vec![8],
                    finished: false,
                    finish_reason: 0,
                    spec: None,
                },
                ResultRow {
                    accepted: vec![9],
                    finished: true,
                    finish_reason: 2,
                    spec: None,
                },
            ],
            &[kv_row(idx[0], 65, 100), kv_row(idx[1], 33, 200)],
        );
        // out_len counts the prefill-sampled token plus the decode token.
        assert_eq!(
            events,
            vec![Event::Finished {
                core_idx: idx[1],
                reason: 2,
                out_len: 2
            }]
        );
        // The finished req's KV is now in the tree.
        assert!(core.tree().total_size() >= 32);
        // It stays in running until the next plan's head-filter removes it
        // (Python `filter_batch()` runs before every plan).
        assert_eq!(core.running().len(), 2);

        // Next plan: the head-filter drops the finished req before the
        // planner sees it, so `finished_removed` is empty.
        let out = core.plan(&env());
        let d = out.plan.decode.as_ref().unwrap();
        assert_eq!(d.decode.len(), 1);
        assert!(d.finished_removed.is_empty());
        assert_eq!(core.running(), &[idx[0]]);
    }

    #[test]
    fn spec_result_updates_counters() {
        let mut core = SchedulerCore::new(cfg(), EvictionPolicy::Lru);
        let idx = core.ingest(vec![IngressReq {
            rid: 1,
            pool_idx: 0,
            origin: origin(32, 7),
            max_new_tokens: 16,
            priority: 0,
            arrival_seq: 0,
            routing_key: 0,
            ignore_eos: false,
        }]);

        // Prefill admit + sampled token.
        let out = core.plan(&env());
        assert_eq!(out.plan.mode, crate::types::MODE_PREFILL);
        core.apply_result(
            &[decode_result(1, false)[0].clone()],
            &[kv_row(idx[0], 32, 100)],
        );

        // Decode step with a settled spec row: 3 accepted tokens,
        // accept_len 4 (drafts + bonus), block/cap lens present.
        let out = core.plan(&env());
        assert_eq!(out.plan.mode, crate::types::MODE_DECODE);
        core.apply_result(
            &[ResultRow {
                accepted: vec![11, 12, 13],
                finished: false,
                finish_reason: 0,
                spec: Some(ResultSpec {
                    accept_len: 4,
                    settled: true,
                    block_accept_len: Some(2),
                    cap_len: Some(3),
                }),
            }],
            &[],
        );
        assert_eq!(core.req_out_len(idx[0]), 4); // 1 prefill + 3 spec
        let c = core.spec_counters(idx[0]).unwrap();
        assert_eq!(c.spec_verify_ct, 1);
        assert_eq!(c.spec_num_correct_drafts, 3);
        assert_eq!(c.spec_num_block_accept_tokens, 2);
        assert_eq!(c.spec_num_cap_tokens, 3);
        assert_eq!(c.correct_drafts_histogram, vec![0, 0, 0, 1]);
        assert_eq!(c.cap_lens_histogram, vec![0, 0, 0, 1]);

        // An unsettled row (retracted / pre-finished) commits nothing and
        // touches no counters.
        core.plan(&env());
        core.apply_result(
            &[ResultRow {
                accepted: Vec::new(),
                finished: false,
                finish_reason: 0,
                spec: Some(ResultSpec {
                    accept_len: 2,
                    settled: false,
                    block_accept_len: None,
                    cap_len: Some(5),
                }),
            }],
            &[],
        );
        assert_eq!(core.req_out_len(idx[0]), 4);
        let c = core.spec_counters(idx[0]).unwrap();
        assert_eq!(c.spec_verify_ct, 1);
        assert_eq!(c.spec_num_cap_tokens, 3);

        // A non-spec decode row leaves the counters alone too.
        core.plan(&env());
        core.apply_result(&decode_result(1, false), &[]);
        let c = core.spec_counters(idx[0]).unwrap();
        assert_eq!(c.spec_verify_ct, 1);
        assert_eq!(core.req_out_len(idx[0]), 5);
    }

    #[test]
    fn drop_request_waiting_and_running() {
        let mut core = SchedulerCore::new(cfg(), EvictionPolicy::Lru);
        let idx = core.ingest(vec![
            IngressReq {
                rid: 1,
                pool_idx: 0,
                origin: origin(16, 1),
                max_new_tokens: 16,
                priority: 0,
                arrival_seq: 0,
                routing_key: 0,
                ignore_eos: false,
            },
            IngressReq {
                rid: 2,
                pool_idx: 1,
                origin: origin(16, 2),
                max_new_tokens: 16,
                priority: 0,
                arrival_seq: 1,
                routing_key: 0,
                ignore_eos: false,
            },
        ]);

        // Waiting drop: no resources yet, just the abort notice.
        let events = core.drop_request(idx[0]);
        assert_eq!(
            events,
            vec![Event::Finished {
                core_idx: idx[0],
                reason: 4,
                out_len: 0
            }]
        );
        assert_eq!(core.waiting(), &[idx[1]]);

        // Only req 2 is admitted and runs to decode.
        let out = core.plan(&env());
        assert_eq!(out.plan.prefill.as_ref().unwrap().admitted.len(), 1);
        let events = core.apply_result(
            &[ResultRow {
                accepted: vec![5],
                finished: false,
                finish_reason: 0,
                spec: None,
            }],
            &[kv_row(idx[1], 16, 300)],
        );
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, Event::StashRowWrite { .. }))
                .count(),
            1
        );
        let out = core.plan(&env());
        assert_eq!(out.plan.mode, crate::types::MODE_DECODE);
        assert_eq!(core.running(), &[idx[1]]);
        assert_eq!(core.tree().protected_size(), 16);

        // Running drop: the prefill stash tree-backed the whole row
        // (protected == committed == 16), so nothing row-allocated is freed;
        // releasing the admission lock makes the 16 tokens evictable.
        let events = core.drop_request(idx[1]);
        assert_eq!(
            events,
            vec![Event::Finished {
                core_idx: idx[1],
                reason: 4,
                out_len: 1
            }]
        );
        assert!(core.running().is_empty());
        assert_eq!(core.tree().protected_size(), 0);
        assert_eq!(core.tree().evictable_size(), 16);

        // Slot recycling: the dropped slot (LIFO free stack) is reused.
        let idx2 = core.ingest(vec![IngressReq {
            rid: 3,
            pool_idx: 2,
            origin: origin(8, 3),
            max_new_tokens: 8,
            priority: 0,
            arrival_seq: 0,
            routing_key: 0,
            ignore_eos: false,
        }]);
        assert_eq!(idx2, vec![idx[1]]);
    }

    #[test]
    fn chunked_prefill_flow() {
        let c = Config {
            chunked_prefill_size: Some(4),
            ..cfg()
        };
        let mut core = SchedulerCore::new(c, EvictionPolicy::Lru);
        let idx = core.ingest(vec![IngressReq {
            rid: 1,
            pool_idx: 0,
            origin: origin(10, 1),
            max_new_tokens: 8,
            priority: 0,
            arrival_seq: 0,
            routing_key: 0,
            ignore_eos: false,
        }]);
        let c0 = idx[0];

        // Pass 1: the req is a waiting request that ENTERS chunked prefill:
        // chunk 0..4, parked as the new chunked req.
        let out = core.plan(&env());
        let p = out.plan.prefill.as_ref().unwrap();
        assert_eq!(p.admitted.len(), 1);
        assert_eq!(p.admitted[0].waiting_idx, 0);
        assert_eq!(
            (p.admitted[0].extend_start, p.admitted[0].extend_end),
            (0, 4)
        );
        assert_eq!(p.chunked, Some(0));
        assert_eq!(core.chunked_idx(), Some(c0));

        core.apply_result(
            &[ResultRow {
                accepted: vec![],
                finished: false,
                finish_reason: 0,
                spec: None,
            }],
            &[kv_row(c0, 4, 300)],
        );
        // The parked chunk stashed 0..4 into the tree.
        assert_eq!(core.tree().total_size(), 4);

        // Pass 2: the parked chunk continues as a CHUNKED_IDX entry: 4..8.
        let out = core.plan(&env());
        let p = out.plan.prefill.as_ref().unwrap();
        assert_eq!(p.admitted[0].waiting_idx, CHUNKED_IDX);
        assert_eq!(
            (p.admitted[0].extend_start, p.admitted[0].extend_end),
            (4, 8)
        );
        assert_eq!(p.chunked, Some(CHUNKED_IDX));
        core.apply_result(
            &[ResultRow {
                accepted: vec![],
                finished: false,
                finish_reason: 0,
                spec: None,
            }],
            &[kv_row(c0, 8, 300)],
        );
        assert_eq!(core.tree().total_size(), 8);

        // Pass 3: final chunk 8..10 completes; req joins running next plan.
        let out = core.plan(&env());
        let p = out.plan.prefill.as_ref().unwrap();
        assert_eq!(p.admitted[0].waiting_idx, CHUNKED_IDX);
        assert_eq!(
            (p.admitted[0].extend_start, p.admitted[0].extend_end),
            (8, 10)
        );
        assert_eq!(p.chunked, None);
        assert_eq!(core.chunked_idx(), None);
        core.apply_result(
            &[ResultRow {
                accepted: vec![42],
                finished: false,
                finish_reason: 0,
                spec: None,
            }],
            &[kv_row(c0, 10, 300)],
        );

        let out = core.plan(&env());
        assert_eq!(out.plan.mode, crate::types::MODE_DECODE);
        assert_eq!(core.running(), &[c0]);
        assert_eq!(core.tree().total_size(), 10);
    }

    #[test]
    fn decode_retract_frees_row_and_requeues() {
        let c = Config {
            page_size: 64,
            ..cfg()
        };
        let mut core = SchedulerCore::new(c, EvictionPolicy::Lru);
        let idx = core.ingest(vec![
            IngressReq {
                rid: 1,
                pool_idx: 0,
                origin: origin(64, 1),
                max_new_tokens: 16,
                priority: 0,
                arrival_seq: 0,
                routing_key: 0,
                ignore_eos: false,
            },
            IngressReq {
                rid: 2,
                pool_idx: 1,
                origin: origin(128, 2),
                max_new_tokens: 16,
                priority: 0,
                arrival_seq: 1,
                routing_key: 0,
                ignore_eos: false,
            },
        ]);
        // Prefill both (chunked off, page 64: req0 needs 1 page, req1 2).
        let out = core.plan(&env());
        assert_eq!(out.plan.mode, crate::types::MODE_PREFILL);
        core.apply_result(
            &decode_result(2, false),
            &[kv_row(idx[0], 64, 400), kv_row(idx[1], 128, 500)],
        );
        let out = core.plan(&env()); // merge -> running
        assert_eq!(out.plan.mode, crate::types::MODE_DECODE);

        // Tight pool: both page-aligned reqs need 128 tokens, none available.
        let e = StepEnv {
            allocator_avail_tokens: 0,
            tree_evictable_tokens: 0,
            num_allocatable_reqs: u32::MAX,
            batch_is_full: false,
            mixed_chunk_allowed: true,
        };
        let out = core.plan(&e);
        let d = out.plan.decode.as_ref().unwrap();
        // Length policy retracts the shorter-output req first (both out 1,
        // tie -> shorter origin... equal out_len: ascending origin puts the
        // longer-origin last, so req1 (origin 128) is retracted).
        assert_eq!(d.retract, vec![1]);
        assert_eq!(d.decode, vec![0]);
        assert!(core.waiting().contains(&idx[1]));
        assert!(core.req_retracted_stain(idx[1]));
        let frees: Vec<_> = out
            .events
            .iter()
            .filter_map(|e| match e {
                Event::FreeSegments { pool_idx, ranges } => Some((*pool_idx, ranges.clone())),
                _ => None,
            })
            .collect();
        // The prefill stash tree-backed the whole row (protected ==
        // committed == 128), so Python's retraction
        // `release_kv_cache(is_insert=False)` frees
        // [cache_protected_len, committed) = the empty range: the KV stays
        // in the tree, where it is evictable.
        assert!(frees.is_empty());
        // NTR took the post-retract estimate over the survivor:
        // (1 + 20) / (16 + 1) -> capped 1.0.
        assert_eq!(core.new_token_ratio(), 1.0);
    }
}
