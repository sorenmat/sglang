//! The scheduling decision engine.
//!
//! [`plan_next_batch`] is the pure function both front ends build on:
//! - the stateless shadow `Planner` (Phase 2 / `planner` flag): Python owns
//!   the queues, tree, and allocator and passes snapshots in each
//!   iteration; the plan is applied (and diffed, in shadow mode) by Python;
//! - the stateful `SchedulerCore` (Phase 3 / `core` flag): the same engine
//!   over owned queues + the Rust radix tree, with result bookkeeping
//!   folded into each step.
//!
//! It mirrors `get_next_batch_to_run` for the base-MHA, non-PP, non-DP,
//! non-spec path: prefill first (`_get_new_batch_prefill_raw` minus the
//! LoRA/HiSparse/DLLM/grammar branches), else decode
//! (`update_running_batch` minus beam/DLLM/mamba).

use crate::adder::{Adder, AddResult, AdmissionTree, NullTree};
use crate::ntr::Ntr;
use crate::policy;
use crate::types::{
    AdmitReq, BatchPlan, Config, DecodePlan, PlanReq, PrefillPlan, StepEnv, CHUNKED_IDX,
    MODE_DECODE, MODE_NONE, MODE_PREFILL,
};

/// One iteration's decision.
///
/// - `waiting` / `running`: queue snapshots (running in batch order).
/// - `chunked`: the in-flight chunked request's snapshot, if any.
/// - `scores` / `deprioritized`: per-waiting-req LPM scores and the
///   in-batch dedup set (Python-computed in shadow mode; core mode passes
///   the tree-derived values).
/// - `iter`: monotonic plan counter (drives the deterministic random
///   shuffle).
#[allow(clippy::too_many_arguments)]
pub fn plan_next_batch(
    cfg: &Config,
    ntr: &Ntr,
    waiting: &[PlanReq],
    running: &[PlanReq],
    chunked: Option<&PlanReq>,
    scores: &[u32],
    deprioritized: &[bool],
    env: &StepEnv,
    iter: u64,
) -> BatchPlan {
    let mut tree = NullTree;
    plan_next_batch_with_tree(
        cfg, ntr, waiting, running, chunked, scores, deprioritized, env, iter, &mut tree,
    )
}

/// The decision engine over an admission tree. Shadow mode passes
/// [`NullTree`] (Python owns the tree and performs the temporary locks);
/// the core engine passes its live tree.
#[allow(clippy::too_many_arguments)]
pub fn plan_next_batch_with_tree(
    cfg: &Config,
    ntr: &Ntr,
    waiting: &[PlanReq],
    running: &[PlanReq],
    chunked: Option<&PlanReq>,
    scores: &[u32],
    deprioritized: &[bool],
    env: &StepEnv,
    iter: u64,
    tree: &mut dyn AdmissionTree,
) -> BatchPlan {
    // Prefill gate, mirroring `_get_new_batch_prefill_raw`:
    // `(batch_is_full or waiting empty) and chunked is None -> no prefill`.
    let prefill_gate =
        (env.batch_is_full || waiting.is_empty()) && chunked.is_none();
    // `get_num_allocatable_reqs <= 0 and chunked is None -> batch_is_full`.
    let alloc_gate = env.num_allocatable_reqs == 0 && chunked.is_none();

    let mut batch_is_full = env.batch_is_full || alloc_gate;

    let prefill: Option<PrefillPlan> = if prefill_gate || alloc_gate {
        None
    } else {
        plan_prefill(
            cfg,
            ntr,
            waiting,
            running,
            chunked,
            scores,
            deprioritized,
            env,
            iter,
            &mut batch_is_full,
            tree,
        )
    };

    if prefill.as_ref().is_some_and(|p| !p.admitted.is_empty()) {
        return BatchPlan {
            mode: MODE_PREFILL,
            batch_is_full,
            prefill,
            decode: None,
        };
    }

    // Decode path: `update_running_batch`. Python skips it entirely when the
    // running batch is empty (`ret = None`), leaving `batch_is_full`
    // untouched.
    let decode = if running.is_empty() {
        None
    } else {
        plan_decode(cfg, ntr, running, env)
    };
    // `update_running_batch`: after `filter_batch`, an emptied batch resets
    // `batch_is_full` (and returns early); later, any shrank batch (finished
    // filtered out, or retracted/aborted reqs dropped) resets it too:
    // `if batch.batch_size() < initial_bs: batch.batch_is_full = False`.
    let decode_shrunk = match &decode {
        None => false,
        Some(d) => {
            d.decode.is_empty()
                || !d.finished_removed.is_empty()
                || !d.retract.is_empty()
                || !d.abort.is_empty()
        }
    };
    BatchPlan {
        mode: match &decode {
            None => MODE_NONE,
            Some(d) => {
                if d.decode.is_empty() {
                    MODE_NONE
                } else {
                    MODE_DECODE
                }
            }
        },
        batch_is_full: if decode_shrunk { false } else { batch_is_full },
        prefill: None,
        decode,
    }
}

/// `get_new_batch_prefill` body (base MHA). Returns `None` when the pass
/// admits nothing (Python: `can_run_list` empty).
///
/// `tree` performs the temporary admission lock (the shadow planner passes
/// [`NullTree`]; the core engine passes its live RadixTree). Candidates must
/// already carry a fresh `prefix_len`/`last_node` (the shadow planner's
/// Python side and the core's scoring block both call the equivalent of
/// `init_next_round_input` before this pass).
#[allow(clippy::too_many_arguments)]
pub(crate) fn plan_prefill(
    cfg: &Config,
    ntr: &Ntr,
    waiting: &[PlanReq],
    running: &[PlanReq],
    chunked: Option<&PlanReq>,
    scores: &[u32],
    deprioritized: &[bool],
    env: &StepEnv,
    iter: u64,
    batch_is_full: &mut bool,
    tree: &mut dyn AdmissionTree,
) -> Option<PrefillPlan> {
    let order = policy::order_waiting(cfg, waiting, running, scores, deprioritized, iter);

    let mut adder = Adder::new(cfg, running, ntr, env);

    let mut chunked_extend: Option<(u32, u32)> = None;
    let mut chunked_still: bool = false;
    if let Some(c) = chunked {
        let (range, truncated) = adder.add_chunked_req(c);
        chunked_extend = Some(range);
        chunked_still = truncated;
    }

    for &idx in &order {
        // The per-req allocatable-reqs gate (running_bs is constant here:
        // the running batch is not mutated during the prefill pass). The
        // in-flight chunked req counts toward `can_run_list` in Python.
        if adder.can_run_len() as u32 >= env.num_allocatable_reqs {
            *batch_is_full = true;
        }
        if *batch_is_full {
            break;
        }
        let res = adder.add_one_req(tree, idx, &waiting[idx as usize]);
        if res != AddResult::Continue {
            if res == AddResult::NoToken {
                // base (non-hi-cache) arm: `running_batch.batch_is_full = True`
                *batch_is_full = true;
            }
            break;
        }
    }

    // Python: `can_run_list` is non-empty whenever the chunked req was
    // re-admitted (it is pushed first), so a chunked continuation always
    // yields a prefill batch even with an empty waiting queue.
    if adder.can_run().is_empty() && chunked_extend.is_none() {
        return None;
    }

    let mut admitted: Vec<AdmitReq> = Vec::new();
    let mut extend_tokens: u32 = 0;
    let mut alloc_pages: u32 = 0;
    if let Some((s, e)) = chunked_extend {
        admitted.push(AdmitReq {
            waiting_idx: CHUNKED_IDX,
            prefix_len: s,
            extend_start: s,
            extend_end: e,
        });
        extend_tokens += e - s;
        alloc_pages += alloc_pages_for(cfg, s, e);
    }
    for &idx in adder.can_run() {
        let (s, e) = adder.extend_of(idx).expect("admitted req has a range");
        let r = &waiting[idx as usize];
        admitted.push(AdmitReq {
            waiting_idx: idx,
            prefix_len: r.prefix_len,
            extend_start: s,
            extend_end: e,
        });
        extend_tokens += e - s;
        alloc_pages += alloc_pages_for(cfg, s, e);
    }

    let chunked = if chunked_still {
        // The continuing chunked req: Python keeps it parked.
        Some(CHUNKED_IDX)
    } else {
        // A new request entered chunked prefill this pass (or None).
        adder.new_chunked_req()
    };

    let mixed = cfg.mixed_chunk
        && !running.is_empty()
        && env.mixed_chunk_allowed
        && !admitted.is_empty();

    Some(PrefillPlan {
        admitted,
        chunked,
        mixed,
        extend_tokens,
        alloc_extend_pages: alloc_pages,
    })
}

/// `get_num_new_pages(seq_lens=end, prefix_lens=start)` for one extend
/// range: `ceil(end/page) - ceil(start/page)`.
fn alloc_pages_for(cfg: &Config, start: u32, end: u32) -> u32 {
    let p = cfg.page_size as i64;
    let after = (end as i64 + p - 1) / p;
    let before = (start as i64 + p - 1) / p;
    (after - before).max(0) as u32
}

/// `update_running_batch` planning: filter, decode-mem check (with tree
/// eviction of the shortfall), the retract loop, and the decode alloc
/// pages.
///
/// The pool model is exact for the base MHA path: `check_decode_mem`
/// evicts only the shortfall from the tree, and once the tree's evictable
/// pool is exhausted (which is the only case retraction triggers), the
/// per-recheck `evict_from_tree_cache` calls in the retract loop are no-ops,
/// so the pool grows purely by the released KV of each retracted req.
pub(crate) fn plan_decode(cfg: &Config, ntr: &Ntr, running: &[PlanReq], env: &StepEnv) -> Option<DecodePlan> {
    // filter_batch: drop finished reqs (Python also resets batch_is_full
    // when the batch shrinks — handled by the caller's batch_is_full out).
    let mut decode: Vec<u32> = Vec::with_capacity(running.len());
    let mut finished_removed: Vec<u32> = Vec::new();
    for (i, r) in running.iter().enumerate() {
        if r.finished {
            finished_removed.push(i as u32);
        } else {
            decode.push(i as u32);
        }
    }
    if decode.is_empty() {
        return Some(DecodePlan {
            decode: Vec::new(),
            finished_removed,
            retract: Vec::new(),
            abort: Vec::new(),
            evict_tokens: 0,
            alloc_decode_pages: 0,
            ntr: ntr.current(),
        });
    }

    let page = cfg.page_size as i64;
    let total_pool = |avail: i64, evictable: i64| avail + evictable;

    // check_decode_mem on the full set: evict only the shortfall from the
    // tree, then the pool must cover `page * count(kv_len % page == 0)`.
    let required = |selected: &[u32]| -> i64 {
        selected
            .iter()
            .filter(|&&i| running[i as usize].committed_len as i64 % page == 0)
            .count() as i64
            * page
    };

    let mut avail = env.allocator_avail_tokens as i64;
    let mut evictable = env.tree_evictable_tokens as i64;
    let need = required(&decode);
    // `evict_from_tree_cache` (standard allocator arm): evict the shortfall.
    let evict = (need - avail).max(0).min(evictable);
    avail += evict;
    evictable -= evict;

    let fits = |avail: i64, evictable: i64, selected: &[u32]| {
        total_pool(avail, evictable) >= required(selected)
    };

    let mut plan = DecodePlan {
        decode: Vec::new(),
        finished_removed,
        retract: Vec::new(),
        abort: Vec::new(),
        evict_tokens: evict as u32,
        alloc_decode_pages: 0,
        ntr: ntr.current(),
    };

    if fits(avail, evictable, &decode) {
        plan.decode = decode;
        // No retraction: `new_token_ratio_tracker.decay_step()`.
        plan.ntr = ntr.next_after_decay();
    } else {
        // retract_decode: order most-preferred-first, pop from the end.
        let mut selected: Vec<u32> = if cfg.priority_scheduling {
            // `retraction_policy == "priority"`:
            // key = (priority * -priority_sign, *length_key), reverse=True.
            let sign = cfg.priority_sign();
            let mut v = decode.clone();
            v.sort_by_key(|&i| {
                let r = &running[i as usize];
                (
                    -(r.priority * -sign),
                    -(r.out_len as i64),
                    r.origin_len as i64,
                )
            });
            v
        } else {
            // length policy: key = (len(output), -len(origin)), reverse=True.
            let mut v = decode.clone();
            v.sort_by_key(|&i| {
                let r = &running[i as usize];
                (-(r.out_len as i64), r.origin_len as i64)
            });
            v
        };
        // NOTE: Python sorts by `(out_len, -origin_len)` reverse=True ==
        // descending out_len then ascending origin_len; the key above
        // encodes that as a single tuple sort.

        let mut retracted: Vec<u32> = Vec::new();
        while !fits(avail, evictable, &selected) && selected.len() > 1 {
            let worst = *selected.last().unwrap();
            selected.pop();
            // Releasing the req frees its full KV to the combined pool:
            // the suffix returns to the allocator, the cached prefix becomes
            // tree-evictable after the unlock. Net: +kv_len tokens.
            let kv = running[worst as usize].committed_len as i64;
            avail += kv;
            retracted.push(worst);
        }
        if selected.len() <= 1 && !fits(avail, evictable, &selected) {
            // Even the last remaining request cannot fit: abort it.
            if let Some(last) = selected.pop() {
                plan.abort.push(last);
            }
        }
        plan.decode = selected;
        plan.retract = retracted;
        // NTR: the estimate over the surviving reqs.
        let out_lens: Vec<u32> = plan.decode.iter().map(|&i| running[i as usize].out_len).collect();
        let max_news: Vec<u32> = plan
            .decode
            .iter()
            .map(|&i| running[i as usize].max_new_tokens)
            .collect();
        plan.ntr = Ntr::estimate_after_retract(&out_lens, &max_news, cfg.retract_decode_steps);
    }

    plan.alloc_decode_pages = plan
        .decode
        .iter()
        .filter(|&&i| running[i as usize].committed_len as i64 % page == 0)
        .count() as u32;
    Some(plan)
}

impl BatchPlan {
    /// The admitted (prefill) extend token count, or 0.
    pub fn extend_tokens(&self) -> u32 {
        self.prefill.as_ref().map(|p| p.extend_tokens).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Config {
        Config::default()
    }
    fn ntr() -> Ntr {
        Ntr::from_config(&Config::default())
    }
    fn env() -> StepEnv {
        StepEnv {
            allocator_avail_tokens: 1_000_000,
            tree_evictable_tokens: 0,
            num_allocatable_reqs: u32::MAX,
            batch_is_full: false,
            mixed_chunk_allowed: true,
        }
    }

    fn waiting(origin: u32, prefix: u32) -> PlanReq {
        PlanReq {
            origin_len: origin,
            committed_len: origin,
            prefix_len: prefix,
            last_node: u32::MAX,
            max_new_tokens: 128,
            ..Default::default()
        }
    }

    #[test]
    fn prefill_admits_in_score_order() {
        let c = cfg();
        let waiting = vec![waiting(100, 0), waiting(80, 0), waiting(60, 0)];
        let running: Vec<PlanReq> = vec![];
        let scores = [100u32, 80, 60]; // all fully cached -> high LPM scores
        let plan = plan_next_batch(
            &c,
            &ntr(),
            &waiting,
            &running,
            None,
            &scores,
            &[false; 3],
            &env(),
            0,
        );
        assert_eq!(plan.mode, MODE_PREFILL);
        let p = plan.prefill.unwrap();
        // Admission order follows LPM: 0, 1, 2.
        let idx: Vec<u32> = p.admitted.iter().map(|a| a.waiting_idx).collect();
        assert_eq!(idx, vec![0, 1, 2]);
        assert_eq!(p.extend_tokens, 240);
    }

    #[test]
    fn idle_when_nothing_to_do() {
        let c = cfg();
        let plan = plan_next_batch(&c, &ntr(), &[], &[], None, &[], &[], &env(), 0);
        assert_eq!(plan.mode, MODE_NONE);
        assert!(plan.prefill.is_none() && plan.decode.is_none());
    }

    #[test]
    fn decode_plan_filters_finished_and_counts_pages() {
        let c = Config {
            page_size: 64,
            ..cfg()
        };
        // page 64; running reqs with committed 63, 64, 65 -> pages needed:
        // 64 % 64 == 0 -> 1 page; 63, 65 -> 0.
        let running = vec![
            PlanReq {
                committed_len: 63,
                out_len: 13,
                origin_len: 50,
                ..Default::default()
            },
            PlanReq {
                committed_len: 64,
                out_len: 14,
                origin_len: 50,
                ..Default::default()
            },
            PlanReq {
                committed_len: 65,
                out_len: 15,
                origin_len: 50,
                finished: true,
                ..Default::default()
            },
        ];
        let plan = plan_next_batch(&c, &ntr(), &[], &running, None, &[], &[], &env(), 1);
        assert_eq!(plan.mode, MODE_DECODE);
        let d = plan.decode.unwrap();
        assert_eq!(d.decode, vec![0, 1]);
        assert_eq!(d.finished_removed, vec![2]);
        assert_eq!(d.alloc_decode_pages, 1);
        assert!(d.retract.is_empty());
    }

    #[test]
    fn decode_evicts_shortfall_and_retracts() {
        let c = Config {
            page_size: 64,
            ..cfg()
        };
        let e = StepEnv {
            allocator_avail_tokens: 0,
            tree_evictable_tokens: 100,
            num_allocatable_reqs: u32::MAX,
            ..Default::default()
        };
        // page 64; reqs at committed 64 and 128 (both page-aligned) ->
        // required = 128 tokens. avail 0 -> evict min(128, 100) = 100 ->
        // pool 100 < 128 -> retract the shortest-output req.
        let running = vec![
            PlanReq {
                committed_len: 64,
                out_len: 54,
                origin_len: 10,
                ..Default::default()
            },
            PlanReq {
                committed_len: 128,
                out_len: 118,
                origin_len: 10,
                max_new_tokens: 512,
                ..Default::default()
            },
        ];
        let plan = plan_next_batch(&c, &ntr(), &[], &running, None, &[], &[], &e, 1);
        let d = plan.decode.unwrap();
        assert_eq!(d.evict_tokens, 100);
        // Length policy pops the least-preferred (shortest output) first:
        // req 0 (out 54) is retracted; req 1 stays.
        assert_eq!(d.retract, vec![0]);
        assert_eq!(d.decode, vec![1]);
        // NTR is the post-retract estimate over the surviving reqs:
        // (118 + 20*1) / (512 + 1).
        assert!((d.ntr - 138.0 / 513.0).abs() < 1e-12);
    }

    #[test]
    fn decode_aborts_last_req_when_pool_cannot_cover() {
        let c = cfg();
        let e = StepEnv {
            allocator_avail_tokens: 0,
            tree_evictable_tokens: 0,
            num_allocatable_reqs: u32::MAX,
            ..Default::default()
        };
        // page 64; single req at committed 0 needs 64; pool is empty.
        let running = vec![PlanReq {
            committed_len: 0,
            out_len: 0,
            origin_len: 10,
            ..Default::default()
        }];
        let plan = plan_next_batch(&c, &ntr(), &[], &running, None, &[], &[], &e, 1);
        let d = plan.decode.unwrap();
        assert_eq!(d.abort, vec![0]);
        assert!(d.decode.is_empty());
    }

    #[test]
    fn batch_is_full_blocks_prefill() {
        let c = cfg();
        let e = StepEnv {
            batch_is_full: true,
            ..env()
        };
        let waiting = vec![waiting(100, 0)];
        let plan = plan_next_batch(&c, &ntr(), &waiting, &[], None, &[100], &[false], &e, 0);
        assert_eq!(plan.mode, MODE_NONE);
        assert!(plan.batch_is_full);
    }

    #[test]
    fn chunked_continuation_runs_prefill_even_when_waiting_empty() {
        let c = Config {
            chunked_prefill_size: Some(10),
            ..cfg()
        };
        let chunked = PlanReq {
            origin_len: 100,
            committed_len: 100,
            prefix_len: 10,
            last_node: u32::MAX,
            max_new_tokens: 128,
            ..Default::default()
        };
        let plan = plan_next_batch(&c, &ntr(), &[], &[], Some(&chunked), &[], &[], &env(), 0);
        assert_eq!(plan.mode, MODE_PREFILL);
        let p = plan.prefill.unwrap();
        assert_eq!(p.admitted.len(), 1);
        assert_eq!(p.admitted[0].waiting_idx, CHUNKED_IDX);
        assert_eq!((p.admitted[0].extend_start, p.admitted[0].extend_end), (10, 20));
        assert_eq!(p.chunked, Some(CHUNKED_IDX));
    }

    #[test]
    fn ntr_decays_without_retract() {
        let c = cfg();
        let mut n = ntr();
        let running = vec![PlanReq {
            committed_len: 1,
            out_len: 1,
            origin_len: 10,
            ..Default::default()
        }];
        let e = env();
        let plan = plan_next_batch(&c, &n, &[], &running, None, &[], &[], &e, 0);
        let d0 = plan.decode.unwrap().ntr;
        n.decay_step();
        assert_eq!(d0, n.current());
    }
}
