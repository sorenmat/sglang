//! Compact value types at the Rust/Python boundary.
//!
//! The planner is a pure decision engine: Python owns the request queues,
//! the radix tree, and the allocator, and passes compact snapshots in; the
//! planner returns a [`BatchPlan`] Python applies with a handful of torch
//! ops. The stateful [`crate::core::SchedulerCore`] reuses the same engine
//! over owned state.
//!
//! Determinism contract (plan-for-plan parity with the Python scheduler):
//! - all budget math is int-only except the new-token-ratio terms, which are
//!   `f64` in the same operation order as Python (`min(int, int) * ratio`);
//! - stable sorts with explicit tie-breaks (priority, arrival seq, rid);
//! - the same policy-degradation rules as `schedule_policy.py`
//!   (LPM -> FCFS when the waiting queue exceeds 128, disabled tree -> FCFS).

/// Forward-mode discriminants for [`BatchPlan`].
pub const MODE_NONE: u8 = 0;
pub const MODE_PREFILL: u8 = 1;
pub const MODE_DECODE: u8 = 2;

/// One request's scheduling-relevant snapshot.
///
/// Indices (`waiting_idx` / running list position) identify the request in
/// the caller's snapshot; no rid strings cross the boundary, so no hashing
/// is involved.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlanReq {
    /// `req_pool_idx` (row in `req_to_token_pool`).
    pub pool_idx: u32,
    /// `len(origin_input_ids)`.
    pub origin_len: u32,
    /// `len(output_ids)` — accepted output tokens so far.
    pub out_len: u32,
    /// Committed KV length. Running requests: `kv_committed_len` at plan
    /// time (the next decode token lands at this offset). Waiting requests:
    /// `origin_len + out_len` (`full_untruncated_fill_len`).
    pub committed_len: u32,
    /// `len(prefix_indices)` — matched tree prefix, page-floored.
    pub prefix_len: u32,
    /// Tree node handle (`RadixTree` `NodeId` in core mode; ignored,
    /// conventionally `u32::MAX`, in shadow mode).
    pub last_node: u32,
    /// Request priority (0 when unset).
    pub priority: i32,
    /// Monotonic arrival tick (queue-entry order), FCFS tie-break.
    pub arrival_seq: u64,
    /// `sampling_params.max_new_tokens`.
    pub max_new_tokens: u32,
    /// Routing key (0 = none), `routing-key` policy.
    pub routing_key: u64,
    /// `sampling_params.ignore_eos`.
    pub ignore_eos: bool,
    /// Finished (any reason) — filtered out of decode batches.
    pub finished: bool,
    /// Set when the request was retracted before; the admission loop
    /// accounts it differently in logs only (budget math is unchanged).
    pub retracted_stain: bool,
    /// Host-tier hit length (0 for the base device-only tree).
    pub host_hit_length: u32,
}

impl PlanReq {
    /// `len(full_untruncated_fill_ids)`.
    pub fn fill_len(&self) -> u32 {
        self.origin_len + self.out_len
    }
}

/// CPU-mirror environment snapshot for one plan call. Every field is known
/// without a GPU sync (allocator `available_size()` is a tensor shape,
/// `evictable_size()` is tree bookkeeping).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StepEnv {
    /// `token_to_kv_pool_allocator.available_size()` (tokens).
    pub allocator_avail_tokens: u32,
    /// `tree_cache.evictable_size()` (tokens).
    pub tree_evictable_tokens: u32,
    /// `scheduler.get_num_allocatable_reqs(running_bs)` — the
    /// min(pp budget, req-pool free slots) gate.
    pub num_allocatable_reqs: u32,
    /// `running_batch.batch_is_full` carried from the previous iteration
    /// (set by preemption/NO_TOKEN/allocatable gates).
    pub batch_is_full: bool,
    /// Python-side gate for mixed-style chunked prefill (no logprobs, no
    /// input embeds, no beam rows in the running batch).
    pub mixed_chunk_allowed: bool,
}

/// Admission-policy names, matching `schedule_policy.py`
/// (`CacheAwarePolicy` + `CacheAgnosticPolicy`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    Fcfs,
    Lpm,
    DfsWeight,
    Lof,
    Random,
    RoutingKey,
}

impl Policy {
    /// Parse the `--schedule-policy` string.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "fcfs" => Ok(Self::Fcfs),
            "lpm" => Ok(Self::Lpm),
            "dfs-weight" => Ok(Self::DfsWeight),
            "lof" => Ok(Self::Lof),
            "random" => Ok(Self::Random),
            "routing-key" => Ok(Self::RoutingKey),
            other => Err(format!("unknown schedule_policy: {other}")),
        }
    }

    /// The Python-side equivalent of `isinstance(policy, CacheAwarePolicy)`.
    pub fn cache_aware(&self) -> bool {
        matches!(self, Self::Lpm | Self::DfsWeight)
    }
}

/// Static scheduling configuration — the ~15 `server_args` flags the
/// planner needs. Everything is immutable for the lifetime of the engine.
#[derive(Debug, Clone)]
pub struct Config {
    pub policy: Policy,
    /// Paged allocator page size.
    pub page_size: u32,
    /// `max_prefill_tokens` (the `rem_input_tokens` budget).
    pub max_prefill_tokens: u32,
    /// `chunked_prefill_size`; `None` disables chunked prefill.
    pub chunked_prefill_size: Option<u32>,
    /// `is_mixed_chunk` (running decode tokens join the prefill batch).
    pub mixed_chunk: bool,
    /// `enable_priority_scheduling`.
    pub priority_scheduling: bool,
    /// `schedule_low_priority_values_first` (priority_sign = 1 iff true).
    pub low_priority_values_first: bool,
    /// `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION` (default 4096).
    pub clip_max_new_tokens: u32,
    /// `IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD` (default 32).
    pub in_batch_check_threshold: u32,
    /// `IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD` (default 32).
    pub in_batch_deprioritize_threshold: u32,
    /// `get_schedule().prefill_max_requests`.
    pub prefill_max_requests: Option<u32>,
    /// `truncation_align_size` (deterministic-inference flashinfer alignment).
    pub truncation_align_size: Option<u32>,
    /// LPM degrades to FCFS when `waiting.len() > lpm_queue_degrade_at`.
    pub lpm_queue_degrade_at: u32,
    /// Seed for the `random` policy's deterministic shuffle.
    pub random_seed: u64,
    /// Tree cache disabled (`--disable-radix-cache`): LPM/DFS policies fall
    /// back to FCFS, ignore-eos requests take the `req_states` path.
    pub disable_tree: bool,
    /// `SGLANG_INIT_NEW_TOKEN_RATIO` (default 0.7).
    pub ntr_init_raw: f64,
    /// `get_schedule().schedule_conservativeness` (default 1.0).
    pub schedule_conservativeness: f64,
    /// `SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR` (default 0.1).
    pub ntr_min_factor: f64,
    /// `SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS` (default 600).
    pub ntr_decay_steps: u32,
    /// `SGLANG_RETRACT_DECODE_STEPS` (default 20).
    pub retract_decode_steps: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            policy: Policy::Lpm,
            page_size: 1,
            max_prefill_tokens: 16_384,
            chunked_prefill_size: None,
            mixed_chunk: false,
            priority_scheduling: false,
            low_priority_values_first: false,
            clip_max_new_tokens: 4096,
            in_batch_check_threshold: 32,
            in_batch_deprioritize_threshold: 32,
            prefill_max_requests: None,
            truncation_align_size: None,
            lpm_queue_degrade_at: 128,
            random_seed: 0,
            disable_tree: false,
            ntr_init_raw: 0.7,
            schedule_conservativeness: 1.0,
            ntr_min_factor: 0.1,
            ntr_decay_steps: 600,
            retract_decode_steps: 20,
        }
    }
}

impl Config {
    pub fn priority_sign(&self) -> i32 {
        if self.low_priority_values_first {
            1
        } else {
            -1
        }
    }

    /// Port of `SchedulePolicy._validate_and_adjust_policy`: with a
    /// disabled tree, cache-aware policies degrade to FCFS.
    pub fn active_policy(&self) -> Policy {
        if self.disable_tree && self.policy.cache_aware() {
            Policy::Fcfs
        } else {
            self.policy
        }
    }
}

/// One admitted (or chunk-truncated) request in the prefill plan.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AdmitReq {
    /// Index into the waiting snapshot (or `CHUNKED_IDX` when the entry
    /// carries the continuing chunked request).
    pub waiting_idx: u32,
    /// Page-floored matched prefix at admission time.
    pub prefix_len: u32,
    /// Extend range `[extend_start, extend_end)` in token offsets;
    /// `extend_start == prefix_len`.
    pub extend_start: u32,
    pub extend_end: u32,
}

/// Sentinel `waiting_idx` for the continuing chunked request (it is not in
/// the waiting queue).
pub const CHUNKED_IDX: u32 = u32::MAX;

/// The prefill half of a [`BatchPlan`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PrefillPlan {
    /// Admitted requests, in execution order (chunked continuation first,
    /// matching Python's `can_run_list`).
    pub admitted: Vec<AdmitReq>,
    /// The chunked request is still mid-prompt after this batch
    /// (`waiting_idx`), or `None` when the chunk completed.
    pub chunked: Option<u32>,
    /// Mixed-style chunked prefill: merge the running decode reqs into this
    /// batch (`mix_with_running`).
    pub mixed: bool,
    /// `sum(extend_end - extend_start)`.
    pub extend_tokens: u32,
    /// `get_num_new_pages(seq_lens=extend_end, prefix_lens=extend_start)`
    /// summed over admitted reqs — feeds `alloc_extend(num_new_pages=..)`.
    pub alloc_extend_pages: u32,
}

/// The decode half of a [`BatchPlan`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DecodePlan {
    /// Running reqs that survive this iteration, in original batch order.
    pub decode: Vec<u32>,
    /// Running reqs removed by `filter_batch` (finished), in original order.
    pub finished_removed: Vec<u32>,
    /// Running reqs retracted (requeued with `is_retracted=True`),
    /// retraction order.
    pub retract: Vec<u32>,
    /// Running reqs aborted (last-req OOM), retraction order.
    pub abort: Vec<u32>,
    /// Tree tokens to evict before the decode alloc (shortfall only,
    /// mirrors `check_decode_mem`'s `evict_from_tree_cache`).
    pub evict_tokens: u32,
    /// `get_num_new_pages(decode=True)` over the surviving reqs — feeds
    /// `alloc_decode`.
    pub alloc_decode_pages: u32,
    /// New-token-ratio after this iteration (retract estimate or decay).
    pub ntr: f64,
}

/// One scheduler iteration's decision: exactly one of `prefill` / `decode`
/// is `Some` (or neither, when idle).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BatchPlan {
    pub mode: u8,
    /// Carried forward to the next iteration's `StepEnv.batch_is_full`.
    pub batch_is_full: bool,
    pub prefill: Option<PrefillPlan>,
    pub decode: Option<DecodePlan>,
}
