//! Port of `PrefillAdder` (`schedule_policy.py`) — the per-request
//! admission budget engine.
//!
//! Scope: the base MHA path. The hybrid-SWA, Mamba/GDN, DLLM, LoRA,
//! preemption, prefill-delayer, and HIP tile-budget branches are out of
//! scope for this port (documented per the plan's "fastest solution for the
//! Qwen3.8 base-MHA target"); their budget terms are all zero on that path.
//!
//! Float parity: `rem_total_token_offset` is `f64` in Python from the first
//! `offset += min(int, int) * new_token_ratio` step, and
//! `rem_total_tokens` is `int - float`; the Rust mirror keeps the same
//! operation order so `total_tokens >= rem_total_tokens` compares
//! bit-identical values. `cur_rem_token_offset` stays int throughout.

use crate::ntr::Ntr;
use crate::types::{Config, PlanReq, StepEnv};

/// Admission outcome, mirroring `AddReqResult`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddResult {
    Continue,
    NoToken,
    Other,
}

/// Tree interaction around the temporary admission lock. The shadow
/// planner uses [`NullTree`] (no tree work — Python owns it); the core
/// engine locks/unlocks its Rust tree.
pub trait AdmissionTree {
    fn temp_lock(&mut self, node: u32);
    fn temp_unlock(&mut self, node: u32);
}

/// No-op tree (shadow mode): Python performs the real `inc/dec_lock_ref`.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullTree;

impl AdmissionTree for NullTree {
    fn temp_lock(&mut self, _node: u32) {}
    fn temp_unlock(&mut self, _node: u32) {}
}

pub struct Adder<'a> {
    cfg: &'a Config,
    running: &'a [PlanReq],
    ntr: f64,
    env: &'a StepEnv,
    rem_input_tokens: i64,
    rem_chunk_tokens: Option<i64>,
    rem_total_token_offset: f64,
    cur_rem_token_offset: i64,
    can_run: Vec<u32>,
    extend: Vec<(u32, u32)>,
    /// The in-flight chunked request's extend range this pass
    /// (`add_chunked_req`); `None` when there is no chunked request.
    chunked_extend: Option<(u32, u32)>,
    /// Mirrors Python: the chunked req is pushed into `can_run_list` before
    /// the waiting-queue loop, so it counts for the "can_run empty" gates.
    chunked_pushed: bool,
    /// The waiting idx that became the in-flight chunked request this pass
    /// (`None` unless `add_one_req` took the chunked branch).
    new_chunked: Option<u32>,
    /// Admitted waiting-queue reqs, in `can_run` order (by value — the
    /// `ignore_eos` req-states walk reads the admitted snapshots, mirroring
    /// Python's walk over `can_run_list`).
    admitted_reqs: Vec<PlanReq>,
    /// `ignore_eos` + disabled-tree path state:
    /// `Vec<(tokens_left, tokens_occupied)>` sorted by `tokens_left`.
    req_states: Option<Vec<(f64, i64)>>,
}

impl<'a> Adder<'a> {
    /// `PrefillAdder.__init__` (base MHA).
    pub fn new(cfg: &'a Config, running: &'a [PlanReq], ntr: &Ntr, env: &'a StepEnv) -> Self {
        let mixed = if cfg.mixed_chunk { running.len() as i64 } else { 0 };
        // Python: `offset = mixed_bs; offset += sum([terms])` where `sum`
        // left-folds the terms over int 0 — so the running-batch fold starts
        // from 0.0 and the result is added to `mixed` as a second op. Each
        // term is `min(max_new - out, CLIP) * ratio` (no lower clamp in
        // Python's `_get_running_request_total_token_offset`).
        let mut running_offset: f64 = 0.0;
        for r in running {
            let remaining = r.max_new_tokens as i64 - r.out_len as i64;
            let clipped = remaining.min(cfg.clip_max_new_tokens as i64);
            running_offset += clipped as f64 * ntr.current();
        }
        let rem_total_token_offset = mixed as f64 + running_offset;
        Self {
            cfg,
            running,
            ntr: ntr.current(),
            env,
            rem_input_tokens: cfg.max_prefill_tokens as i64 - mixed,
            rem_chunk_tokens: cfg.chunked_prefill_size.map(|c| c as i64 - mixed),
            rem_total_token_offset,
            cur_rem_token_offset: mixed,
            can_run: Vec::new(),
            extend: Vec::new(),
            chunked_extend: None,
            chunked_pushed: false,
            new_chunked: None,
            admitted_reqs: Vec::new(),
            req_states: None,
        }
    }

    /// Whether the in-flight chunked request was re-admitted this pass
    /// (Python: it sits in `can_run_list` ahead of the waiting reqs).
    pub fn chunked_in_can_run(&self) -> bool {
        self.chunked_pushed
    }

    /// Effective `can_run_list` length, chunked req included.
    pub fn can_run_len(&self) -> usize {
        self.can_run.len() + usize::from(self.chunked_pushed)
    }

    /// Waiting-queue indices admitted this pass, in `can_run_list` order.
    pub fn can_run(&self) -> &[u32] {
        &self.can_run
    }

    /// Extend range `(start, end)` for an admitted waiting-queue index.
    pub fn extend_of(&self, waiting_idx: u32) -> Option<(u32, u32)> {
        self.can_run
            .iter()
            .position(|&i| i == waiting_idx)
            .map(|pos| self.extend[pos])
    }

    /// The in-flight chunked req's extend range this pass.
    pub fn chunked_extend(&self) -> Option<(u32, u32)> {
        self.chunked_extend
    }

    /// The req that became the in-flight chunked request this pass.
    pub fn new_chunked_req(&self) -> Option<u32> {
        self.new_chunked
    }

    fn page(&self) -> i64 {
        self.cfg.page_size as i64
    }

    /// `ceil_paged_tokens`: `-(-tokens // page_size) * page_size`.
    ///
    /// Python's `//` is floor division, so the result is exact for negative
    /// inputs too (a degenerate parked chunk can carry a negative extend
    /// length). `div_euclid` with a positive page gives floor division.
    fn ceil_paged(&self, tokens: i64) -> i64 {
        let p = self.page();
        let a = tokens.wrapping_neg();
        -(a.div_euclid(p)) * p
    }

    fn total_avail(&self) -> i64 {
        self.env.allocator_avail_tokens as i64 + self.env.tree_evictable_tokens as i64
    }

    /// `rem_total_tokens` property (int - float offset).
    fn rem_total_tokens(&self) -> f64 {
        self.total_avail() as f64 - self.rem_total_token_offset
    }

    /// `cur_rem_tokens` property (int - int).
    fn cur_rem_tokens(&self) -> i64 {
        self.total_avail() - self.cur_rem_token_offset
    }

    /// `budget_state`.
    pub fn budget_state(&self) -> AddResult {
        let no_token = self.rem_total_tokens() <= 0.0 || self.cur_rem_tokens() <= 0;
        if no_token {
            return AddResult::NoToken;
        }
        if self.rem_input_tokens <= 0 {
            return AddResult::Other;
        }
        if self.rem_chunk_tokens.is_some_and(|rem| rem <= 0) {
            return AddResult::Other;
        }
        AddResult::Continue
    }

    /// `_update_prefill_budget` (base MHA; `retracted_stain` only affects
    /// Python-side logging, so it is not carried here).
    fn update_budget(&mut self, prefix_len: u32, extend_input_len: i64, max_new_tokens: i64) {
        let _ = prefix_len; // log_* accounting only
        let extend = self.ceil_paged(extend_input_len);
        let page_overhead = self.page();
        self.rem_total_token_offset += (extend + max_new_tokens + page_overhead) as f64;
        self.cur_rem_token_offset += extend + page_overhead;
        self.rem_input_tokens -= extend;
        if let Some(rem) = self.rem_chunk_tokens.as_mut() {
            *rem -= extend;
        }
    }

    /// `add_chunked_req` (base MHA): the in-flight chunked request is
    /// re-admitted every pass. Returns the extend range and whether the
    /// chunk is still truncated.
    pub fn add_chunked_req(&mut self, req: &PlanReq) -> ((u32, u32), bool) {
        // `int(self.rem_total_tokens)`: truncation toward zero
        // (Rust `f64 as i64` truncates toward zero, saturating on overflow).
        let rem_total = self.rem_total_tokens() as i64;
        let mut rem = self.rem_chunk_tokens.unwrap_or(i64::MAX).min(rem_total);
        if rem <= 0 {
            // "the chunked_req must be added to the list; otherwise it
            // leaks" — fall back to rem_chunk_tokens (may be 0).
            rem = self.rem_chunk_tokens.unwrap_or(0);
        }

        let cand = req.fill_len() as i64 - req.prefix_len as i64;
        let truncated = cand > rem;
        // `min(cand, rem)` — may be negative when the fallback
        // `rem_chunk_tokens` itself is negative (mixed running_bs exceeds the
        // chunked size). The budget charge keeps the exact value; the extend
        // range is clamped to a zero-length range (Python's degenerate parked
        // chunk runs with an empty extend input in that case).
        let new_len = cand.min(rem);
        let start = req.prefix_len;
        let end = start + new_len.max(0) as u32;
        self.chunked_extend = Some((start, end));
        self.chunked_pushed = true;
        // Python pushes the chunked req into `can_run_list`, so the
        // ignore-eos req-states walk counts it.
        self.admitted_reqs.push(*req);

        let max_new = if !truncated {
            (req.max_new_tokens as i64).min(self.cfg.clip_max_new_tokens as i64)
        } else {
            0
        };
        self.update_budget(0, new_len, max_new);
        ((start, end), truncated)
    }

    /// `add_one_req` (base MHA). `tree` performs the temporary admission
    /// lock; `waiting_idx` is the request's index in the caller's queue.
    pub fn add_one_req(
        &mut self,
        tree: &mut dyn AdmissionTree,
        waiting_idx: u32,
        req: &PlanReq,
    ) -> AddResult {
        if self.cfg.prefill_max_requests.is_some_and(|m| self.can_run_len() as u32 >= m) {
            return AddResult::Other;
        }
        if req.ignore_eos && self.cfg.disable_tree {
            return self.add_one_req_ignore_eos(waiting_idx, req);
        }

        let max_new =
            ((req.max_new_tokens as i64 - req.out_len as i64).max(0))
                .min(self.cfg.clip_max_new_tokens as i64);
        let cand = req.fill_len() as i64 - req.prefix_len as i64;
        let total_tokens = cand + max_new + self.page();

        let real_input = self.ceil_paged(cand - req.host_hit_length as i64);

        if total_tokens as f64 >= self.rem_total_tokens() {
            return AddResult::NoToken;
        }

        let chunk_tokens_limit = self.rem_chunk_tokens;
        // (hybrid-SWA chunk-cap branch: out of scope, base MHA)

        if self.rem_chunk_tokens.is_none()
            && self.can_run_len() != 0
            && real_input >= self.rem_input_tokens
        {
            // Without chunked prefill: honor max_prefill_tokens once the
            // first request is in; the first request is always accepted.
            // A re-admitted chunked req counts as "already in", matching
            // Python's `can_run_list`.
            return AddResult::Other;
        }

        // Temporary admission lock (released in `finally` by Python).
        tree.temp_lock(req.last_node);
        let result = self.admit_locked(req, waiting_idx, chunk_tokens_limit);
        tree.temp_unlock(req.last_node);
        result
    }

    /// Body of `add_one_req` between the lock acquire/release.
    fn admit_locked(&mut self, req: &PlanReq, waiting_idx: u32, chunk_tokens_limit: Option<i64>) -> AddResult {
        // `rem_total_tokens` may have decreased after the lock.
        let total_tokens = {
            let max_new = ((req.max_new_tokens as i64 - req.out_len as i64).max(0))
                .min(self.cfg.clip_max_new_tokens as i64);
            (req.fill_len() as i64 - req.prefix_len as i64) + max_new + self.page()
        };
        if total_tokens as f64 >= self.rem_total_tokens() {
            return AddResult::NoToken;
        }

        let input_tokens = self.ceil_paged(req.fill_len() as i64 - req.prefix_len as i64);
        if self.rem_chunk_tokens.is_none()
            && self.can_run_len() != 0
            && input_tokens >= self.rem_input_tokens
        {
            return AddResult::Other;
        }

        if chunk_tokens_limit.is_none() || input_tokens <= chunk_tokens_limit.unwrap_or(i64::MAX)
        {
            // Non-chunked prefill: the whole sequence commits this iter.
            let start = req.prefix_len;
            let end = req.fill_len();
            self.can_run.push(waiting_idx);
            self.admitted_reqs.push(*req);
            self.extend.push((start, end));
            self.update_budget(
                req.prefix_len,
                input_tokens,
                (req.max_new_tokens as i64).min(self.cfg.clip_max_new_tokens as i64),
            );
        } else {
            // Chunked prefill.
            let mut trunc_len = chunk_tokens_limit.unwrap_or(i64::MAX);
            trunc_len = trunc_len / self.page() * self.page();
            if trunc_len <= 0 {
                return AddResult::Other;
            }
            if let Some(align) = self.cfg.truncation_align_size {
                let align = align as i64;
                if trunc_len < align {
                    return AddResult::Other;
                }
                trunc_len = align * (trunc_len / align);
            }
            let mut now_input_len = trunc_len + req.prefix_len as i64;
            now_input_len = now_input_len / self.page() * self.page();
            trunc_len = now_input_len - req.prefix_len as i64;
            if trunc_len <= 0 {
                return AddResult::Other;
            }
            let start = req.prefix_len;
            let end = start + trunc_len as u32;
            self.can_run.push(waiting_idx);
            self.admitted_reqs.push(*req);
            self.extend.push((start, end));
            self.new_chunked = Some(waiting_idx);
            self.update_budget(req.prefix_len, trunc_len, 0);
        }
        self.budget_state()
    }

    /// `add_one_req_ignore_eos` (base MHA, no SWA/DLLM/tile budget).
    fn add_one_req_ignore_eos(&mut self, waiting_idx: u32, req: &PlanReq) -> AddResult {
        let cand = req.fill_len() as i64 - req.prefix_len as i64;
        let paged_input = self.ceil_paged(cand);
        // `min(self.cur_rem_tokens, self.rem_total_tokens)`: int vs float.
        let gate = (self.cur_rem_tokens() as f64).min(self.rem_total_tokens());
        if paged_input as f64 > gate {
            return AddResult::NoToken;
        }

        let add_req_state = |r: &PlanReq, states: &mut Vec<(f64, i64)>, insert_sort: bool| {
            let ratio = if r.ignore_eos { 1.0 } else { self.ntr };
            let tokens_left = r.max_new_tokens as f64 * ratio - r.out_len as f64;
            let tokens_occupied = (r.origin_len + r.out_len) as i64;
            if tokens_left <= 0.0 {
                return;
            }
            if !insert_sort {
                states.push((tokens_left, tokens_occupied));
            } else {
                // Python's insert walk: first `i` with
                // `tokens_left <= states[i][0]`; when none, `i` stays at
                // `len - 1` (reproduced exactly, including that corner).
                let i = states
                    .iter()
                    .position(|s| tokens_left <= s.0)
                    .unwrap_or(if states.is_empty() { 0 } else { states.len() - 1 });
                states.insert(i, (tokens_left, tokens_occupied));
            }
        };

        if self.req_states.is_none() {
            let mut states: Vec<(f64, i64)> = Vec::new();
            add_req_state(req, &mut states, false);
            for r in self.running {
                add_req_state(r, &mut states, false);
            }
            for r in &self.admitted_reqs {
                add_req_state(r, &mut states, false);
            }
            states.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("finite"));
            self.req_states = Some(states);
        } else {
            add_req_state(req, self.req_states.as_mut().unwrap(), true);
        }

        {
            let cur_rem = self.cur_rem_tokens() - self.ceil_paged(cand);
            let mut tokens_freed = 0i64;
            let states = self.req_states.as_ref().unwrap();
            for (i, &(tokens_left, tokens_occupied)) in states.iter().enumerate() {
                let bs = (states.len() - i) as f64;
                let min_free = cur_rem as f64 + tokens_freed as f64 - tokens_left * bs;
                if min_free <= 1.0 * bs {
                    return AddResult::NoToken;
                }
                tokens_freed += tokens_occupied;
            }
        }

        if self.rem_chunk_tokens.is_none() || cand <= self.rem_chunk_tokens.unwrap_or(i64::MAX) {
            // Non-chunked.
            let start = req.prefix_len;
            let end = req.fill_len();
            self.can_run.push(waiting_idx);
            self.admitted_reqs.push(*req);
            self.extend.push((start, end));
            self.update_budget(
                0,
                end as i64 - start as i64,
                (req.max_new_tokens as i64).min(self.cfg.clip_max_new_tokens as i64),
            );
        } else {
            // `rem_chunk_tokens` is Some here (the `is_none()` arm is the
            // non-chunked branch above).
            let rem = self.rem_chunk_tokens.unwrap_or(0);
            if rem <= 0 {
                return AddResult::Other;
            }
            // Python asserts `len(req.prefix_indices) == 0` on this path.
            debug_assert_eq!(req.prefix_len, 0);
            let start = req.prefix_len;
            let end = start + rem as u32;
            self.can_run.push(waiting_idx);
            self.admitted_reqs.push(*req);
            self.extend.push((start, end));
            self.new_chunked = Some(waiting_idx);
            self.update_budget(0, rem, 0);
        }

        self.budget_state()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(avail: u32, evict: u32) -> StepEnv {
        StepEnv {
            allocator_avail_tokens: avail,
            tree_evictable_tokens: evict,
            num_allocatable_reqs: u32::MAX,
            batch_is_full: false,
            mixed_chunk_allowed: false,
        }
    }

    fn ntr() -> Ntr {
        Ntr::from_config(&Config::default())
    }

    fn waiting(origin: u32, prefix: u32, max_new: u32) -> PlanReq {
        PlanReq {
            origin_len: origin,
            out_len: 0,
            committed_len: origin,
            prefix_len: prefix,
            last_node: u32::MAX,
            max_new_tokens: max_new,
            ..Default::default()
        }
    }

    #[test]
    fn first_req_always_admitted_without_chunking() {
        let cfg = Config {
            page_size: 64,
            max_prefill_tokens: 100, // smaller than the req's input
            chunked_prefill_size: None,
            ..Default::default()
        };
        let w = [waiting(500, 0, 100)];
        let e = env(10_000, 0);
        let mut adder = Adder::new(&cfg, &[], &ntr(), &e);
        let mut tree = NullTree;
        // can_run empty -> the first prefill request is always PUSHED to
        // can_run_list even though ceil_paged(500) = 512 > rem_input = 100.
        // `add_one_req` still returns `budget_state()` at the end: with
        // rem_input now -412 that is OTHER, which stops the caller's loop
        // after this admission (the req itself still runs).
        assert_eq!(adder.add_one_req(&mut tree, 0, &w[0]), AddResult::Other);
        assert_eq!(adder.can_run(), &[0]);
        assert_eq!(adder.extend_of(0), Some((0, 500)));
    }

    #[test]
    fn second_req_blocked_by_max_prefill_tokens() {
        let cfg = Config {
            page_size: 64,
            max_prefill_tokens: 100,
            chunked_prefill_size: None,
            ..Default::default()
        };
        let e = env(10_000, 0);
        let a = waiting(90, 0, 10); // ceil_paged(90) = 128
        let b = waiting(20, 0, 10); // ceil_paged(20) = 64
        let w = [a, b];
        let mut adder = Adder::new(&cfg, &[], &ntr(), &e);
        let mut tree = NullTree;
        // a is pushed (can_run was empty) but takes ceil(90)=128 > 100 =
        // rem_input, so the returned budget state is already OTHER.
        assert_eq!(adder.add_one_req(&mut tree, 0, &w[0]), AddResult::Other);
        // b is blocked: rem_input is now -28, so the input gate fires.
        assert_eq!(adder.add_one_req(&mut tree, 1, &w[1]), AddResult::Other);
        assert_eq!(adder.can_run(), &[0]);
    }

    #[test]
    fn no_token_when_total_exceeds_budget() {
        let cfg = Config {
            page_size: 1,
            max_prefill_tokens: 100_000,
            chunked_prefill_size: None,
            ..Default::default()
        };
        // avail + evictable = 50; total_tokens = 40 + 100 + 1 > 50.
        let e = env(50, 0);
        let w = [waiting(40, 0, 100)];
        let mut adder = Adder::new(&cfg, &[], &ntr(), &e);
        let mut tree = NullTree;
        assert_eq!(adder.add_one_req(&mut tree, 0, &w[0]), AddResult::NoToken);
        assert!(adder.can_run().is_empty());
    }

    #[test]
    fn chunked_truncation_and_align() {
        let cfg = Config {
            page_size: 64,
            chunked_prefill_size: Some(100),
            truncation_align_size: Some(64),
            ..Default::default()
        };
        let e = env(100_000, 0);
        let w = [waiting(1000, 0, 100)];
        let mut adder = Adder::new(&cfg, &[], &ntr(), &e);
        let mut tree = NullTree;
        assert_eq!(adder.add_one_req(&mut tree, 0, &w[0]), AddResult::Continue);
        // trunc_len = 100//64*64 = 64; align 64: ok; now = (64+0)//64*64 =
        // 64; trunc = 64 - 0.
        assert_eq!(adder.extend_of(0), Some((0, 64)));
        assert_eq!(adder.new_chunked_req(), Some(0));
    }

    #[test]
    fn chunked_continuation_completes() {
        let cfg = Config {
            page_size: 1,
            chunked_prefill_size: Some(10),
            ..Default::default()
        };
        let e = env(100_000, 0);
        let mut adder = Adder::new(&cfg, &[], &ntr(), &e);
        // The chunked req: prefix advanced to 10, fill 25 -> cand 15 > 10.
        let chunked = PlanReq {
            origin_len: 25,
            committed_len: 25,
            prefix_len: 10,
            ..Default::default()
        };
        let ((start, end), truncated) = adder.add_chunked_req(&chunked);
        assert_eq!((start, end), (10, 20));
        assert!(truncated);

        // Next pass: prefix 20, fill 25 -> cand 5 <= 10: complete.
        let mut adder2 = Adder::new(&cfg, &[], &ntr(), &e);
        let chunked2 = PlanReq {
            origin_len: 25,
            committed_len: 25,
            prefix_len: 20,
            ..Default::default()
        };
        let ((start, end), truncated) = adder2.add_chunked_req(&chunked2);
        assert_eq!((start, end), (20, 25));
        assert!(!truncated);
    }

    #[test]
    fn running_offset_charges_f64() {
        // One running req: max_new 100, out 0 -> min(100, 4096) * 0.7 = 70.
        // Budget: avail 100, evict 0 -> rem_total = 100 - 70 = 30.
        // Candidate: cand 20, max_new 10 -> total = 31 >= 30 -> NO_TOKEN.
        let cfg = Config::default();
        let running = vec![PlanReq {
            max_new_tokens: 100,
            committed_len: 50,
            origin_len: 50,
            ..Default::default()
        }];
        let e = env(100, 0);
        let r1 = waiting(20, 0, 10);
        let r2 = waiting(15, 0, 10);
        let mut adder = Adder::new(&cfg, &running, &ntr(), &e);
        let mut tree = NullTree;
        assert_eq!(adder.add_one_req(&mut tree, 0, &r1), AddResult::NoToken);
        // cand 15 -> total 26 < 30 -> admitted (Python uses >= on the
        // boundary, so 30 would still be NO_TOKEN).
        assert_eq!(adder.add_one_req(&mut tree, 1, &r2), AddResult::Continue);
    }

    #[test]
    fn budget_state_no_token_and_other() {
        let cfg = Config {
            max_prefill_tokens: 0,
            chunked_prefill_size: Some(10),
            ..Default::default()
        };
        let e = env(1000, 0);
        let adder = Adder::new(&cfg, &[], &ntr(), &e);
        // rem_input_tokens = 0 -> OTHER (not NO_TOKEN: token budgets fine).
        assert_eq!(adder.budget_state(), AddResult::Other);

        let cfg2 = Config {
            max_prefill_tokens: 100,
            chunked_prefill_size: Some(0),
            ..Default::default()
        };
        let adder2 = Adder::new(&cfg2, &[], &ntr(), &e);
        // rem_chunk_tokens = 0 -> OTHER.
        assert_eq!(adder2.budget_state(), AddResult::Other);

        // Exhausted total budget -> NO_TOKEN.
        let cfg3 = Config {
            max_prefill_tokens: 100,
            ..Default::default()
        };
        let e3 = env(0, 0);
        let adder3 = Adder::new(&cfg3, &[], &ntr(), &e3);
        assert_eq!(adder3.budget_state(), AddResult::NoToken);
    }

    #[test]
    fn ignore_eos_disabled_tree_admits() {
        let cfg = Config {
            disable_tree: true,
            max_prefill_tokens: 100_000,
            chunked_prefill_size: None,
            ..Default::default()
        };
        let e = env(10_000, 0);
        let r = PlanReq {
            origin_len: 100,
            committed_len: 100,
            max_new_tokens: 1000,
            ignore_eos: true,
            last_node: u32::MAX,
            ..Default::default()
        };
        let w = [r];
        let mut adder = Adder::new(&cfg, &[], &ntr(), &e);
        assert_eq!(adder.add_one_req(&mut NullTree, 0, &w[0]), AddResult::Continue);
        assert_eq!(adder.extend_of(0), Some((0, 100)));
    }
}
