//! Speculative-decoding (spec-v2) accept-run resolution — plan §9 / M6.
//!
//! Port of `batch_result_processor._resolve_spec_v2_tokens`: from the
//! stride-padded `next_token_ids` buffer + `accept_lens` it computes each
//! req's committed accepted run and the batch/session spec counters.
//!
//! Boundaries (what stays in Python):
//!
//! - the **grammar FSM** (the per-req xgrammar object): it runs over the
//!   raw slice in `advance_grammar_fsm` and memoizes the grammar-legal run
//!   as `result.grammar_retained_tokens`; that run arrives here as
//!   `SpecRow::grammar_retained` and replaces the raw slice when present.
//! - the **adaptive controller** (`on_verify_complete_cpu`): fed from the
//!   returned `num_correct_drafts_per_req`.
//! - the **KV move** (`move_accept_tokens_to_target_kvcache`): torch/GPU.
//!
//! Slice semantics mirror Python list slicing exactly (clamped, never
//! panics on a short tail), so a differential test can drive both sides
//! from the same inputs and compare byte-for-byte.

use std::error::Error;
use std::fmt;

/// One req's spec-v2 inputs for this step.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SpecRow {
    /// `accept_lens[i]` — accepted run length *before* grammar
    /// truncation (drafts + bonus; ≥ 1 in practice).
    pub accept_len: u32,
    /// The req was retracted before this step → nothing settles.
    pub retracted: bool,
    /// The req was already finished before this step → nothing settles.
    pub finished: bool,
    /// The grammar-legal run (`result.grammar_retained_tokens[i]`);
    /// `None` when the req has no grammar (or never settled).
    pub grammar_retained: Option<Vec<i64>>,
    /// `block_accept_lens[i]` — `None` when the batch carries no block
    /// lens at all (the column is absent, not zero).
    pub block_accept_len: Option<u32>,
    /// `cap_lens[i]` — `None` when the batch carries no cap lens.
    pub cap_len: Option<u32>,
}

/// One req's settled output.
#[derive(Debug, Clone, PartialEq)]
pub struct SpecRun {
    /// The committed run (grammar-truncated when applicable). Empty for
    /// unsettled rows (retracted / pre-finished): Python's
    /// `predict_tokens` still carries the raw slice there, but nothing
    /// downstream consumes it (the overlap skip path drops the row
    /// before any commit).
    pub tokens: Vec<i64>,
    /// Whether this req settled its counters this step
    /// (`!retracted && !finished` at resolve time).
    pub settled: bool,
}

/// Batch-level resolution (`_resolve_spec_v2_tokens` return + result
/// fields).
#[derive(Debug, Clone, PartialEq)]
pub struct SpecResolution {
    /// Per-req committed run, batch order.
    pub runs: Vec<SpecRun>,
    /// `sum(accept_lens) - n_reqs` (bonus excluded).
    pub num_correct_drafts: u32,
    /// Per-req `accept_lens[i] - 1` (the adaptive-controller input).
    pub num_correct_drafts_per_req: Vec<u32>,
    /// `sum(block_accept_lens)` (0 when the column is absent).
    pub num_block_accept_tokens: u32,
    /// `sum(cap_lens)` (0 when the column is absent).
    pub num_cap_tokens: u32,
}

/// `stride == 0` (a caller bug; Python asserts a non-None
/// `speculative_num_draft_tokens` and never uses 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecError {
    StrideZero,
}

impl fmt::Display for SpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpecError::StrideZero => f.write_str("spec stride is zero"),
        }
    }
}

impl Error for SpecError {}

/// Resolve the per-req accepted runs + batch counters.
///
/// `next_token_ids` is the flat stride-padded buffer: req `i`'s draft
/// slots are `[i * stride, i * stride + stride)`. Out-of-range reads are
/// clamped like Python slices.
pub fn resolve_spec_runs(
    next_token_ids: &[i64],
    stride: u32,
    rows: &[SpecRow],
) -> Result<SpecResolution, SpecError> {
    if stride == 0 {
        return Err(SpecError::StrideZero);
    }
    let n = rows.len();
    let mut runs = Vec::with_capacity(n);
    let mut per_req = Vec::with_capacity(n);
    let mut num_correct_drafts = 0u32;
    let mut num_block_accept_tokens = 0u32;
    let mut num_cap_tokens = 0u32;

    for (i, row) in rows.iter().enumerate() {
        let settled = !row.retracted && !row.finished;
        let start = i * stride as usize;
        let end = next_token_ids.len().min(start + row.accept_len as usize);
        let raw = if start < end {
            &next_token_ids[start..end]
        } else {
            &[]
        };

        let tokens = if !settled {
            Vec::new()
        } else if let Some(retained) = &row.grammar_retained {
            retained.clone()
        } else {
            raw.to_vec()
        };
        runs.push(SpecRun { tokens, settled });

        // Batch fields mirror the Python result fields exactly — they are
        // computed over ALL rows (settled or not).
        let correct = row.accept_len.saturating_sub(1);
        per_req.push(correct);
        num_correct_drafts += correct;
        if let Some(b) = row.block_accept_len {
            num_block_accept_tokens += b;
        }
        if let Some(c) = row.cap_len {
            num_cap_tokens += c;
        }
    }

    Ok(SpecResolution {
        runs,
        num_correct_drafts,
        num_correct_drafts_per_req: per_req,
        num_block_accept_tokens,
        num_cap_tokens,
    })
}

/// Per-req spec counters — the `Req` fields at
/// `schedule_batch.py:1159` (`spec_verify_ct`, ...) plus the two
/// growable histograms.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpecCounters {
    pub spec_verify_ct: u32,
    pub spec_num_correct_drafts: u64,
    pub spec_num_block_accept_tokens: u64,
    pub spec_num_cap_tokens: u64,
    /// `spec_correct_drafts_histogram`: index = correct-draft count.
    pub correct_drafts_histogram: Vec<u64>,
    /// `spec_cap_lens_histogram`: index = cap len.
    pub cap_lens_histogram: Vec<u64>,
}

impl SpecCounters {
    /// One settled spec step (the commit block of
    /// `_resolve_spec_v2_tokens`): counters + histogram updates.
    pub fn update(
        &mut self,
        correct_drafts: u32,
        block_accept_len: Option<u32>,
        cap_len: Option<u32>,
    ) {
        self.spec_verify_ct += 1;
        self.spec_num_correct_drafts += u64::from(correct_drafts);
        Self::bump(&mut self.correct_drafts_histogram, correct_drafts as usize);
        if let Some(b) = block_accept_len {
            self.spec_num_block_accept_tokens += u64::from(b);
        }
        if let Some(c) = cap_len {
            self.spec_num_cap_tokens += u64::from(c);
            Self::bump(&mut self.cap_lens_histogram, c as usize);
        }
    }

    /// `update_spec_*_histogram`: grow with zeros up to the index, then
    /// `+= 1`.
    fn bump(hist: &mut Vec<u64>, idx: usize) {
        if hist.len() <= idx {
            hist.resize(idx + 1, 0);
        }
        hist[idx] += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(accept_len: u32) -> SpecRow {
        SpecRow {
            accept_len,
            ..Default::default()
        }
    }

    #[test]
    fn basic_resolution() {
        // stride 4, 3 reqs: buffer rows [10,11,12,13, 20,21,22,23, 30,31,32,33]
        let buf: Vec<i64> = [10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33].to_vec();
        let rows = vec![row(2), row(4), row(1)];
        let r = resolve_spec_runs(&buf, 4, &rows).unwrap();
        assert_eq!(
            r.runs.iter().map(|x| x.tokens.clone()).collect::<Vec<_>>(),
            vec![vec![10, 11], vec![20, 21, 22, 23], vec![30]]
        );
        assert!(r.runs.iter().all(|x| x.settled));
        assert_eq!(r.num_correct_drafts, 7 - 3);
        assert_eq!(r.num_correct_drafts_per_req, vec![1, 3, 0]);
        assert_eq!(r.num_block_accept_tokens, 0);
        assert_eq!(r.num_cap_tokens, 0);
    }

    #[test]
    fn grammar_retained_replaces_raw_slice() {
        let buf = vec![5, 6, 7, 8];
        let rows = vec![SpecRow {
            accept_len: 4,
            grammar_retained: Some(vec![5, 6]),
            ..Default::default()
        }];
        let r = resolve_spec_runs(&buf, 4, &rows).unwrap();
        assert_eq!(r.runs[0].tokens, vec![5, 6]);
        // The batch fields still use the pre-truncation accept_len.
        assert_eq!(r.num_correct_drafts, 3);
        assert_eq!(r.num_correct_drafts_per_req, vec![3]);
    }

    #[test]
    fn unsettled_rows_commit_nothing() {
        let buf = vec![5, 6, 7, 8];
        let rows = vec![
            SpecRow {
                retracted: true,
                ..row(4)
            },
            SpecRow {
                finished: true,
                grammar_retained: Some(vec![5]),
                ..row(2)
            },
        ];
        let r = resolve_spec_runs(&buf, 4, &rows).unwrap();
        assert_eq!(r.runs[0].tokens, Vec::<i64>::new());
        assert_eq!(r.runs[1].tokens, Vec::<i64>::new());
        assert!(!r.runs[0].settled && !r.runs[1].settled);
        // Batch fields are still computed over all rows: 6 - 2.
        assert_eq!(r.num_correct_drafts, 4);
        assert_eq!(r.num_correct_drafts_per_req, vec![3, 1]);
    }

    #[test]
    fn clamps_like_python_slices() {
        // accept_len past the buffer end → clamped (Python slice semantics).
        let buf = vec![1, 2, 3];
        let rows = vec![row(5), row(2)];
        let r = resolve_spec_runs(&buf, 4, &rows).unwrap();
        assert_eq!(r.runs[0].tokens, vec![1, 2, 3]);
        assert_eq!(r.runs[1].tokens, Vec::<i64>::new()); // start 4 >= len
    }

    #[test]
    fn block_and_cap_totals() {
        let rows = vec![
            SpecRow {
                block_accept_len: Some(2),
                cap_len: Some(3),
                ..row(2)
            },
            SpecRow {
                block_accept_len: Some(0),
                cap_len: Some(1),
                ..row(1)
            },
        ];
        let r = resolve_spec_runs(&[], 1, &rows).unwrap();
        assert_eq!(r.num_block_accept_tokens, 2);
        assert_eq!(r.num_cap_tokens, 4);
    }

    #[test]
    fn stride_zero_is_an_error() {
        assert_eq!(
            resolve_spec_runs(&[1], 0, &[row(1)]),
            Err(SpecError::StrideZero)
        );
    }

    #[test]
    fn counters_and_histograms() {
        let mut c = SpecCounters::default();
        c.update(2, Some(1), Some(4));
        c.update(5, None, Some(4));
        c.update(1, None, None);
        assert_eq!(c.spec_verify_ct, 3);
        assert_eq!(c.spec_num_correct_drafts, 8);
        assert_eq!(c.spec_num_block_accept_tokens, 1);
        assert_eq!(c.spec_num_cap_tokens, 8);
        // correct-drafts: idx2 once, idx5 once, idx1 once.
        assert_eq!(c.correct_drafts_histogram, vec![0, 1, 1, 0, 0, 1]);
        // cap-lens: idx4 twice.
        assert_eq!(c.cap_lens_histogram, vec![0, 0, 0, 0, 2]);
    }
}
