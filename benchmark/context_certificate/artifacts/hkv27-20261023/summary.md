# HKV-27 — Finish-path orphan leak: root cause, fix, 45-finish validation

Branch `host-kv/main`, image `hostkv:4ccff141-25` (base lineage `4ccff141db`).
Instance `instance-hkv27` (us-central1-b, g4-standard-48 spot, RTX PRO 6000
96 GB), Qwen3.8-Flash-Next NVFP4, TP=1, NEXTN speculative (3 steps, 4 draft
tokens), tier mode keep=1700. Run 2026-10-23 13:03–13:15 UTC.

## 1. The leak

Every finished request permanently lost ~2 pool pages (128 tokens of KV
budget). `alloc_free` decayed monotonically: 91904 -> 86144 tokens over 45
finishes on the unfixed image — exactly 128 tokens per finish. At keep=1700
this capped a server life at ~31 finishes, which the HKV-24 soak had to
paper over with a recycle-when-forgotten>30 rule.

Attribution (per-call-site allocator wraps + page-id probes on images
`-21..-24`): every page leaves the free list via `alloc_extend`, and the
orphans are the final verify/draft-tail allocations that are never written
into `req_to_token`. All runtime release paths (`free`, `free_segment`) are
table-read based, so those pages are unreachable garbage — in no ledger,
referenced by no row, held by no tier structure.

## 2. The fix (two defects)

1. **Orphan sweeper** in `forget_row` (`host_kv.py`): a page that appears
   in NEITHER ledger (allocator free list, tier reserve), is held by no
   tier row, and is referenced by no `req_to_token` row across TWO
   consecutive forgets is garbage with certainty — any live owner reaches
   its page via one of those, and an in-flight verify slot is referenced
   by the next forget's scan. Stable orphans are swept back into the
   allocator's free list. The sweep is not gated on `stats_path`: it is
   the fix, not instrumentation.
2. **Per-forward dedupe** in `after_write`: the `(first-slot, size)` guard
   against multi-layer duplicate callbacks was global and leaked across
   forwards — a new request landing on its own recycled pages reproduced
   the previous forward's key and its first chunk was silently dropped
   (`rows[0]` never re-created after forget). The guard is now scoped to
   the `forward_batch` instance (fresh per forward, shared across the
   layers of one forward). Found by the unit test; a production bug, not
   a test artifact.

## 3. Validation — 45 sequential 101k finishes, one server life

`volley27.py` (45 × 101,043-token input, 512 output tokens, ignore_eos),
image `-25`. Evidence: `volley45-ledger.log` (per-forget ledger lines).

- 45/45 requests finished `length`, ~11.7 s each, no admission stall
  (old budget was ~31 finishes).
- `alloc_free` flat, oscillating 91584–92416 tokens across forgets #3–#47
  (old image: monotonic −128/finish). No decay.
- 114 orphan pages swept back over the volley (`orphan_reclaimed=114`),
  `orphan_pending` bounded at 0–13 (transients between the two-forget
  stability checks).
- 0 scheduler exceptions / tracebacks; `restore_starved=0`,
  `verify_mismatches=0`, `coverage_misses=22` (soak-normal range).

Unit lock: `test_host_kv_tier.py::FreeSegmentContamination::
test_orphan_sweeper_reclaims_after_two_forgets` and
`test_orphan_sweeper_runs_without_stats_path` (sweep without stats).

## 4. Operational consequence

The per-server finish budget is gone; the soak's recycle-on-budget rule
(HKV-24 runbook §recycle) is now a safety net rather than a requirement.
Remaining known limitation is unchanged: single-GPU 256k contexts are
memory-infeasible (upstream EP2 / two-server topology, sglang#44291).
