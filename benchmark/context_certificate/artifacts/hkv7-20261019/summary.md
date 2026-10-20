# HKV-7 (multi-row decode + concurrency) — 2026-10-19, instance-20261019-hkv7

**Three results: (1) multi-row gather is implemented and coverage-proven —
zero coverage misses across single and 4-request volleys; (2) simultaneous
decode is blocked by SCHEDULER ADMISSION, not the tier — the plain
tier-free server serializes identically; (3) temp-0 output diffing is
unusable as a gate on this build (the plain server differs from itself),
so correctness is carried by the new post-forward coverage audit plus
bitwise KV verify.**

## Multi-row gather (the G3-era known gap)

`before_gather` now attributes each token's selections to its own row via
extend_seq_lens deltas and restores per row (`_restore_row_pages`);
CPU fixtures cover the attribution (rows restore their own pages, cross
pages stay retired). On GPU: 4x159k and 4x101k volleys, 0 coverage
misses, 0 verify mismatches, decode steady ~34-46 tok/s (159k) /
~42 tok/s (101k).

## The concurrency verdict

4x101k, 512 forced-length decodes, conservativeness 1.5: tier and plain
servers BOTH run every decode batch with #running-req: 1. Prefill bursts
and decode bursts strictly alternate. Cause: admission reserves each
request's full input against the 200k pool; two 101k inputs (202k) never
fit simultaneously, so requests time-share. True simultaneous decode of
100k-class requests needs admission that accounts for the tier's
streaming (resident << input) — an upstream scheduler change, not an
attention-hook overlay. Until then "32 users" is time-shared: each user
sees full-rate decode (~50 tok/s) during their slot.

## Correctness tooling

- audit_coverage (post-forward): every SELECTED page must be non-retired
  after the forward; a miss means a zeroed-page read. Counted in stats
  (coverage_misses). First runs: all zero.
- Output diffing tier-vs-plain: REJECTED as a gate — the plain server
  produces different temp-0 outputs run-to-run on this build (greedy
  argmax ties flip under nondeterministic kernels; G2-era notes said the
  same). Coverage + bitwise KV verify are the real signals.

## Also fixed / worked around

- Idle-leak kill: retire decrements kv_allocated_len, the runtime's
  finish subtracts the original extent again — a bounded double-subtract
  (~4 pages/volley) that the strict idle checker escalated to a kill.
  Launcher sets SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=false (warn);
  the proper fix is retiring via the runtime's own free path.
- Docker gotcha for the record: failed `docker run` attempts make the
  daemon create missing bind-mount sources AS DIRECTORIES (root-owned);
  later launches then mount a directory over the .py path and Python
  reports ModuleNotFoundError. Fix: sudo rm the bogus dirs, re-scp real
  files, verify with `file`.

## Config deltas vs HKV-6

SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=false,
--schedule-conservativeness 1.5 (no effect on interleaving), prefetch on,
FREEZE_STREAK still 999999 (off).
