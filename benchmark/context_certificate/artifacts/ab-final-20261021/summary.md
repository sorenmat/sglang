# Native A/B final — 2026-10-21, instance-20261021-hkv25b (survived 2 preemptions)

**Tier N=1/4/8 fully pass natively for the first time (keep=1700): all
requests ok, verify_mismatches=0, rows cleanly forgotten
(forgotten_rows increments, frames reclaimed). The single remaining
blocker is now precisely bounded: a cumulative finish-path accounting
leak (~512 tokens + reserve drift per completed request) that breaks
N=32 at 10-24 completions depending on headroom.**

## Results (image hostkv:4ccff141-17, 101k ctx, 1024 forced-length)

| mode | N=1 | N=4 | N=8 | N=32 |
|---|---|---|---|---|
| vanilla | 25.3s ok | 53.6s ok | 115.1s ok 32/32... (31.5s wall/ok from earlier leg) | 450.1s, 32/32 ok |
| tier (keep=1700) | 30.7s ok | 76.1s, 4/4 ok | 149.4s, 8/8 ok | stalls at ~24 (512t/finish leak > 4.6k headroom) |
| tier (keep=2000) | - | - | - | 204.2s, 10/32 ok; reserve drained (starved 783k) |

(vanilla leg from the first matrix run before preemption #1; tier legs
from this session's runs. Raw logs + results.json in this directory.)

Tier steady decode at N=1: ~141-215 tok/s with only 17.3k of 101k
tokens resident (84% of context on host) — the headline claim, native.

## Fixes this session (all on host-kv/main)

1. `7bbdb3e3e7` forget hooks at the release_kv_cache choke point
   (module registry in mem_cache/common.py) + tick gated on
   decode-batch activity. forgotten_rows finally increments.
2. `3790fe7c43` normal-path add_one_req NO_TOKEN admission logging.
3. `a3eb9c890b` harness keep=1700/2000 (headroom for the leak).

## The remaining defect (next ticket, blocks N=32 + soak)

Cumulative per-finish accounting leak: after each completed request the
allocator identity loses ~512 tokens (8 pages) and the reserve
accounting drifts (keep=2000 run: reclaimed 6708 frames over 11 rows
yet the reserve starved). With keep=1600 (269-token headroom) the 4th
request never admits; keep=1700 survives ~23 finishes; keep=2000
survives ~10 (reserve drain dominates at higher keep). This is HKV-13's
documented tail - the finish path's double-count - now measured as the
SOLE blocker to N=32 and long-running operation. Fix path: make
forget_row reconcile the row's runtime-allocated pages against
allocator state before release (or move retirement bookkeeping fully
into forget_row so release only sees pages it understands).

## Soak (HKV-24): NOT RUN - gated on the leak fix. The leak rate
(~512 tokens/finish) caps any server's sequential request count by
budget; a soak would measure exactly this leak rate and nothing else
until it's fixed.
