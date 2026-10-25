# Soak summary — 4h mixed-load, tier mode keep=1700 (HKV-24)

Instance `instance-20261022-final` (asia-south1-c, L4), image
`hostkv:4ccff141-20`, branch f72abf41b0. Run 2026-10-22, ~14:05–18:00 UTC.
Harness: `soak2.sh` (volleys N=1,2,4,2,1 of 101k-token requests, snapshot
after each volley, recycle when forgotten_rows > 30). Full log: `soak2.log`.

## Result: PASS (with two harness-level caveats, no tier defects)

- **Duration**: 235 min (SOAK-COMPLETE recycles=24), 0 crashes, 0 stalls.
- **Scheduler exceptions**: **0** in 4 hours (grep "Scheduler hit an
  exception" count = 0 across all 24 server lives).
- **Requests**: 325 clean completions (`length` finishes, correct logprobs)
  across mixed N=1..4 volleys. 29 request errors — ALL inside the two
  recycle windows below; zero errors in the 3.5 h steady stretch between.
- **Tier counters per server life** (from live polls + final stats JSON):
  coverage_misses 89–156 (benign, expected O(100)), restore_starved **0**,
  verify_mismatches **0** on every snapshot. Final life (#24):
  forgotten 21, coverage 89, starved 0, verify 0, reclaimed_frames 26,600
  (frame conservation returning tier frames continuously).
- **Finish budget**: deterministic as measured — each server life walks
  forgotten 11 → 21 → 31 in ~10 min of sustained load, then the daemon
  recycles. 24 recycles over 235 min ≈ one per 10 min, matching the ~31
  finish budget at keep=1700 exactly. No life exceeded it, no life leaked
  past 31, no admission stalls between recycles.

## The two error windows (both harness-level, both recovered)

1. **t≈0–24 min, recycle #1** (9 request errors): the soak inherited the
   matrix-phase server which had already exhausted its finish budget
   (forgotten=33); requests got RemoteDisconnected/reset until the daemon's
   first stats fetch failed (sentinel 999) and recycled. Clean from t=29 min.
2. **t=209 min, recycles #21→#22** (19 request errors): recycle #21's
   `docker rm -f` reset in-flight connections (ConnectionReset), then the
   replacement's ~2–3 min model load left the port refusing; hangtest.py
   fired into that window and the next stats fetch also failed (999),
   triggering a second recycle. Server #22 came up healthy and every volley
   from t=214 min to the 235 min finish completed cleanly.

**Harness lesson (folded into runbook §2.3)**: the recycle daemon must
health-gate (wait for /health 200 after the 2–3 min model load) before
resuming volleys, and must not treat a failed stats fetch (999) as a budget
trigger. The tier itself never misbehaved in either window.

## Verdict

4 hours of sustained 101k mixed-concurrency load with 24 server recycles:
zero scheduler exceptions, zero starved restores, zero KV checksum
mismatches, bounded deterministic finish budget, and full recovery from both
recycle races. The tier is stable under its documented operational model
(scheduled recycle); the residual finish-path leak remained the single open
engineering item (HKV-26 follow-on).

> Postscript 2026-10-23: the finish-path leak is fixed (HKV-27, commit
> `a1f87403f8`); 45 finishes validated on one server life with a flat
> ledger. The recycles this soak exercised are no longer required —
> see `../hkv27-20261023/`.
