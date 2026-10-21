# HKV-26 fix + capacity A/B — 2026-10-21 (three preemptions survived)

**The leak is fixed: N=32 completes 32/32 natively for the first time
(33 rows forgotten, 42,560 frames reclaimed, starved 0, verify 0). The
160k capacity A/B ran head-to-head: at 160k x N=4 vanilla and tier BOTH
complete all requests, vanilla slightly faster — the tier's capacity
payoff requires contexts/pools where vanilla cannot hold even one, or
the 256k target (needs a 262k-token pool, a config change).**

## The fix (43baf52741, image -18)

Closed frame ownership: every frame entering the reserve joins the
tier_frames ownership set (the confiscation snapshot missed recycled
entries); forget_row routes by provenance - tier frames to the reserve,
no-provenance frames to the allocator. The smoking gun was 610 frames
reclaimed per request against a ~30-page working set. New CPU
conservation tests: no frame in both ledgers across the churn cycle;
corrupt-provenance frames route to the allocator.

## Leak ladder (keep=1700, 101k ctx, one long-lived server)

| N | result |
|---|---|
| 2 | 2/2 (19.1/36.0s) |
| 4 | 4/4 (16.7-66.9s) |
| 8 | 8/8 (17.3-130.1s) |
| 32 | **32/32** (18.9-526.3s; serialized admissions at this budget - queueing delay, not failure) |

Counters after N=32: forgotten 33, reclaimed 42,560, coverage 144,
starved 0, verify_mismatches 0. Idle-checker still warns a ~6.8k-token
identity deficit at rest (down from unbounded; did not block 33
finishes) - recorded as residual.

## 160k capacity A/B (159,367-token prompts, keep=2510 tier)

| workload | vanilla (200k pool) | tier (160.6k budget) |
|---|---|---|
| N=2 | 2/2 (44.3/28.4s) | 2/2 (50.1/26.9s) |
| N=4 | 4/4 (16.1-63.4s), 1 retract | 4/4 (37.0-106.8s), 1 retract, cov 274k/starved 274k |

Interpretation: at 160k, vanilla's 200k pool holds one request and
serializes - and that is ENOUGH to complete N=4 with modest walls; the
tier matches completion but pays ~40-70% wall overhead at this budget
(615-page reserve starved its restores mid-flight: starved 274k). The
tier's structural payoff is at contexts >= the vanilla pool (256k needs
max-total-tokens 262144 - a launcher config change, untested) or pools
shrunk below context size. **For 160k-class workloads on this hardware,
vanilla remains the right choice today** - honestly recorded.
