# HKV-11 bulk session — 2026-10-19, instance-20261019-hkv11b

**True concurrent decode at 32-user scale: 32/32 OK, wall 229.6 s (G3:
316 s, HKV-6: 334 s), decode batches observed at #running-req up to 10
per server, 124,559 staged restores across both servers with ZERO
verify mismatches, ZERO coverage misses, 100% prefetch hit rates, and
graceful idle stops.**

## What landed (commit 254eb4f14a)

1. **Bulk retire** (frontier-tracked): candidates are pages newly below
   the horizon plus restored pages - O(churn) per call, never the
   O(context) scan (~1,600-page iteration per decode step that helped
   pin the earlier run at ~2 tok/s). One table-zeroing op, one reserve
   cat, one dedup free-set build, one kv_allocated_len clamp per call.
2. **Bulk restore**: one reserve slice per gather, one host-gather + H2D
   per layer (24 copies per gather, not per page), one table write, one
   stream sync; fast/sync partition computed before any mutation.
3. **CORRECTNESS FIX (important)**: the staged restore previously read
   mirror[frame_slots] - empty on a page's first restore since the
   mirror holds bytes at the ORIGINAL slots - staging zeros and silently
   deadening selected positions in every shrunk-pool run to date (the
   old bitwise verify compared mirror[frame] vs gpu[frame] and passed
   vacuously). The copy now reads mirror[src] and writes gpu[dst]; the
   selfcheck verifies src!=dst and passes on GPU.

## Measured ladder (2x RTX PRO 6000, one server per GPU)

- 2x101k, 512 forced-length decodes: wall 50.6 s (was 562-579 s before
  the bulk rewrite - 11x), running=2, aggregate 87 tok/s over steady
  intervals, 0 coverage misses.
- 4x101k: wall 67.1 s, running observed 2/3/4, aggregate 92 tok/s,
  0 misses. (One request finished at 144 tokens with reason 'stop' -
  the below-min_tokens early stop seen intermittently; unresolved.)
- 32-user verdict (16+16, corpus 101,435-token prompts, 256 max):
  32/32 OK, wall 229.6 s, running up to 10 per server, finish reasons
  27 stop / 5 length (corpus-typical), 0 mismatches, 0 coverage misses
  (68,710 + 55,849 staged pages), clamp_skips 0.

## Standing

The tier now delivers SIMULTANEOUS long-context decode at 32-user scale
with full correctness instrumentation green. Per-request decode rate
under N-way overlap remains bounded by eager execution (no CUDA graphs);
aggregate interval sampling during heavy prefill+decode mixing read
62 tok/s at running=10 - clean per-request-under-overlap rates need a
dedicated measurement (no-prefill steady state), which is the natural
first measurement of the graphs ticket (HKV-12).

Everything torn down: instance-20261019-hkv11b deleted (disk
auto-deleted), zero instances remain.
