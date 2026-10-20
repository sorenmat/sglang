# HKV-25 debug session — 2026-10-20, instance-20261020-hkv25 (europe-north1-a)

**Five root-cause fixes landed on host-kv/main; the native tier is fully
correct at N=1 and intermittently at N=2/4, but a residual wedge remains
under sequential load. The overlay implementation stays the validated
reference. Ten image build/validation cycles; honest record below.**

## Fixes landed (all committed, CPU tests green, images -6..-15)

1. `516518e2da` launcher never passed TIER_FLAGS (prior session).
2. `9961445b94` scheduler init read undefined `server_args`.
3. `3f30ea6bff` DEAD-ROW EXPLOSION: completed requests left row state;
   the recycled stash slot served the next request's selection to a dead
   row - 2.4B coverage misses, reserve drained, live request starved.
4. `3d06f45f66` FROZEN-CAPTURE rows serviced forever (heuristic; later
   superseded by forget_row).
5. `10188b46ea` + `24de39310e` + `fde76390ef` + `a85a28a114` +
   `28c029dd56`: scheduler-driven forget_row replaces ALL heuristics
   (captured_len<=1 also matches live PREFILLING rows - the tick was
   deleting live rows); forget_row reclaims the row's reserve frames
   (restored+prefetched pages leaked frames otherwise - 1.39B starved by
   N=8); the tier handle is passed as a constructor field (frozen
   dataclass setattr raised); the hook runs BEFORE release_kv_cache
   (which nulls req.req_pool_idx, silently failing the after-call
   guard).
6. `061fa25fea` + `5670132391`: instrumentation (tick/post-hook
   exceptions logged once with traceback; SGLANG_HOSTKV_ADMIT_LOG=1
   refusal logging incl. add_chunked_req continuation budget) and the
   stash-coherence guard (skip rows whose capture is incoherent with
   tier state - recycled slots hold the previous request's selection
   during the next request's prefill).

## Validated states

- Image -7 (fixes 1-4): N=2 PASS (22.7/40.2s), N=4 PASS (all four
  full-length: 14.7/29.2/43.7/60.0s), N=8 stall - diagnosed as the
  forget-row frame leak (fix 5).
- Image -15 (all fixes): N=1 PASS (native quality: 17k-resident decode
  of a 101k context at 143-215 tok/s). N=2 intermittent: request #1
  completes (28.6s), request #2 times out at 900s while the scheduler
  spins INSIDE the tick (py-spy: _restore_row_pages -> tick ->
  scheduler_tick, active+gil) with br_ticks at 4.5M and counting;
  decode bursts appear when the loop does schedule.

## The residual wedge (open)

Under sequential load the between-replay tick dominates the scheduler
loop's CPU (py-spy consistently inside _restore_row_pages); batches run
in bursts and a queued request's progress crawls until client timeout.
Counters stay HEALTHY during the wedge (coverage_misses ~8, starved 0,
verify 0) - it is a per-iteration cost/starvation problem, not
corruption. Suspects for the next session: (a) the per-tick stash read
(`indices[0, rows_t].cpu()` - a GPU sync) and the 12-layer restore
copies at full loop speed; (b) the tick running while the loop is idle
(needs a ran-last-batch gate); (c) stash width (token_topk+tail) making
each tick restore dozens of pages. First move: gate the tick on "a
decode batch ran since the last tick" and measure tick wall time; the
overlay's identical logic passed at N=32, so a structural delta (e.g.
tick firing on iterations where the overlay's did not) is likely.

## Repro

Image hostkv:4ccff141-15 on 2xRTX PRO 6000; launch-hostkv.sh MODE=tier
KEEP=1600; n2repro.py (2x101k, 1024 forced-length). leak.log in this
directory; artifacts/ab-native-20261020 has the earlier evidence.
