# Native A/B first execution — 2026-10-20, instance-20261020-hkv23

**Partial execution with a hard blocker found and root-caused to two
distinct bugs (both fixed, one residual). Vanilla mode: full clean
matrix. Tier mode: N=1 fully correct at native quality (17k-resident
decode of a 101k context at 205-215 tok/s); N>=4 stalls after the first
completion.**

## What ran (image hostkv:4ccff141-5, MODE=vanilla|tier, 101k ctx)

- vanilla N=1/4/8: walls 26.1 / 56.0 / 113.7 s, ALL requests ok,
  decode steady ~140 tok/s single. The fork with the flag off behaves
  like stock (HKV-21's regression leg).
- tier N=1: wall 33.7s, ok; decode logs show #full token 17,024 for a
  101,043-token context (84k retired to host!) at 205-215 tok/s with
  accept ~3.8/4 - the native tier delivering the headline claim.
- tier N=4: request #1 completes; request #2 stalls after its first
  prefill chunk (row ctx frozen at 8192), scheduler spins in the tick,
  GPU idle. Two root causes fixed en route; a residual stall remains.

## Bugs found and fixed this session (all committed to host-kv/main)

1. launcher never passed TIER_FLAGS to docker run (516518e2da).
2. scheduler init read undefined `server_args` (9961445b94).
3. chat-completions logprobs live under choices[0].logprobs.content —
   the harness's parse AttributeError'd every request (3f2c96258d).
4. DEAD-ROW EXPLOSION (3f30ea6bff): completed requests left row state;
   the stash slot then held the NEXT request's selection, so every tick
   restored a dead row into fresh reserve frames and audited foreign
   positions — 2.4 BILLION coverage misses, reserve drained, live
   request starved. Fix: captured_len<=1 (the stash's never-captured
   sentinel) drops dead state.
5. FROZEN-CAPTURE ROWS (3d06f45f66): a finished request's stash slot
   keeps its last capture until reuse (sentinel never fires); the tick
   serviced the dead row forever. Fix: captured_len frozen for 128
   ticks => drop. After both fixes: coverage_misses fell 2.4B -> 6,
   starved -> 0, and dead rows disappear from stats.

## The residual N>=4 stall (open)

Signature (image -5): after the first completion, the next request
admits (row state created, first chunk's 6-8192 tokens written) then
never progresses; the scheduler ends parked in recv with the waiting
queue drained and no forwards; br_ticks freezes. Suspects: the tick's
interaction with the between-requests idle loop (per-iteration torch
work while idle), or the finish-path accounting leaving availability
temporarily too low for the next admission and the request vanishing on
client-visible state without erroring. Evidence: ab-tier-final.log
(container logs), the inline decode snapshot above, py-spy stacks
(tick / recv). Needs a dedicated debug session with admission logging
(the diagnosis overlay exists for exactly this).

## Proven this session (HKV-21 acceptance)

One image (hostkv:4ccff141-5, built by Dockerfile.hostkv from the
branch): MODE=vanilla serves stock config with zero tier lines;
MODE=tier arms natively ([host-kv] armed: 12 layers, budget=102400
tokens, reserve=97536), confiscates, retires mid-prefill (row 15
mirrored 1579 pages), selfcheck verified=True, between-replay tick
active with bounded misses (6).
