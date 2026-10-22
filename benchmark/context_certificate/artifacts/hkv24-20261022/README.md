# HKV-24 — Production readiness: 4-hour soak, runbook, known limitations

Branch `host-kv/main` @ `5d2b6f7a15` (image `hostkv:4ccff141-20`, base lineage `4ccff141db`).
Hardware: 1× L4 (g4-standard-4 spot), Qwen3.8-Next (QSA + Flash Sparse), TP=1.
Soak instance: `instance-20261022-final` (asia-south1-c), 2026-10-22.

## 1. Soak — 4 hours, mixed concurrency, tier mode keep=1700

Harness: `/home/runner/soak2.sh` on the instance. Loop of volleys at concurrency
N = 1, 2, 4, 2, 1 (101k-token inputs, `hangtest.py`, ignore_eos off), each volley
followed by a counter snapshot (`GET /health` style stats dump from the tier:
`forgotten_rows`, `coverage_misses`, `restore_starved`, `verify_mismatches`).
The script recycles the tier server whenever `forgotten_rows > 30` (the per-server
finish budget, see §3.1) and logs `recycled` per event. Log: `soak2.log`.

Results: `soak-summary.md` + raw `soak2.log`. PASS in 235 min: 0 scheduler
exceptions, 325 clean request completions, starved=0 and verify=0 on every
snapshot of all 24 server lives, 24 recycles at a deterministic ~10-min
cadence matching the ~31-finish budget, and clean recovery from two
harness-level recycle races (no tier defects).

## 2. Runbook

### 2.1 Launch (single L4, tier mode)

```bash
bash launch-hostkv.sh 0 30000 tier-1700 tier 1700
# AUTO default (tier dormant until an oversized request arrives):
bash launch-hostkv.sh 0 30000 auto-1700 auto 1700
```

`launch-hostkv.sh GPU PORT NAME MODE [KEEP] [CTX160K]` — MODE ∈ {vanilla, tier, auto}.
Keep `KEEP=1700` for 101k-class inputs (1600 is too tight — intake clamp rejects).
Wait for `hostkv image OK` + `The server is fired up and ready to roll!` in the log.

### 2.2 Monitor

- Health: `curl -s localhost:30000/health | jq .health` (or `/get_server_info`).
- Tier counters: the server logs periodic tier stats; key fields:
  - `forgotten_rows` — completed requests reaped. **Alert > 30** → recycle (§2.3).
  - `restore_starved` — restores that found no free GPU frame. **Must stay 0.**
  - `verify_mismatches` — GPU/host KV checksum drift on sampled rows. **Must stay 0.**
  - `coverage_misses` — sparse-index stash lookups that missed (expected,
    benign at O(100)/server-life at this workload; investigate if it grows
    into the millions — that was the HKV-25 failure signature).
- Exceptions: `grep -c "Scheduler hit an exception"` in the server log — **must stay 0.**

### 2.3 Recycle (the operational answer to the finish-path leak)

When `forgotten_rows > 30` (or starved/verify go nonzero — fail fast):

```bash
# stop old server, start a fresh one on the same port, keep the name
docker rm -f <container>; docker run -d --gpus '"device=0"' --name <name> \
  -p 30000:30000 -v /home/runner/sglang:/app/sglang --memory-swappiness 0 \
  hostkv:4ccff141-20 ...  # same flags as launch-hostkv.sh prints
```

`soak2.sh` automates exactly this; reuse it as the recycle daemon with one
fix learned in the soak: health-gate after relaunch (wait ~2–3 min for the
model load + `/health` 200) before resuming traffic, and never treat a
failed stats fetch as a budget trigger — firing volleys into the load window
caused a double recycle at t=209min (recovered cleanly, see soak-summary).
Recycle cost is a cold model load (~2–3 min on L4); at the soak's workload
that is <5% duty cycle. In-flight requests must drain first.

### 2.4 Teardown

```bash
gcloud compute instances delete INSTANCE --zone ZONE --project kapeta-staging --quiet
gcloud compute instances list --project kapeta-staging   # must be empty
```

GPU instances are spot; ALWAYS delete when done (GPU_HOST_POLICY.md).

## 3. Known limitations

1. **Per-server finish budget (~31 finishes at keep=1700).** The residual
   finish-path leak (HKV-26 root cause fixed; a small residual ~180 tokens of
   headroom lost per completed request remains) caps a tier server at ~31
   completed 101k-class requests before intake headroom is exhausted.
   Mitigation: scheduled recycle (§2.3). Follow-up ticket for the residual.
2. **160k-class workloads: vanilla is the right choice today.** On this card
   (96 GB, ~80.5 GB weights → ~200–211k-token KV pool) 160k inputs fit without
   the tier, and vanilla is measurably faster there (no retire/restore traffic).
   The tier's win is at/above the vanilla ceiling: 190k near-ceiling it wins
   (39 s vs 42 s wall at N=2, matrix cell 190k/N2, ab-definitive-20261022).
   AUTO mode (`--host-kv-arm-min-input-tokens`) makes this a non-decision:
   identical to vanilla until an oversized request arms it.
3. **256k single-GPU is memory-infeasible in every mode.** 258k input is
   rejected 400 by the runtime intake clamp regardless of tier/vanilla/auto —
   the clamp is upstream of any pool sizing. Serving 256k needs either
   2×TP1 two-server topology or the upstream EP2 fix (sglang #44291).
4. **Idle-checker warnings during sustained load are expected.** The watchdog
   logs warnings when the scheduler loop is busy >X s on long prefills; they
   are noise at 101k inputs and did not precede any failure in the soak.
5. **CUDA-graph + tier**: supported via capture-mode stand-down + between-replay
   ticks, but the tick only runs on decode/target-verify batches; a server
   doing pure prefill work does not tick (fine — nothing to retire mid-prefill).
