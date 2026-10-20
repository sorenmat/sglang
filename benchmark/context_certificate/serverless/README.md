# Running Phase 0 as a serverless GPU job (Verda)

No SSH, no instance babysitting: one container, one command, results printed
to the job log as they land (safe against preemption). The job runs the
[Phase-0 gates](../proof_kit/PROOF_PLAN.md) from the host-KV plan.

## Console recipe

1. **Deployment type:** Job (beta), **On-Demand** (spot also works — the job
   is restartable, but each restart re-pays any incomplete stage).
2. **GPU:** RTX PRO 6000 (96 GB) × **2** — the 2× container is required for
   stage m03 (TP2) and costs 2 × $2.182/h.
3. **Image:** `lmsysorg/sglang:dev-qwen38-next-local` — verify the registry
   digest is `sha256:9d2a843c706c74bc259c0d9abf360551eb2734e1e7d255ab012a6965f10480b6`
   before running; the pinned driver line on Verda hosts (R580) matches.
4. **Assets:** mount a volume at `/assets` containing `workload.jsonl`
   (copy: `benchmark/context_certificate/artifacts/workload.jsonl`). Put the
   job script itself on the same volume at `/assets/phase0_job.py`.
   Leave the volume mounted at a second path (or reuse `/assets`) so the
   88 GB checkpoint lands on persistent storage instead of the 500 GiB
   ephemeral disk — repeat jobs then skip the download.
5. **Command:**

   ```bash
   python /assets/phase0_job.py
   ```

   Stage selection: `PHASE0_STAGES="m01 m02"` etc. Defaults to all four.
   M0.4 alone (no model needed) can run on any 1-GPU container.

6. **Cost estimate:** ~2.5–3.5 h of 2-GPU time ≈ **$11–16** at listed
   prices, including the one-time checkpoint download (~12 min).

## What each stage prints

- `m01_30k_profile` / `m01_long_profile` — ms/step at ~30k×8 vs one ~170k
  request. **Gate:** long-context step ≤ ~1.5× the 30k single-request step.
- `m02_batch16_clean` / `m02_profile` — N=16 with the profiler armed only
  after all 16 requests are running. **Gate:** ≤ ~2× the N=8 step time.
- `m03_tp2_n8_profile` — first-ever TP2 measurement. **Gate:** ≤ ~35 ms/step
  at 8×30k for the 2,560 tok/s aggregate target to be plausible.
- `m04_link_bandwidth` — pinned H2D GB/s per GPU.

Full traces stay in the container at `/root/phase0/traces` (ephemeral) —
the parsed JSON in the log is the durable result. Upload traces from the
job only if object-storage credentials are wired in.

## Notes and caveats

- `sglang:num_running_reqs` from `/metrics` gates the m02 arm condition; if
  Verda's runtime exposes metrics differently, the arm falls back to timing.
- The 13-kernels-per-step step normalization assumes the same QSA layer
  count as the measured model (12 layers + 1 draft layer).
- Everything is printed live; if the job dies mid-stage, earlier `===
  RESULT` blocks in the log remain valid.
