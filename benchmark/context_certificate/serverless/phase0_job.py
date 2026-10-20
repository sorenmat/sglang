"""Phase-0 measurement job for a serverless GPU container (Verda-style).

One process, no SSH: downloads (or reuses) the pinned checkpoint, launches
the serving runtime in-process configurations, runs the Phase-0 gates, and
prints every result to stdout as it lands (spot-loss safe by design).

Stages (select with PHASE0_STAGES="m01 m02 m03 m04", default all):
  m01  context flatness — largest corpus request (~170k tokens) vs the 30k
       baseline, profiled decode windows
  m02  clean steady-state batch — N=16 distinct ~30k requests, profiler
       armed only once all 16 are running
  m03  TP2 baseline — same measurement on two GPUs (--tp-size 2)
  m04  link bandwidth — pinned host<->device streams

Requirements inside the container: the pinned image
lmsysorg/sglang:dev-qwen38-next-local (torch, sglang, huggingface_hub), a
writable WORKDIR, and ASSETS_DIR containing workload.jsonl (the recorded
request corpus). All numbers printed as JSON blocks; full traces stay in
WORKDIR/traces for optional upload afterwards.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

IMAGE_DIGEST = "sha256:9d2a843c706c74bc259c0d9abf360551eb2734e1e7d255ab012a6965f10480b6"
CHECKPOINT = "nvidia/Qwen3.8-Flash-Next-NVFP4"
REVISION = "fc694b54fb0174e0913e6adf86691ef85a4ead47"
BASE = "http://127.0.0.1:30000"

WORKDIR = Path(os.environ.get("PHASE0_WORKDIR", "/root/phase0"))
ASSETS = Path(os.environ.get("PHASE0_ASSETS", "/assets"))
WEIGHTS = WORKDIR / "weights"
TRACES = WORKDIR / "traces"


def log(event):
    print(f"[phase0 {time.strftime('%H:%M:%S')}] {event}", flush=True)


def emit(name, payload):
    print(f"=== RESULT {name} ===\n{json.dumps(payload, indent=2)}", flush=True)


def fetch_weights():
    if (WEIGHTS / "config.json").exists() and (WEIGHTS / "model.safetensors.index.json").exists():
        log("weights already present, skipping download")
        return
    log(f"downloading {CHECKPOINT}@{REVISION[:8]}")
    from huggingface_hub import snapshot_download
    snapshot_download(CHECKPOINT, revision=REVISION, local_dir=str(WEIGHTS))
    log("weights ready")


def wait_ready(timeout=5400, proc=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError("server process exited before becoming healthy")
        try:
            with urllib.request.urlopen(BASE + "/v1/models", timeout=5) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        time.sleep(10)
    raise RuntimeError("server did not become healthy")


def running_requests():
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=5) as r:
            body = r.read().decode()
        for line in body.splitlines():
            # gauge lines carry exporter labels: name{...} value
            match = re.match(r"^sglang:num_running_reqs(?:\{[^}]*\})?\s+(\S+)", line)
            if match:
                return float(match.group(1))
    except OSError:
        pass
    return -1


class Server:
    """Launch sglang serve as a subprocess; one config at a time."""

    def __init__(self, name, extra_args):
        self.name, self.extra = name, extra_args
        self.proc = None
        self.log = WORKDIR / f"server-{name}.log"

    def __enter__(self):
        cmd = ["sglang", "serve",
               "--model-path", str(WEIGHTS), "--served-model-name", "flash-next-nvfp4",
               "--json-model-override-args", '{"language_model_only":true}',
               "--moe-runner-backend", "flashinfer_cutlass", "--ple-offload-embedding",
               "--max-mamba-cache-size", "48",
               "--mamba-ssm-dtype", "bfloat16", "--chunked-prefill-size", "8192",
               "--speculative-algorithm", "NEXTN", "--speculative-num-steps", "3",
               "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", "4",
               "--cuda-graph-backend-prefill", "disabled",
               "--enable-metrics", "--host", "127.0.0.1", "--port", "30000"] + self.extra
        log(f"starting server {self.name}: {' '.join(cmd[4:])}")
        self.handle = open(self.log, "wb")
        self.proc = subprocess.Popen(cmd, stdout=self.handle, stderr=subprocess.STDOUT,
                                     env={**os.environ,
                                          "SGLANG_TORCH_PROFILER_DIR": str(TRACES),
                                          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        wait_ready(proc=self.proc)
        log(f"server {self.name} ready")
        return self

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.handle.close()
        log(f"server {self.name} stopped")


def select_requests(count, min_tokens, max_tokens, source=None):
    rows = [json.loads(line) for line in open(source or (ASSETS / "workload.jsonl"))]
    band = [r for r in rows if min_tokens <= r["observed_prompt_tokens"] <= max_tokens]
    if len(band) < count:
        raise RuntimeError(f"only {len(band)} requests in band")
    band.sort(key=lambda r: r["observed_prompt_tokens"])
    step = (len(band) - 1) / max(count - 1, 1)
    return [band[round(i * step)] for i in range(count)]


def submit(payload, max_tokens):
    body = json.dumps({**payload, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        return json.loads(r.read())


def start_profile(num_steps=64):
    payload = {"output_dir": str(TRACES), "num_steps": num_steps,
               "activities": ["GPU"]}
    if os.environ.get("PHASE0_BY_STAGE") == "1":
        payload["profile_by_stage"] = True  # opt-in: stage-split flush
                                            # failed server-side 3x
    req = urllib.request.Request(
        BASE + "/start_profile",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            log(f"start_profile: {r.read().decode()[:120]}")
    except urllib.error.HTTPError as e:
        log(f"start_profile failed (continuing): {e.read().decode()[:200]}")


def run_window(name, rows, max_tokens, wait_running=None, num_steps=None):
    """Submit rows concurrently; arm the profiler once steady; return usages."""
    results = [None] * len(rows)

    def worker(i, row):
        try:
            answer = submit(row["request"], max_tokens)
            results[i] = {"sha": row["request_sha256"][:12],
                          "prompt": row["observed_prompt_tokens"],
                          "completion": answer.get("usage", {}).get("completion_tokens")}
        except Exception as error:  # noqa: BLE001 - record and continue
            results[i] = {"sha": row["request_sha256"][:12], "error": str(error)[:200]}

    threads = [threading.Thread(target=worker, args=(i, r)) for i, r in enumerate(rows)]
    for t in threads:
        t.start()
    before = set(TRACES.glob("*.trace.json.gz"))
    if num_steps is None:
        # A window that spans prefill+decode needs more than 64 scheduler
        # iterations; 16 staggered prefills alone consume ~100+.
        num_steps = int(os.environ.get("PHASE0_PROFILE_STEPS", "600"))
    if wait_running:
        start = time.monotonic()
        deadline = start + 150
        while time.monotonic() < deadline:
            running = running_requests()
            # -1 tolerance: the token pool can keep one request permanently
            # queued, capping concurrent running below the nominal batch.
            if running >= wait_running - 1:
                break
            time.sleep(2)
        log(f"arm condition: running={running_requests()} (target {wait_running}, "
            f"waited {time.monotonic() - start:.0f}s)")
    else:
        time.sleep(2)
    start_profile(num_steps)
    for t in threads:
        t.join()
    # Close the by-stage session; an open session silently swallows the next
    # window's trace, and files flush asynchronously after the stop.
    try:
        urllib.request.urlopen(BASE + "/stop_profile", timeout=60).read()
    except (urllib.error.HTTPError, OSError):
        pass
    trace_path = wait_for_new_trace(before, name)
    return results, trace_path


def wait_for_new_trace(before, name, timeout=120, poll=3):
    """Wait for ANY new trace file after a stop_profile call.

    The by-stage profiler has produced suffix-less `TP-0.trace.json.gz`,
    `-DECODE`-suffixed, and (window 5) nothing at all; keying on the
    suffix lost traces twice. Any new file wins; DECODE-suffixed files
    are preferred when several appear."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        new = set(TRACES.glob("*.trace.json.gz")) - before
        if new:
            decode = [p for p in new if "DECODE" in p.name]
            pick = max(decode or new, key=lambda p: p.stat().st_mtime)
            suffix = "-DECODE" if "DECODE" in pick.name else "-TRACE"
            renamed = TRACES / f"{name}{suffix}.trace.json.gz"
            pick.rename(renamed)
            log(f"captured {pick.name} -> {renamed.name} "
                f"(candidates: {sorted(p.name for p in new)})")
            return renamed
        time.sleep(poll)
    log(f"WARNING: no new trace file for {name} within {timeout}s "
        f"(pre-existing: {len(before)}); profiler session may not have "
        f"flushed — check server logs for the stop_profile response")
    return None


def parse_trace(path, assume_scorer="kernel_kernel"):
    import gzip
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        trace = json.load(handle)
    clusters, kernels, counts = defaultdict(float), defaultdict(float), defaultdict(int)
    index = ("index_score", "index_topk", "index_prep")
    table = {"index_score": r"tilelang|qsa.*score|weighted_mqa", "index_topk": r"fast_topk|topk",
             "index_prep": r"qsa_index_(q|k)_", "gemm": r"gemm|cutlass|nvjet|marlin|matmul"}
    total_events = 0
    for event in trace.get("traceEvents", []):
        if event.get("cat") != "kernel":
            continue
        name, dur = event.get("name", ""), event.get("dur", 0)
        if not name or dur <= 0:
            continue
        cluster = "index_score" if name == assume_scorer else next(
            (c for c, pat in table.items() if re.search(pat, name, re.IGNORECASE)), "other")
        clusters[cluster] += dur
        kernels[name] += dur
        counts[cluster] += 1
        total_events += 1
    total = sum(clusters.values())
    steps = sum(c for k, c in counts.items() if "topk" in k.lower()) // 13
    return {"total_ms": round(total / 1e3, 1), "steps": steps,
            "ms_per_step": round(total / max(steps, 1) / 1e3, 2),
            "gemm_ms_per_step": round(clusters["gemm"] / max(steps, 1) / 1e3, 2),
            "index_ms_per_step": round(sum(clusters[c] for c in index) / max(steps, 1) / 1e3, 2)}


def latest_decode_prefix(tag):
    files = sorted(TRACES.glob("*.trace.json.gz"), key=lambda p: p.stat().st_mtime)
    decode = [p for p in files if "DECODE" in p.name]
    if not decode:
        raise RuntimeError(f"no DECODE trace produced for {tag}")
    newest = decode[-1]
    renamed = TRACES / f"{tag}-DECODE.trace.json.gz"
    newest.rename(renamed)
    return renamed


def m01():
    rows = select_requests(2, 25_000, 40_000)
    big = sorted((json.loads(l) for l in open(ASSETS / "workload.jsonl")),
                 key=lambda r: -r["observed_prompt_tokens"])[0]
    # One window per server lifetime: the runtime's profiler allows a single
    # by-stage session per server, so a second window re-parses the first.
    with Server("m01-short", ["--context-length", "262144", "--max-total-tokens", "200000",
                              "--max-running-requests", "8", "--mem-fraction-static", "0.95"]):
        short, trace = run_window("m01-30k", rows[:1], 256, num_steps=64)
        emit("m01_single_30k", {"requests": short})
        if trace:
            emit("m01_30k_profile", parse_trace(trace))
    with Server("m01-long", ["--context-length", "262144", "--max-total-tokens", "200000",
                             "--max-running-requests", "8", "--mem-fraction-static", "0.95"]):
        long_res, trace = run_window("m01-long", [big], 192, num_steps=64)
        emit("m01_single_%dk" % (big["observed_prompt_tokens"] // 1000), {"requests": long_res})
        if trace:
            emit("m01_long_profile", parse_trace(trace))


def m02():
    rows = select_requests(16, 22_000, 36_000)
    with Server("m02", ["--context-length", "262144", "--max-total-tokens", "520000",
                        "--max-running-requests", "16", "--mem-fraction-static", "0.95"]):
        results, trace = run_window("m02", rows, 256, wait_running=16, num_steps=64)
        emit("m02_batch16_clean", {"requests": results})
        if trace:
            emit("m02_profile", parse_trace(trace))


def m03():
    rows = select_requests(8, 25_000, 40_000)
    with Server("m03-tp2", ["--tp-size", "2", "--ep-size", "2", "--context-length", "262144",
                            "--max-total-tokens", "520000", "--max-running-requests", "16",
                            "--mem-fraction-static", "0.93"]):
        one, trace = run_window("m03-n1", rows[:1], 256, num_steps=64)
        emit("m03_tp2_n1", {"requests": one})
        if trace:
            emit("m03_tp2_n1_profile", parse_trace(trace))
        eight, trace = run_window("m03-n8", rows[:8], 256, wait_running=8, num_steps=64)
        emit("m03_tp2_n8", {"requests": eight})
        if trace:
            emit("m03_tp2_n8_profile", parse_trace(trace))


def m04():
    import torch
    report = {"device": torch.cuda.get_device_name(0), "gpus": torch.cuda.device_count()}
    for gpu in range(torch.cuda.device_count()):
        torch.cuda.set_device(gpu)
        host = torch.empty(2 * 1024**3 // 4, dtype=torch.float32).pin_memory()
        dev = torch.empty_like(host, device="cuda")
        for _ in range(3):
            dev.copy_(host, non_blocking=True)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(10):
            dev.copy_(host, non_blocking=True)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / 10
        report[f"gpu{gpu}_h2d_GBps"] = round(2 * 1024**3 / (ms / 1e3) / 1e9, 1)
    emit("m04_link_bandwidth", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", default=os.environ.get("PHASE0_STAGES", "m01 m02 m03 m04"))
    parser.add_argument("--plan", action="store_true", help="print the plan and exit (no GPU)")
    args = parser.parse_args()
    if args.plan:
        print(json.dumps({"image": "lmsysorg/sglang:dev-qwen38-next-local", "digest": IMAGE_DIGEST,
                          "stages": args.stages.split(), "workdir": str(WORKDIR),
                          "assets": str(ASSETS), "weights": str(WEIGHTS)}, indent=2))
        return 0
    for path in (WORKDIR, TRACES):
        path.mkdir(parents=True, exist_ok=True)
    all_stages = {"m01": m01, "m02": m02, "m03": m03, "m04": m04}
    stages = {name: all_stages[name] for name in args.stages.split() if name in all_stages}
    if not stages:
        raise SystemExit(f"no valid stages in {args.stages!r}; choose from {sorted(all_stages)}")
    if any(name != "m04" for name in stages):
        fetch_weights()
        if not (ASSETS / "workload.jsonl").exists():
            raise SystemExit(f"workload.jsonl missing at {ASSETS}")
    for name, fn in stages.items():
        log(f"===== stage {name} =====")
        try:
            fn()
        except Exception as error:  # noqa: BLE001 - keep later stages running
            emit(name, {"status": "failed", "error": str(error)[:500]})
    log("all requested stages complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
