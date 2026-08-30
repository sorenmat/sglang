"""Agent-workload benchmark: Qwen3.8-class hybrid models on a single GPU.

Drives coding-agent-shaped sessions against a launched SGLang server:

    context prompt (8K-128K tokens) -> decode burst -> tool return
    (2-8K tokens) -> decode burst -> tool return -> ... -> finish

and reports aggregate output token throughput, TTFT, and inter-token
latency (mean / p50 / p99) per (concurrency x context length) cell. The
point is to A/B the scheduler and memory features this branch adds:

    --mamba-mode auto|fixed     --mamba-full-memory-ratio auto vs 0.9
    --adaptive on|off           --enable-adaptive-prefill
    --nvfp4-mirror on|off       SGLANG_NVFP4_DQ_MIRROR_FRACTION

Target configuration (RTX PRO 6000, 1 GPU):
    python bench_agent_workload.py \
        --model <qwen3.8-27b-nvfp4-path> \
        --mtp 3/1/4 --kv-cache-dtype fp8_e4m3 \
        --concurrencies 1 4 8 16 24 --contexts 8192 32768 65536 131072 \
        --runs auto adaptive

Each "run" is a named feature configuration; the sweep relaunches the
server between runs and prints a comparison table at the end.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

DEFAULT_PORT = 31555


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------

@dataclass
class AgentSessionResult:
    output_tokens: int = 0
    ttft_s: float = 0.0
    itls: list = field(default_factory=list)
    total_s: float = 0.0


def build_context(num_tokens: int, seed: int) -> str:
    """~1 token per 3-4 chars of code-ish filler; deterministic per seed."""
    import random

    rng = random.Random(seed)
    lines = [
        f"def handler_{rng.randrange(1 << 30)}(req, ctx):",
        "    # validated invariants: session keys are scoped per agent",
        "    cache = ctx.shared_pool.acquire(req.session_id)",
        "    if cache.stale:",
        "        cache.refresh(policy='lazy')",
        "    return route(req.payload, cache)",
        "",
    ]
    chunk = "\n".join(lines)
    repeats = max(1, int(num_tokens * 3.2 / len(chunk)))
    return chunk * repeats


def tool_return(seed: int, num_tokens: int) -> str:
    return (
        f"\n[tool result {seed}] exit=0 stdout follows\n"
        + build_context(num_tokens, seed + 1)
        + "\n[end tool result]\nContinue.\n"
    )


def run_agent_session(
    base_url: str,
    worker: int,
    context_tokens: int,
    tool_tokens: int,
    num_tools: int,
    decode_tokens: int,
    timeout_s: float,
) -> AgentSessionResult:
    """One agent: big context, then alternating decode bursts and tool returns."""
    import requests

    result = AgentSessionResult()
    messages = [
        {
            "role": "system",
            "content": f"You are CODING_AGENT_{worker}. Context follows.\n"
            + build_context(context_tokens, seed=worker),
        }
    ]
    t_session0 = time.monotonic()
    first = True
    for turn in range(num_tools + 1):
        messages.append(
            {
                "role": "user",
                "content": "Continue the task." if turn == 0 else "Proceed.",
            }
        )
        t0 = time.monotonic()
        resp = requests.post(
            base_url + "/v1/chat/completions",
            json={
                "model": "default",
                "messages": messages,
                "max_tokens": decode_tokens,
                "temperature": 0,
                "stream": True,
            },
            stream=True,
            timeout=timeout_s,
        )
        resp.raise_for_status()
        content = []
        last_chunk_t = None
        ttft_recorded = False
        for line in resp.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            payload = line[len(b"data: ") :]
            if payload == b"[DONE]":
                break
            delta = json.loads(payload)["choices"][0].get("delta", {})
            text = delta.get("content")
            now = time.monotonic()
            if text:
                if not ttft_recorded:
                    result.ttft_s = now - t0
                    ttft_recorded = True
                elif last_chunk_t is not None:
                    result.itls.append(now - last_chunk_t)
                last_chunk_t = now
                content.append(text)
        messages.append({"role": "assistant", "content": "".join(content)})
        result.output_tokens += max(1, int(sum(len(c) for c in content) / 3.5))
        if turn < num_tools:
            messages.append(
                {"role": "user", "content": tool_return(worker * 97 + turn, tool_tokens)}
            )
    result.total_s = time.monotonic() - t_session0
    return result


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------


def parse_run_spec(run_name: str) -> tuple:
    """A run is `name` or `name@--flag=v,--flag2=v2` (comma-separated extra
    server args applied only for that run)."""
    if "@" in run_name:
        name, extra = run_name.split("@", 1)
        extras = [a.strip() for a in extra.split(",") if a.strip()]
        return name, extras
    return run_name, []


def server_cmd(args, run_name: str) -> list:
    name, run_extra_args = parse_run_spec(run_name)
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model",
        args.model,
        "--trust-remote-code",
        "--port",
        str(args.port),
        "--host",
        "127.0.0.1",
        "--kv-cache-dtype",
        args.kv_cache_dtype,
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--context-length",
        str(max(args.contexts) + 8192),
    ]
    if args.mtp:
        steps, topk, draft = args.mtp.split("/")
        cmd += [
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            steps,
            "--speculative-eagle-topk",
            topk,
            "--speculative-num-draft-tokens",
            draft,
        ]
        # MTP-enabled checkpoints carry the draft weights (mtp.* tensors);
        # only pass an explicit draft path when one was given.
        if args.draft_model:
            cmd += ["--speculative-draft-model-path", args.draft_model]
    if name in ("auto", "auto+adaptive"):
        cmd += ["--mamba-full-memory-ratio", "auto", "--max-running-requests", str(max(args.concurrencies))]
    elif args.mamba_ratio is not None:
        cmd += ["--mamba-full-memory-ratio", str(args.mamba_ratio)]
    if name in ("adaptive", "auto+adaptive"):
        cmd += [
            "--enable-adaptive-prefill",
            "--decode-latency-budget-ms",
            str(args.decode_latency_budget_ms),
        ]
    cmd += args.extra_server_args
    cmd += run_extra_args
    return cmd


def launch_server(args, run_name: str, log_path: str):
    env = dict(os.environ)
    env["SGLANG_NVFP4_DQ_MIRROR_FRACTION"] = (
        str(args.nvfp4_mirror_fraction) if run_name.endswith("+mirror") else "0.0"
    )
    cmd = server_cmd(args, run_name.replace("+mirror", ""))
    print(f"[launch] {' '.join(cmd[:8])} ... (log: {log_path})")
    log = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
    )
    base = f"http://127.0.0.1:{args.port}"
    deadline = time.time() + args.launch_timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early; see {log_path}")
        try:
            with urllib.request.urlopen(base + "/health_generate", timeout=5):
                return proc, base
        except Exception:
            time.sleep(3)
    proc.terminate()
    raise RuntimeError(f"server did not become healthy in {args.launch_timeout_s}s")


def server_derived_config(base: str) -> dict:
    try:
        with urllib.request.urlopen(base + "/get_server_info", timeout=10) as r:
            info = json.loads(r.read())
        return {
            "max_running_requests": info.get("max_running_requests"),
            "max_total_num_tokens": info.get("max_total_num_tokens"),
            "mamba_full_memory_ratio": (info.get("internal_states", {}) or {})
            .get("server_args", {})
            .get("mamba_full_memory_ratio"),
        }
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def percentile(values, q):
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def run_sweep_cell(base, conc, ctx_tokens, args) -> dict:
    tool_tokens = max(2048, ctx_tokens // 16)
    with ThreadPoolExecutor(max_workers=conc) as pool:
        t0 = time.monotonic()
        results = list(
            pool.map(
                lambda w: run_agent_session(
                    base,
                    worker=w,
                    context_tokens=ctx_tokens,
                    tool_tokens=tool_tokens,
                    num_tools=args.num_tools,
                    decode_tokens=args.decode_tokens,
                    timeout_s=args.request_timeout_s,
                ),
                range(conc),
            )
        )
    wall = time.monotonic() - t0
    itls = [x for r in results for x in r.itls]
    total_out = sum(r.output_tokens for r in results)
    return {
        "concurrency": conc,
        "context_tokens": ctx_tokens,
        "wall_s": wall,
        "output_tokens_per_s": total_out / wall,
        "ttft_p50_s": percentile([r.ttft_s for r in results], 0.5),
        "ttft_p99_s": percentile([r.ttft_s for r in results], 0.99),
        "itl_mean_ms": 1000 * sum(itls) / max(len(itls), 1),
        "itl_p50_ms": 1000 * percentile(itls, 0.5),
        "itl_p99_ms": 1000 * percentile(itls, 0.99),
    }


def run_serve_cell(base, conc, input_len, args) -> dict:
    """Closed-loop bench_serve-style cell: `conc` requests in flight at all
    times until --num-prompts complete. Each prompt = one long shared prefix
    (--shared-prefix-len, the prefix-cache reality of agent deployments) +
    a random unique tail to --input-len. Streaming, so TTFT/ITL are real."""
    import random as _random
    import threading

    import requests

    rng = _random.Random(1234)
    shared_prefix = build_context(args.shared_prefix_len, seed=99)
    lock = threading.Lock()
    remaining = [args.num_prompts]
    ttfts, itls, out_tokens = [], [], []

    def one_request():
        while True:
            with lock:
                if remaining[0] <= 0:
                    return
                remaining[0] -= 1
            tail = build_context(max(input_len - args.shared_prefix_len, 64), seed=rng.randrange(1 << 30))
            prompt = shared_prefix + "\n" + tail + "\nSummarize the above in one sentence."
            t0 = time.monotonic()
            ttft = None
            last = None
            count = 0
            resp = requests.post(
                base + "/v1/completions",
                json={
                    "model": "default",
                    "prompt": prompt,
                    "max_tokens": args.output_len,
                    "temperature": 0,
                    "stream": True,
                },
                stream=True,
                timeout=args.request_timeout_s,
            )
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                payload = line[len(b"data: "):]
                if payload == b"[DONE]":
                    break
                delta = json.loads(payload)["choices"][0]
                now = time.monotonic()
                text = (delta.get("text") or "").strip()
                if text and ttft is None:
                    ttft = now - t0
                elif text and last is not None:
                    itls.append(now - last)
                if text:
                    last = now
                    count += 1
            with lock:
                if ttft is not None:
                    ttfts.append(ttft)
                out_tokens.append(max(count, 1))

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=conc) as pool:
        list(pool.map(lambda _: one_request(), range(conc)))
    wall = time.monotonic() - t0
    return {
        "concurrency": conc,
        "context_tokens": input_len,
        "wall_s": wall,
        "output_tokens_per_s": sum(out_tokens) / wall,
        "ttft_p50_s": percentile(ttfts, 0.5),
        "ttft_p99_s": percentile(ttfts, 0.99),
        "itl_mean_ms": 1000 * sum(itls) / max(len(itls), 1),
        "itl_p50_ms": 1000 * percentile(itls, 0.5),
        "itl_p99_ms": 1000 * percentile(itls, 0.99),
        "itl_per_stream_p50_ms": 1000 * percentile(itls, 0.5),
    }


def count_retractions(log_path: str) -> int:
    try:
        with open(log_path, errors="ignore") as f:
            return sum(
                1
                for line in f
                if "etract" in line and "retraction_policy" not in line
            )
    except OSError:
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", default=None, help="MTP draft weights (omit when the checkpoint embeds mtp.* tensors)")
    parser.add_argument("--mtp", default="3/1/4", help="steps/topk/draft-tokens")
    parser.add_argument("--kv-cache-dtype", default="fp8_e4m3")
    parser.add_argument("--mem-fraction-static", type=float, default=0.88)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 4, 8, 16, 24])
    parser.add_argument("--contexts", type=int, nargs="+", default=[8192, 32768, 65536, 131072])
    parser.add_argument("--runs", nargs="+", default=["baseline", "auto"],
                        help="feature configs: baseline auto adaptive auto+adaptive <any>+mirror")
    parser.add_argument("--mamba-ratio", type=float, default=None,
                        help="fixed --mamba-full-memory-ratio for non-auto runs (default server 0.9)")
    parser.add_argument("--decode-latency-budget-ms", type=float, default=50.0)
    parser.add_argument("--nvfp4-mirror-fraction", type=float, default=1.0)
    parser.add_argument("--num-tools", type=int, default=3)
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--launch-timeout-s", type=float, default=1800)
    parser.add_argument("--request-timeout-s", type=float, default=3600)
    parser.add_argument("--skip-cells", type=int, nargs="*", default=[],
                        help="ctx values to skip (e.g. contexts that exceed VRAM at high c)")
    parser.add_argument("--extra-server-args", nargs="*", default=[])
    parser.add_argument("--mode", default="agent", choices=["agent", "serve"],
                        help="agent: multi-turn sessions; serve: closed-loop bench_serve-"
                             "style with a long shared prefix (deployment-realistic)")
    parser.add_argument("--input-len", type=int, default=16384, help="serve mode: prompt length")
    parser.add_argument("--output-len", type=int, default=1024, help="serve mode: generation length")
    parser.add_argument("--shared-prefix-len", type=int, default=0,
                        help="serve mode: tokens of shared prefix across all prompts "
                             "(models the prefix-cache-hit reality of agent deployments)")
    parser.add_argument("--num-prompts", type=int, default=64, help="serve mode: total requests")
    args = parser.parse_args()

    all_results = {}
    for run_name in args.runs:
        log_path = f"/tmp/sgl_agent_bench_{run_name}.log"
        proc, base = launch_server(args, run_name, log_path)
        try:
            derived = server_derived_config(base)
            print(f"[{run_name}] server config: {derived}")
            contexts = [args.input_len] if args.mode == "serve" else args.contexts
            for ctx in contexts:
                if ctx in args.skip_cells:
                    continue
                for conc in args.concurrencies:
                    if args.mode == "serve":
                        cell = run_serve_cell(base, conc, ctx, args)
                    else:
                        cell = run_sweep_cell(base, conc, ctx, args)
                    retractions = count_retractions(log_path)
                    cell.update(run=run_name, retractions=retractions, **{
                        k: v for k, v in derived.items() if k != "error"
                    })
                    all_results[(run_name, conc, ctx)] = cell
                    print(
                        f"[{run_name}] c={conc:2d} ctx={ctx:6d} "
                        f"out tok/s={cell['output_tokens_per_s']:8.1f} "
                        f"ttft_p99={cell['ttft_p99_s']:7.2f}s "
                        f"itl_p99={cell['itl_p99_ms']:8.1f}ms "
                        f"retractions={retractions}"
                        + ("  ** FAIL (retraction>0) **" if retractions > 0 else "")
                    )
        finally:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            time.sleep(10)

    # Comparison table: per (c, ctx), relative throughput of each run vs the
    # first run.
    first = args.runs[0]
    print("\n==== throughput vs '%s' (output tokens/s, +%%) ====" % first)
    header = f"{'c':>3s} {'ctx':>7s}" + "".join(f"{r:>14s}" for r in args.runs)
    print(header)
    for ctx in args.contexts:
        for conc in args.concurrencies:
            base_cell = all_results.get((first, conc, ctx))
            if base_cell is None:
                continue
            row = f"{conc:3d} {ctx:7d}"
            for r in args.runs:
                cell = all_results.get((r, conc, ctx))
                if cell is None:
                    row += f"{'-':>14s}"
                else:
                    rel = (
                        cell["output_tokens_per_s"]
                        / base_cell["output_tokens_per_s"]
                        - 1
                    )
                    row += f"{rel * 100:+13.1f}%"
            print(row)

    out_json = os.path.abspath("agent_bench_results.json")
    with open(out_json, "w") as f:
        json.dump(
            [
                {"run": r, **v}
                for (r, _, _), v in sorted(all_results.items())
            ],
            f,
            indent=2,
        )
    print(f"\nfull results written to {out_json}")


if __name__ == "__main__":
    main()
