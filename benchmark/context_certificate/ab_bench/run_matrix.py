#!/usr/bin/env python3
"""A/B matrix runner (HKV-22/23).

Usage (on the GPU host, from benchmark/context_certificate/):
  python3 ab_bench/run_matrix.py --port 30030 --out artifacts/ab-bench-<date> \
      [--ladder 1,4,8,16,32] [--ctx 101k,160k] [--modes vanilla,tier] \
      [--server-script ../launch-hostkv.sh] [--gpu 0]

Protocol per cell: fire N requests with forced-length decodes (1024
tokens, min_tokens=900 so EOS cannot shorten them), record per-request
timings, scrape the server's decode-log lines for steady-rate bins, and
dump tier stats. Interleaving A/B/A across modes is driven by --repeat.
"""
import argparse
import json
import re
import subprocess
import time
import urllib.request
from pathlib import Path

CORPUS_TASK = ("alpha beta gamma delta epsilon zeta eta theta iota kappa "
               "lambda mu nu " * 3000)


def build_prompt(tokens_target: int, session: int) -> str:
    chars = int(tokens_target * 2.7)
    body = f"[SESSION {session}] {CORPUS_TASK}"
    while len(body) < chars:
        body += f" [PART {session}] {CORPUS_TASK}"
    return body[:chars]


def real_tokens_setting(ctx: str) -> int:
    # volley chars = TOKENS*2.7, real tokens = chars/4.996
    return {"101k": 187000, "160k": 296000}[ctx]


def wait_healthy(port: int, timeout: float = 900) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        time.sleep(10)
    raise RuntimeError(f"server on :{port} not healthy")


def launch(server_script, gpu, port, name, mode, ctx160k, log):
    # Free the port first: a differently-named leftover container
    # holding it makes docker run fail with "port already allocated".
    holders = subprocess.run(
        ["sudo", "-n", "docker", "ps", "-q", "--filter",
         f"publish={port}"], capture_output=True, text=True).stdout.split()
    for cid in holders:
        subprocess.run(["sudo", "-n", "docker", "rm", "-f", cid],
                       capture_output=True)
    res = subprocess.run(
        ["bash", server_script, str(gpu), str(port), name, mode,
         "1600", "1" if ctx160k else "0"],
        capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"launch failed: {res.stderr[-400:]}")
    wait_healthy(port)


def run_cell(port, n, ctx, session_base=0):
    import threading
    target = real_tokens_setting(ctx)
    results = [None] * n

    def submit(i):
        payload = {
            "model": "flash-next-nvfp4",
            "messages": [{"role": "user",
                          "content": build_prompt(target, session_base + i)}],
            "max_tokens": 1024, "min_tokens": 900,
            "temperature": 0, "ignore_eos": True,
            "logprobs": True, "top_logprobs": 1,
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
            results[i] = {
                "elapsed": round(time.monotonic() - t0, 1),
                "prompt": r["usage"]["prompt_tokens"],
                "completion": r["usage"]["completion_tokens"],
                "finish": r["choices"][0].get("finish_reason"),
                "logprobs": [c.get("logprob") for c in r["choices"][0]
                             .get("logprobs", [])][:900],
                "content": r["choices"][0]["message"]["content"][:4000],
            }
        except Exception as e:  # noqa: BLE001
            results[i] = {"error": repr(e)[:200]}

    threads = [threading.Thread(target=submit, args=(i,)) for i in range(n)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return {"wall": round(time.monotonic() - t0, 1), "results": results}


def scrape_decode_bins(container, since_min=10):
    logs = subprocess.check_output(
        ["sudo", "-n", "docker", "logs", container, "--since",
         f"{since_min}m"], text=True, stderr=subprocess.STDOUT)
    bins = {}
    for line in logs.splitlines():
        m = re.search(r"#running-req: (\d+).*gen throughput \(token/s\): "
                      r"([0-9.]+)", line)
        if m:
            running, rate = int(m.group(1)), float(m.group(2))
            if rate > 5:
                bins.setdefault(running, []).append(rate)
    return {r: {"aggregate": round(sum(v) / len(v), 1),
                "per_req": round(sum(v) / len(v) / r, 1),
                "intervals": len(v)}
            for r, v in sorted(bins.items())}


def tier_stats(container):
    out = subprocess.run(
        ["sudo", "-n", "docker", "exec", container, "cat",
         "/root/phase0/host-kv-stats.json"],
        capture_output=True, text=True)
    if out.returncode != 0:
        return None
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return None


def logprob_compare(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None
    kl = sum(abs(x - y) for x, y in pairs) / len(pairs)
    flips = sum(1 for x, y in pairs if abs(x - y) > 1.0)
    return {"mean_abs_logprob_diff": round(kl, 4),
            "big_flip_rate": round(flips / len(pairs), 4), "n": len(pairs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30030)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ladder", default="1,4,8,16,32")
    ap.add_argument("--ctx", default="101k")
    ap.add_argument("--modes", default="vanilla,tier")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--server-script",
                    default=str(Path(__file__).parent.parent
                                / "launch-hostkv.sh"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ladder = [int(x) for x in args.ladder.split(",")]
    ctxs = args.ctx.split(",")
    modes = args.modes.split(",")
    cells = []
    session = 0

    for rep in range(args.repeat):
        for mode in modes:
            for ctx in ctxs:
                name = f"ab-{mode}"
                launch(args.server_script, args.gpu, args.port,
                       name, mode, ctx == "160k", None)
                time.sleep(5)
                for n in ladder:
                    cell = run_cell(args.port, n, ctx, session)
                    session += n
                    cell.update(mode=mode, n=n, ctx=ctx, rep=rep,
                                decode_bins=scrape_decode_bins(name, 8),
                                tier_stats=tier_stats(name))
                    cells.append(cell)
                    print(json.dumps(
                        {"mode": mode, "n": n, "ctx": ctx,
                         "wall": cell["wall"],
                         "ok": sum(1 for r in cell["results"]
                                   if "error" not in r)}), flush=True)
                    time.sleep(30)  # cool-down between cells
                subprocess.run(["sudo", "-n", "docker", "rm", "-f",
                                name], capture_output=True)
                time.sleep(15)

    # Cross-mode logprob comparison per (ctx, n).
    quality = []
    by_key = {}
    for c in cells:
        by_key.setdefault((c["ctx"], c["n"], c["rep"]), {})[c["mode"]] = c
    for (ctx, n, rep), pair in sorted(by_key.items()):
        if "vanilla" in pair and "tier" in pair:
            for i, (a, b) in enumerate(zip(
                    pair["vanilla"]["results"], pair["tier"]["results"])):
                cmp_ = logprob_compare(a.get("logprobs"), b.get("logprobs"))
                if cmp_:
                    quality.append({"ctx": ctx, "n": n, "rep": rep,
                                    "req": i, **cmp_})

    (out / "results.json").write_text(json.dumps(
        {"cells": cells, "quality": quality}, indent=2) + "\n")

    # Markdown summary
    lines = ["# A/B results", "",
             "| ctx | N | mode | wall | ok | steady (running=k) |", "|---|---|---|---|---|---|"]
    for c in cells:
        ok = sum(1 for r in c["results"] if "error" not in r)
        bins = ", ".join(f"{v['per_req']}@k{k}" for k, v in
                         (c["decode_bins"] or {}).items())
        lines.append(f"| {c['ctx']} | {c['n']} | {c['mode']} | "
                     f"{c['wall']}s | {ok}/{c['n']} | {bins} |")
    q_big = [q for q in quality if q["big_flip_rate"] > 0]
    lines += ["", f"Quality comparisons: {len(quality)} "
              f"(big-flip rows: {len(q_big)})"]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out}/results.json and report.md")


if __name__ == "__main__":
    main()
