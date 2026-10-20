"""Submit N distinct large requests concurrently with a decode profile window.

Draws distinct prompts from the recorded workload corpus (no shared prefixes,
unlike resubmitting one prompt N times), submits them concurrently, arms the
stage-separated torch profiler mid-decode, and records per-request usage.
Stdlib only; run on the host against the mapped server port.
"""

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def select_requests(workload, count, min_tokens, max_tokens):
    rows = [json.loads(line) for line in open(workload)]
    band = [r for r in rows if min_tokens <= r["observed_prompt_tokens"] <= max_tokens]
    if len(band) < count:
        raise SystemExit(f"only {len(band)} requests in [{min_tokens},{max_tokens}] "
                         f"tokens; need {count}")
    band.sort(key=lambda r: r["observed_prompt_tokens"])
    step = (len(band) - 1) / max(count - 1, 1)
    picked = [band[round(i * step)] for i in range(count)]
    seen, distinct = set(), []
    for row in picked:
        if row["request_sha256"] in seen:
            continue
        seen.add(row["request_sha256"])
        distinct.append(row)
    if len(distinct) < count:
        raise SystemExit("selection produced duplicate requests")
    return distinct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--min-tokens", type=int, default=25_000)
    parser.add_argument("--max-tokens", type=int, default=40_000)
    parser.add_argument("--max-tokens-out", type=int, default=192)
    parser.add_argument("--profile-steps", type=int, default=64)
    parser.add_argument("--prefill-wait-seconds", type=float, default=45.0,
                        help="delay before arming the profiler so the window "
                             "lands in steady decode, past chunked prefill")
    parser.add_argument("--base", default="http://127.0.0.1:30000")
    parser.add_argument("--output", type=Path, default=Path("batch-profile-summary.json"))
    args = parser.parse_args()

    rows = select_requests(args.workload, args.requests,
                           args.min_tokens, args.max_tokens)
    print(json.dumps({"selected": [{r["request_sha256"][:12]: r["observed_prompt_tokens"]}
                                   for r in rows]}), flush=True)

    deadline = time.monotonic() + 3600
    while True:
        try:
            with urllib.request.urlopen(args.base + "/v1/models", timeout=5) as response:
                if response.status == 200:
                    break
        except OSError:
            pass
        if time.monotonic() >= deadline:
            raise SystemExit("server did not become healthy in time")
        time.sleep(10)
    print("server ready", flush=True)

    results = [None] * len(rows)

    def submit(index, row):
        payload = dict(row["request"])
        payload["max_tokens"] = args.max_tokens_out
        request = urllib.request.Request(
            args.base + "/v1/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        start = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=3600) as response:
                body = json.loads(response.read())
            results[index] = {
                "sha": row["request_sha256"][:12],
                "prompt_tokens": row["observed_prompt_tokens"],
                "usage": body.get("usage"),
                "elapsed_s": round(time.monotonic() - start, 2),
                "finish": [c.get("finish_reason") for c in body.get("choices", [])],
            }
        except urllib.error.HTTPError as error:
            results[index] = {"sha": row["request_sha256"][:12], "error": error.read().decode()[:300]}

    workers = [threading.Thread(target=submit, args=(i, r)) for i, r in enumerate(rows)]
    for worker in workers:
        worker.start()
    time.sleep(args.prefill_wait_seconds)
    profile_req = urllib.request.Request(
        args.base + "/start_profile",
        data=json.dumps({"output_dir": "/profiles", "num_steps": args.profile_steps,
                         "activities": ["GPU"], "profile_by_stage": True}).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(profile_req, timeout=30) as response:
            print("start_profile:", response.read().decode()[:200], flush=True)
    except urllib.error.HTTPError as error:
        print("start_profile failed:", error.read().decode()[:300], flush=True)
    for worker in workers:
        worker.join()
    try:
        urllib.request.urlopen(args.base + "/stop_profile", timeout=60).read()
    except (urllib.error.HTTPError, OSError):
        pass  # by-stage profiling auto-stops after num_steps

    summary = {"requests": args.requests, "results": results}
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
