"""HKV-9: collect real expert-route distributions for the capacity model.

Serves the model with --enable-return-routed-experts, replays N distinct
corpus requests, extracts per-token routed experts from the responses, and
emits the measured union-vs-batch growth curve that replaces the analytic
expert-union term in phase1_model.py.

Run on the GPU host (needs the serving port; stdlib only).
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path


def select_requests(workload, count, min_tokens, max_tokens):
    rows = [json.loads(line) for line in open(workload)]
    band = [r for r in rows if min_tokens <= r["observed_prompt_tokens"] <= max_tokens]
    band.sort(key=lambda r: r["observed_prompt_tokens"])
    if len(band) < count:
        raise SystemExit(f"only {len(band)} requests in band; need {count}")
    step = (len(band) - 1) / max(count - 1, 1)
    return [band[round(i * step)] for i in range(count)]


def extract_experts(response_json):
    """Pull routed expert ids from a response that carries them.

    The runtime exposes them base64-encoded under sglext.routed_experts
    (int32 token-major ids), and ONLY when the per-request flag
    return_routed_experts=true is set alongside the server flag."""
    import base64
    import struct
    raw = (response_json.get("sglext") or {}).get("routed_experts")
    if not raw:
        return None
    data = base64.b64decode(raw)
    flat = list(struct.unpack(f"<{len(data) // 4}i", data))
    return [flat[i:i + 10] for i in range(0, len(flat), 10)]


def union_curve(token_expert_sets):
    """Growth of the distinct-expert union as tokens accumulate."""
    seen, per_token = set(), []
    for experts in token_expert_sets:
        seen.update(experts)
        per_token.append(len(seen))
    return per_token


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--min-tokens", type=int, default=25_000)
    parser.add_argument("--max-tokens", type=int, default=40_000)
    parser.add_argument("--max-tokens-out", type=int, default=64)
    parser.add_argument("--base", default="http://127.0.0.1:30000")
    parser.add_argument("--output", type=Path,
                        default=Path("route-distributions.json"))
    args = parser.parse_args()

    rows = select_requests(args.workload, args.requests,
                           args.min_tokens, args.max_tokens)
    deadline = time.monotonic() + 5400
    while True:
        try:
            with urllib.request.urlopen(args.base + "/v1/models", timeout=5) as r:
                if r.status == 200:
                    break
        except OSError:
            pass
        if time.monotonic() > deadline:
            raise SystemExit("server not healthy")
        time.sleep(10)
    print("server ready", flush=True)

    collected = []
    for row in rows:
        payload = dict(row["request"])
        payload["max_tokens"] = args.max_tokens_out
        payload["temperature"] = 0
        payload["return_routed_experts"] = True
        request = urllib.request.Request(
            args.base + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=3600) as response:
            body = json.loads(response.read())
        experts = extract_experts(body)
        collected.append({
            "sha": row["request_sha256"][:12],
            "prompt_tokens": row["observed_prompt_tokens"],
            "experts_present": experts is not None,
            "token_experts": experts or [],
            "note": "" if experts is not None else
            "response carried no routed-expert field: confirm the server "
            "was launched with --enable-return-routed-experts and the "
            "runtime's response schema for it",
        })
        print(f"{row['request_sha256'][:12]}: experts={'yes' if experts else 'NO'}", flush=True)

    present = [c for c in collected if c["experts_present"]]
    report = {
        "status": "measured route distributions" if present else
        "collection failed: no routed-expert fields in responses",
        "requests": len(collected),
        "with_expert_data": len(present),
        "collected": collected,
    }
    if present:
        all_token_experts = [e for c in present for e in c["token_experts"]]
        report["union_curve"] = union_curve(all_token_experts)
        report["distinct_experts_total"] = report["union_curve"][-1] if report["union_curve"] else 0
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("status", "requests", "with_expert_data")}), flush=True)


if __name__ == "__main__":
    main()
