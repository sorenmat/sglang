"""G3 sender: 32 concurrent requests split across the two tier servers.

Contexts are built from the corpus by repetition with distinct suffixes
(the corpus's real long requests are ~170k max; the target context is
set by Ctx_TOKENS, default 262144 — requests are padded by repeating
the largest real request's task text with unique markers so each is a
distinct prompt). Reports per-request tok/s and elapsed.
"""
import json
import threading
import time
import urllib.request
from pathlib import Path

CTX_TOKENS = int(__import__("os").environ.get("G3_CTX_TOKENS", "262144"))
TOTAL = 32
PER_SERVER = TOTAL // 2
OUT = Path("/home/ecuser/work/g3-results-live.json")
SERVERS = ["http://127.0.0.1:30010", "http://127.0.0.1:30011"]

rows = [json.loads(l) for l in open("/home/ecuser/work/assets/workload.jsonl")]
base = max(rows, key=lambda r: r["observed_prompt_tokens"])
# estimate: chars/token ~ 3.5 for this corpus; build to target
target_chars = int(CTX_TOKENS * 2.7)
msg = base["request"]["messages"]
task = next(m["content"] for m in msg if m.get("role") == "user")

results = [None] * TOTAL

def build(i):
    filler = (f"\n\n[SESSION {i} | PART {n}]\n" + task for n in range(10_000))
    parts = []
    total = 0
    for n in range(10_000):
        chunk = f"\n\n[SESSION {i} | PART {n}]\n" + task
        parts.append(chunk)
        total += len(chunk)
        if total >= target_chars:
            break
    return "".join(parts)

def wait(server):
    deadline = time.monotonic() + 7200
    while True:
        try:
            with urllib.request.urlopen(server + "/v1/models", timeout=5) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        if time.monotonic() > deadline:
            raise RuntimeError(f"{server} not healthy")
        time.sleep(10)

for s in SERVERS:
    wait(s)
print("both servers ready", flush=True)

def submit(i):
    server = SERVERS[i // PER_SERVER]
    content = build(i)
    payload = {"model": "flash-next-nvfp4",
               "messages": [{"role": "system", "content": msg[0]["content"]},
                            {"role": "user", "content": content}],
               "max_tokens": 256, "temperature": 0}
    req = urllib.request.Request(server + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.monotonic()
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=14400).read())
        elapsed = time.monotonic() - t
        comp = r["usage"]["completion_tokens"]
        prompt = r["usage"]["prompt_tokens"]
        results[i] = {"i": i, "server": i // PER_SERVER, "prompt": prompt,
                      "completion": comp, "elapsed": round(elapsed, 1),
                      "tok_s": round(comp / elapsed, 2) if elapsed > 0 else None,
                      "finish": r["choices"][0].get("finish_reason")}
    except Exception as e:
        results[i] = {"i": i, "server": i // PER_SERVER, "error": repr(e)[:200]}

threads = [threading.Thread(target=submit, args=(i,)) for i in range(TOTAL)]
start = time.monotonic()
for t in threads:
    t.start()
for t in threads:
    t.join()
wall = time.monotonic() - start

ok = [r for r in results if r and "error" not in r]
summary = {"total": TOTAL, "ok": len(ok), "wall_s": round(wall, 1),
           "prompt_tokens_seen": sorted(r["prompt"] for r in ok)[:4],
           "tok_s": sorted((r["tok_s"] for r in ok), reverse=True)[:8],
           "min_tok_s": min((r["tok_s"] for r in ok), default=None),
           "median_tok_s": sorted(r["tok_s"] for r in ok)[len(ok)//2] if ok else None}
OUT.write_text(json.dumps({"summary": summary, "results": results}, indent=2) + "\n")
print(json.dumps(summary), flush=True)
