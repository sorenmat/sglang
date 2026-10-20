"""Send the pinned long request once; write response JSON. Usage: p22_send.py a|b"""
import json
import sys
import time
import urllib.request
from pathlib import Path

tag = sys.argv[1]
base = "http://127.0.0.1:30000"
rows = [json.loads(l) for l in open("/root/work/assets/workload.jsonl")]
big = max(rows, key=lambda r: r["observed_prompt_tokens"])
payload = dict(big["request"])
payload["max_tokens"] = 64
payload["temperature"] = 0
# Force far-back selections: without this, decode attention is purely
# recency-biased and never demands an evicted (old) page.
messages = payload.get("messages", [])
for message in reversed(messages):
    if message.get("role") == "user":
        message["content"] = (message["content"] +
                              "\n\nIMPORTANT: Begin your reply by quoting "
                              "verbatim the first two sentences of the task "
                              "description at the very top of this conversation.")
        break

deadline = time.monotonic() + 5400
while True:
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=5) as r:
            if r.status == 200:
                break
    except OSError:
        pass
    if time.monotonic() > deadline:
        raise SystemExit("server not healthy")
    time.sleep(10)
print("server ready", flush=True)

req = urllib.request.Request(base + "/v1/chat/completions",
                             data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
start = time.monotonic()
with urllib.request.urlopen(req, timeout=3600) as r:
    body = json.loads(r.read())
elapsed = time.monotonic() - start
out = {"tag": tag, "elapsed_s": round(elapsed, 2), "usage": body.get("usage"),
       "finish": body.get("choices", [{}])[0].get("finish_reason"),
       "content": body.get("choices", [{}])[0].get("message", {}).get("content"),
       "prompt_tokens": big["observed_prompt_tokens"]}
Path(f"/root/work/job/response-{tag}.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({k: out[k] for k in ("tag", "elapsed_s", "usage", "finish")}), flush=True)
