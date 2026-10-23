"""HKV-27 volley: sequential 101k finishes against the tier server.

One request at a time so every forget# ledger line maps to exactly one
finish. Prints one line per request; the per-forget ledger lines live in
the docker log.
"""
import json
import random
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
N_REQ = int(sys.argv[2]) if len(sys.argv) > 2 else 45
IN_TOKENS = 101043
OUT_TOKENS = 512

random.seed(20261023)
prompt_ids = [random.randrange(1, 150000) for _ in range(IN_TOKENS)]

url = f"http://127.0.0.1:{PORT}/generate"
for i in range(N_REQ):
    payload = json.dumps({
        "input_ids": prompt_ids,
        "sampling_params": {
            "max_new_tokens": OUT_TOKENS,
            "ignore_eos": True,
            "temperature": 0.0,
        },
    }).encode()
    t0 = time.time()
    try:
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1200) as r:
            out = json.loads(r.read())
        meta = out.get("meta_info", {})
        print(f"req {i:2d}: wall={time.time()-t0:6.1f}s "
              f"finish={meta.get('finish_reason', {}).get('type', '?')} "
              f"completion={len(out.get('text', ''))}ch", flush=True)
    except Exception as e:
        print(f"req {i:2d}: ERR {type(e).__name__} {e}", flush=True)
        time.sleep(5)
print("VOLLEY-DONE", flush=True)
