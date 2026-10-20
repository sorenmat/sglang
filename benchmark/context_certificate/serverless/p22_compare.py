"""P2.2 gate verdict: compare run A (baseline) and run B (90% pages evicted)."""
import json
from pathlib import Path

a = json.loads(Path("/root/work/job/response-a.json").read_text())
b = json.loads(Path("/root/work/job/response-b.json").read_text())
stats_path = Path("/root/work/job/hosttier-stats.json")
stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

identical = a["content"] == b["content"]
report = {
    "gate": "P2.2 correctness slice",
    "prompt_tokens": a["prompt_tokens"],
    "token_streams_identical": identical,
    "finish": {"a": a["finish"], "b": b["finish"],
               "same": a["finish"] == b["finish"]},
    "elapsed_s": {"a": a["elapsed_s"], "b": b["elapsed_s"]},
    "hosttier": {k: stats.get(k) for k in
                 ("snapshotted", "staged_pages", "staged_bytes",
                  "stage_calls", "stage_seconds", "still_evicted_pages")},
    "verdict": "PASS" if identical else "FAIL",
}
print(json.dumps(report, indent=2))
Path("/root/work/job/p22-verdict.json").write_text(json.dumps(report, indent=2) + "\n")
