import copy
import json
from pathlib import Path
import tempfile
import unittest

from workload import build, extract, tool_schema


class WorkloadTest(unittest.TestCase):
    def test_historical_request_not_answer_and_dedup(self):
        response = {"role": "assistant", "content": "DO_NOT_LEAK_SELECTED_ANSWER", "extra": {"response": {"choices": [{"finish_reason": "stop"}], "usage": {"prompt_tokens": 99, "completion_tokens": 4}}}}
        data = {
            "trajectory_format": "mini-swe-agent-1.1", "instance_id": "task-a",
            "info": {"mini_version": "2.4.6", "config": {"model_type": "minisweagent.models.litellm_model.LitellmModel", "model": {"model_name": "openai/test", "model_kwargs": {"parallel_tool_calls": True, "drop_params": True}}}},
            "messages": [
                {"role": "system", "content": "Shared instructions"},
                {"role": "user", "content": "Task"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1", "index": 0, "type": "function", "function": {"name": "bash", "arguments": "{}"}}], "extra": {"response": {"choices": [{}], "usage": {"prompt_tokens": 10}}}},
                {"role": "tool", "tool_call_id": "call-1", "content": "Original tool output", "extra": {"raw_output": "DO_NOT_LEAK_METADATA"}},
                response, {"role": "exit", "content": "DO_NOT_LEAK_EXIT"},
            ],
        }
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            source.mkdir()
            good = source / "a.traj.json"
            good.write_text(json.dumps(data))
            duplicate = copy.deepcopy(data)
            duplicate["instance_id"] = "task-b"
            (source / "b.traj.json").write_text(json.dumps(duplicate))
            malformed = copy.deepcopy(data)
            malformed["messages"][3]["tool_call_id"] = "wrong-id"
            (source / "c.traj.json").write_text(json.dumps(malformed))
            (source / "d.traj.json").write_text("not JSON")
            record = extract(good)
            self.assertEqual(record["observed_prompt_tokens"], 99)
            self.assertEqual(record["source"]["request_index"], 1)
            self.assertEqual(record["source"]["response_message_index"], 4)
            self.assertEqual(len(record["request"]["messages"]), 4)
            self.assertNotIn("DO_NOT_LEAK", json.dumps(record))
            self.assertNotIn("extra", record["request"]["messages"][-1])
            self.assertNotIn("index", record["request"]["messages"][2]["tool_calls"][0])
            self.assertIn("tools", record["invocation_metadata"]["unknown_original_parameters"])
            output, summary_path = root / "out.jsonl", root / "summary.json"
            summary = build(source, output, summary_path)
            self.assertEqual(summary["requests_selected"], 1)
            self.assertEqual(len(summary["skipped"]), 3)
            before = output.read_bytes()
            build(source, output, summary_path)
            self.assertEqual(before, output.read_bytes())
            schema = root / "schema.py"
            schema.write_text("raise RuntimeError('never execute')\nBASH_TOOL = {'type': 'function', 'function': {'name': 'bash'}}\n")
            tools, provenance = tool_schema(schema)
            inferred = extract(good, tools, provenance)
            self.assertEqual(inferred["request"]["tools"], tools)
            self.assertNotIn("tools", inferred["invocation_metadata"]["unknown_original_parameters"])


if __name__ == "__main__":
    unittest.main()
