import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from replay import load_capture
import runtime_capture as capture


def inputs(rows=4, heads=8):
    positions = torch.arange(2050, 2050 + rows, dtype=torch.int64)
    valid = (positions + 1) // 4
    n = int(valid[-1])
    pages = (n + 7) // 8
    cache = (torch.arange((pages + 1) * 8 * 128).reshape(pages + 1, 8, 1, 128) % 97).to(torch.bfloat16)
    mapping = torch.arange(pages, 0, -1, dtype=torch.int64).expand(rows, -1).clone()
    q = torch.zeros((rows, heads, 128), dtype=torch.bfloat16)
    q[:, :4] = torch.randn((rows, 4, 128)).to(torch.bfloat16)
    logits = torch.arange(n, dtype=torch.float32).expand(rows, -1).clone()
    selected = []
    for row, length in enumerate(valid):
        logits[row, int(length):] = -torch.inf
        selected.append(torch.topk(logits[row], 512).indices)
    return dict(q=q, compressed_cache=cache, page_table=mapping, lengths=valid,
                query_positions=positions, native_scores=logits,
                native_selected_ids=torch.stack(selected), row_request_ids=["stable-http-rid"] * rows,
                layer_id=3, forward_mode="target_verify")


def session(path, **kwargs):
    return capture.CaptureSession(path, model_id="test-model", model_revision="model-sha",
                                  runtime_revision="runtime-sha", **kwargs)


class RuntimeCaptureTest(unittest.TestCase):
    def test_disabled_and_rank_do_not_access_inputs_or_files(self):
        class Untouchable:
            def __getattribute__(self, name):
                raise AssertionError("disabled capture touched a tensor")
        value = Untouchable()
        data = dict(q=value, compressed_cache=value, page_table=value, lengths=value,
                    query_positions=value, native_scores=value, native_selected_ids=value,
                    row_request_ids=value, layer_id=0, forward_mode="decode")
        with patch.dict(os.environ, {}, clear=True), patch.object(capture.Path, "mkdir", side_effect=AssertionError):
            self.assertEqual(capture.capture_qsa(**data)["status"], "disabled")
            self.assertEqual(capture.capture_acceptance(request_ids=value, accepted_drafts=value, draft_positions=value)["status"], "disabled")
        with patch.dict(os.environ, {capture.PREFIX + "DIR": "/unused"}, clear=True):
            self.assertEqual(capture.capture_qsa(**data, rank=1)["status"], "nonzero_rank")
            with self.assertRaises(ValueError):
                capture.capture_qsa(**data)
        with patch.dict(os.environ, {capture.PREFIX + "DIR": "/unused", "SGLANG_SIMULATE_ACC_LEN": "2"}, clear=True):
            with self.assertRaisesRegex(ValueError, "SIMULATE"):
                capture.capture_qsa(**data)

    def test_nonidentity_gather_causal_group_and_native_values(self):
        with tempfile.TemporaryDirectory() as directory:
            data = inputs()
            original = {name: value.clone() for name, value in data.items() if torch.is_tensor(value)}
            result = session(directory).capture_qsa(**data)
            self.assertEqual(result["status"], "captured")
            self.assertEqual(len(result["files"]), 1)
            meta, arrays = load_capture(result["files"][0])
            self.assertEqual(meta["request_id"], "stable-http-rid")
            self.assertEqual(meta["source_kind"], "model_capture")
            self.assertEqual(meta["model_revision"], "model-sha")
            np.testing.assert_array_equal(arrays["valid_counts"], [512, 513, 513, 513])
            expected = data["compressed_cache"][data["page_table"][-1], :, 0, :].reshape(-1, 128)[:513].float().numpy()
            np.testing.assert_array_equal(arrays["keys"], expected)
            np.testing.assert_array_equal(arrays["queries"], data["q"][:, :4].float().numpy())
            np.testing.assert_array_equal(arrays["native_scores"], data["native_scores"].numpy())
            for name, value in original.items():
                self.assertTrue(torch.equal(data[name], value), name)
            before = Path(result["files"][0]).read_bytes()
            another = session(directory).capture_qsa(**data)
            self.assertNotEqual(result["files"], another["files"])
            self.assertEqual(Path(result["files"][0]).read_bytes(), before)

    def test_budgets_before_tensor_copy_and_existing_file_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            data = inputs()
            for limits in ({"max_files": 0}, {"max_bytes": 1}):
                with patch.object(capture, "_integers", side_effect=AssertionError("copied before budget")):
                    result = session(directory, **limits).capture_qsa(**data)
                    self.assertEqual(result["status"], "skipped")
            one = session(directory, max_files=1).capture_qsa(**data)
            self.assertEqual(len(one["files"]), 1)
            with patch.object(capture, "_bf16_export", side_effect=AssertionError("exceeded file budget")):
                result = session(directory, max_files=1).capture_qsa(**data)
                self.assertEqual(result["skipped"], {"file_budget": 1})
                result = session(directory, max_bytes=500_000).capture_qsa(**data)
                self.assertEqual(result["skipped"], {"byte_budget": 1})
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(capture, "_bf16_export", side_effect=AssertionError("copied beyond byte/key budget")):
                result = session(directory, max_bytes=350_000).capture_qsa(**inputs())
                self.assertEqual(result["skipped"], {"byte_budget": 1})
                result = session(directory, max_keys=512).capture_qsa(**inputs())
                self.assertEqual(result["skipped"], {"key_budget": 1})

    def test_invalid_native_inputs_and_graph_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            base = inputs()
            invalid = []
            data = inputs(); data["native_selected_ids"][0, 0] = data["native_selected_ids"][0, 1]; invalid.append(data)
            data = inputs(); data["page_table"][0, 0] -= 1; invalid.append(data)
            data = inputs(); data["q"][0, 7, 0] = 1; invalid.append(data)
            data = inputs(); data["query_positions"][1] += 1; invalid.append(data)
            data = inputs(); data["q"] = data["q"].float(); invalid.append(data)
            data = inputs(); data["native_scores"] = data["native_scores"].to(torch.bfloat16); invalid.append(data)
            data = inputs(); data["row_request_ids"] = ["a", "b", "a", "b"]; invalid.append(data)
            for data in invalid:
                with self.assertRaises(ValueError):
                    session(directory).capture_qsa(**data)
            self.assertEqual(list(Path(directory).glob("*.npz")), [])
            with patch.object(capture, "_graph_guard", side_effect=RuntimeError("graph")), patch.object(capture, "_integers", side_effect=AssertionError):
                with self.assertRaisesRegex(RuntimeError, "graph"):
                    session(directory).capture_qsa(**base)
            result = session(directory).capture_qsa(**(base | {"forward_mode": "draft_extend_v2"}))
            self.assertEqual(result["status"], "unsupported_forward_mode")
            result = session(directory, layers=[99]).capture_qsa(**base)
            self.assertEqual(result["status"], "filtered_layer")
            short = inputs(rows=1, heads=4)
            short["lengths"][:] = 511
            short["query_positions"][:] = 2043
            result = session(directory).capture_qsa(**short)
            self.assertEqual(result["skipped"], {"short_context": 1})

    def test_acceptance_counts_metadata_and_shared_byte_event_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = session(directory, max_events=1)
            counts = torch.tensor([0, 3], dtype=torch.int32)
            result = recorder.capture_acceptance(request_ids=["a", "b"], accepted_drafts=counts,
                                                  draft_positions=4, per_request_metadata=[{"finished_before_output_extension": False}, {"grammar_retained_count": 2}])
            event = json.loads(Path(result["path"]).read_text())
            self.assertEqual(event["accepted_output_slots_including_bonus"], [1, 4])
            self.assertEqual(event["per_request_metadata"][1]["grammar_retained_count"], 2)
            self.assertNotIn("emitted_tokens", event)
            self.assertTrue(torch.equal(counts, torch.tensor([0, 3], dtype=torch.int32)))
            result = session(directory, max_events=1).capture_acceptance(request_ids=["a"], accepted_drafts=[0], draft_positions=4)
            self.assertEqual(result["status"], "event_budget")
            with self.assertRaises(ValueError):
                recorder.capture_acceptance(request_ids=["a"], accepted_drafts=[0], draft_positions=4, per_request_metadata=[{"finished": torch.tensor(False)}])
        with tempfile.TemporaryDirectory() as directory:
            result = session(directory, max_bytes=0).capture_acceptance(request_ids=["a"], accepted_drafts=[0], draft_positions=4)
            self.assertEqual(result["status"], "byte_budget")
            self.assertFalse((Path(directory) / "acceptance.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
