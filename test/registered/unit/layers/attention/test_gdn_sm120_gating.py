"""CPU-only unit tests for SM120 (RTX PRO 6000 Blackwell) GDN enablement.

Pins the backend-selection gates: FlashInfer GDN prefill becomes the
auto-default on SM120 (flashinfer ships an SM120 chunked-prefill DSL
kernel; the fp32-initial-state quirk is handled by the kernel wrapper),
and FlashInfer target-verify stays Triton-by-default there unless opted
in via SGLANG_GDN_FLASHINFER_VERIFY_SM120.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _model_runner(state_dtype, key_dim=128, value_dim=128):
    return SimpleNamespace(
        model_config=SimpleNamespace(),
        req_to_token_pool=SimpleNamespace(
            mamba_pool=SimpleNamespace(
                mamba_cache=SimpleNamespace(temporal=SimpleNamespace(dtype=state_dtype))
            )
        ),
    )


class TestPrefillDefaultOnSM120(unittest.TestCase):
    def _publish(self):
        from sglang.srt import runtime_context as rc

        return rc.get_context().override_server_args(
            chunked_prefill_size=8192,
            enable_dynamic_chunking=False,
            linear_attn_backend="triton",
        )

    def _default(self, state_dtype, *, cap=(12, 0), cuda_version="13.0"):
        from sglang.srt.layers.attention.linear.gdn_backend import (
            flashinfer_gdn_prefill_default,
        )
        import sglang.srt.layers.attention.linear.gdn_backend as backend_mod

        with self._publish():
            with (
                mock.patch.object(
                    backend_mod,
                    "hybrid_gdn_config",
                    return_value=SimpleNamespace(
                        linear_key_head_dim=128, linear_value_head_dim=128
                    ),
                ),
                mock.patch(
                    "sglang.srt.layers.attention.linear.kernels.gdn_flashinfer."
                    "is_flashinfer_gdn_prefill_available",
                    return_value=True,
                ),
                mock.patch.object(backend_mod, "is_cuda", return_value=True),
                mock.patch.object(
                    torch.cuda, "get_device_capability", return_value=cap
                ),
                mock.patch.object(torch.version, "cuda", cuda_version),
            ):
                return flashinfer_gdn_prefill_default(_model_runner(state_dtype))

    def test_sm120_bf16_state_gets_flashinfer_prefill(self):
        self.assertEqual(self._default(torch.bfloat16), "flashinfer")

    def test_sm120_fp32_state_gets_flashinfer_prefill(self):
        self.assertEqual(self._default(torch.float32), "flashinfer")

    def test_sm120_cuda12_stays_triton(self):
        # The SM120 chunk kernel is CuTe-DSL and needs CUDA >= 13.
        self.assertIsNone(
            self._default(torch.bfloat16, cuda_version="12.8")
        )

    def test_sm100_still_requires_bf16(self):
        self.assertIsNone(self._default(torch.float32, cap=(10, 0)))
        self.assertEqual(self._default(torch.bfloat16, cap=(10, 0)), "flashinfer")

    def test_sm90_unchanged(self):
        self.assertEqual(
            self._default(torch.float32, cap=(9, 0), cuda_version="12.8"),
            "flashinfer",
        )
        self.assertIsNone(
            self._default(torch.bfloat16, cap=(9, 0), cuda_version="12.8")
        )


class TestVerifyGating(unittest.TestCase):
    def test_default_gates(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
            flashinfer_gdn_verify_supported,
        )

        with envs.SGLANG_GDN_FLASHINFER_VERIFY_SM120.override(False):
            self.assertTrue(flashinfer_gdn_verify_supported(9))
            self.assertTrue(flashinfer_gdn_verify_supported(10))
            self.assertFalse(flashinfer_gdn_verify_supported(12))
            self.assertFalse(flashinfer_gdn_verify_supported(8))
        with envs.SGLANG_GDN_FLASHINFER_VERIFY_SM120.override(True):
            self.assertTrue(flashinfer_gdn_verify_supported(12))


if __name__ == "__main__":
    unittest.main()
