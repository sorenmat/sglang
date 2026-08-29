"""CPU-only unit tests for DFlash2 accept-buffer rotation.

Under overlap scheduling, step N+1's accept/bonus Triton kernel can launch
before step N's consumer (metrics, FutureMap publish, GenerationBatchResult)
reads the accept results, so every output buffer of the accept computation
must rotate on the double-buffer slot. accept_len was the one exception: it
shared storage across consecutive steps, so a step could read the previous
step's accept lengths -- corrupted commit lengths, wrong outputs, and
apparent request mixing at concurrency. These tests pin the rotation
invariant: consecutive allocations never alias, and all five outputs of one
allocation share the same slot.
"""

import unittest

import torch

from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _worker():
    w = object.__new__(DFlashWorkerV2)
    w.device = torch.device("cpu")
    w.block_size = 4
    w._accept_bonus_buffer_cap = 0
    w._accept_bonus_buffer_slot = 0
    w._accept_len_bufs = []
    w._commit_lens_bufs = []
    w._bonus_id_bufs = []
    w._out_tokens_bufs = []
    w._new_seq_lens_bufs = []
    return w


class TestAcceptBufferRotation(unittest.TestCase):
    def test_consecutive_allocations_do_not_alias(self):
        """The two rotations must be backed by disjoint storage for ALL five
        outputs (accept_len included)."""
        w = _worker()
        first = w._next_accept_bonus_buffers(8)
        second = w._next_accept_bonus_buffers(8)
        for name, a, b in zip(
            (
                "accept_len",
                "commit_lens",
                "bonus",
                "out_tokens",
                "new_seq_lens",
            ),
            first,
            second,
        ):
            self.assertNotEqual(
                a.data_ptr(),
                b.data_ptr(),
                f"{name} aliased across consecutive steps",
            )

    def test_rotation_wraps_at_two(self):
        """Slot cycles 0 -> 1 -> 0; the third allocation reuses slot 0's
        storage (that is the point of double buffering: two in-flight steps)."""
        w = _worker()
        a = w._next_accept_bonus_buffers(4)
        _ = w._next_accept_bonus_buffers(4)
        c = w._next_accept_bonus_buffers(4)
        for x, z in zip(a, c):
            self.assertEqual(x.data_ptr(), z.data_ptr())

    def test_grow_preserves_disjointness(self):
        """Capacity growth re-allocates both slots; the invariant must hold
        across the growth boundary too."""
        w = _worker()
        first = w._next_accept_bonus_buffers(2)
        second = w._next_accept_bonus_buffers(8)  # triggers regrow
        for a, b in zip(first, second):
            self.assertNotEqual(a.data_ptr(), b.data_ptr())
        # ... and the buffers are actually sized for the larger batch.
        for b in second:
            self.assertEqual(b.shape[0], 8)
        # regrow keeps exactly two slots per output family
        self.assertEqual(len(w._accept_len_bufs), 2)
        self.assertEqual(len(w._commit_lens_bufs), 2)
        self.assertEqual(len(w._out_tokens_bufs), 2)

    def test_writes_to_step_n_do_not_corrupt_step_n_plus_one(self):
        """The race the rotation prevents: writing step N+1's accept_len must
        leave step N's accept_len tensor untouched."""
        w = _worker()
        step_n = w._next_accept_bonus_buffers(4)
        step_n1 = w._next_accept_bonus_buffers(4)
        step_n[0].fill_(7)  # accept kernel of step N
        step_n1[0].fill_(1)  # accept kernel of step N+1 overwrites its slot
        self.assertTrue(torch.all(step_n[0] == 7))


if __name__ == "__main__":
    unittest.main()
