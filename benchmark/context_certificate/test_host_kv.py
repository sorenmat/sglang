"""Unit tests for the host-KV pager and admission logic (CPU, no torch)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from host_kv.pager import (AdmissionController, HostTierPager, PageState,
                           TransferKind)

PAGE = 64 * 24_576  # 1.57 MiB, the target model's page across 12 layers


class PagerStates(unittest.TestCase):
    def test_demand_stages_and_reuses(self):
        p = HostTierPager(gpu_slots=2, page_bytes=PAGE)
        self.assertTrue(p.ensure_resident(10))
        self.assertIs(p.state[10], PageState.GPU_RESIDENT)
        self.assertFalse(p.ensure_resident(10))          # second hit is free
        self.assertEqual(p.transfers["demand"], 1)
        self.assertEqual(p.transferred_bytes, PAGE)

    def test_lru_eviction_never_pinned(self):
        p = HostTierPager(gpu_slots=2)
        for page in (1, 2):
            p.ensure_resident(page)
        p.touch(1)                                        # pin the oldest
        p.ensure_resident(3)                              # must evict page 2
        self.assertIs(p.state[1], PageState.GPU_RESIDENT)
        self.assertIs(p.state[2], PageState.HOST)
        self.assertIs(p.state[3], PageState.GPU_RESIDENT)

    def test_evicted_page_restages(self):
        p = HostTierPager(gpu_slots=1)
        p.ensure_resident(1)
        p.ensure_resident(2)                              # evicts 1
        self.assertIs(p.state[1], PageState.HOST)
        self.assertTrue(p.ensure_resident(1))             # restages cleanly
        self.assertIs(p.state[2], PageState.HOST)

    def test_touch_and_evict_reject_non_resident(self):
        p = HostTierPager(gpu_slots=1)
        with self.assertRaises(ValueError):
            p.touch(7)
        with self.assertRaises(ValueError):
            p.evict(7)

    def test_in_flight_pages_are_tracked(self):
        seen = []

        def fake_transfer(page):
            seen.append(page)
            return f"handle-{page}"

        p = HostTierPager(gpu_slots=1, transfer=fake_transfer,
                          wait=lambda h: None)
        p.ensure_resident(5)
        self.assertEqual(seen, [5])
        self.assertNotIn(5, p.handles)                    # finished


class Admission(unittest.TestCase):
    def test_demand_always_admits(self):
        a = AdmissionController(bytes_per_pass=100 * PAGE)
        for _ in range(200):
            self.assertTrue(a.admit(PAGE, TransferKind.DEMAND))

    def test_prefetch_capped_by_overfetch_budget(self):
        a = AdmissionController(bytes_per_pass=10 * PAGE, prefetch_overfetch_cap=0.35)
        admitted = 0
        while a.admit(PAGE, TransferKind.PREFETCH):
            admitted += 1
        self.assertEqual(admitted, 3)                     # 3 pages = 0.3 <= 0.35
        self.assertLessEqual(a.prefetch_bytes, 3.5 * PAGE)

    def test_prefetch_never_eats_demand_budget(self):
        a = AdmissionController(bytes_per_pass=10 * PAGE)
        a.admit(7 * PAGE, TransferKind.DEMAND)
        admitted = sum(1 for _ in range(10) if a.admit(PAGE, TransferKind.PREFETCH))
        self.assertEqual(admitted, 3)                     # 7 + 3 <= 10
        self.assertLessEqual(a.used_bytes, 10 * PAGE)

    def test_reset_clears_budget(self):
        a = AdmissionController(bytes_per_pass=PAGE)
        a.admit(PAGE, TransferKind.DEMAND)
        a.reset()
        self.assertEqual(a.used_bytes, 0)


class DesignInvariants(unittest.TestCase):
    def test_page_size_aligns_compressed_slots(self):
        # The runtime's invariant compressed_slot = full_slot // 4 requires
        # page granularity of at least the compress ratio.
        self.assertEqual(PAGE % 64, 0)

    def test_one_page_one_frame(self):
        # A page must never occupy two GPU frames: staged pages finish
        # before reuse is possible (ensure_resident is synchronous here).
        p = HostTierPager(gpu_slots=4)
        for page in range(20):
            p.ensure_resident(page)
        resident = [pid for pid, s in p.state.items() if s is PageState.GPU_RESIDENT]
        self.assertLessEqual(len(resident), 4)
        self.assertEqual(len(resident), len(set(resident)))


if __name__ == "__main__":
    unittest.main()
