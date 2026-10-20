"""G2 CPU tests: the multi-request tier against fake runtime objects.

16 simulated requests share one pool, table, and mirror; verifies
per-request accounting, retirement isolation, per-row restore, and the
row budget — with no GPU required.
"""

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "host_kv"))

KV_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]
PAGE = 64
ROWS, CTX, SLOTS = 20, 8192, 64 * 300


class FakePool:
    def __init__(self):
        self.full_attention_layer_id_mapping = {l: i for i, l in enumerate(KV_LAYERS)}
        self.k = [torch.randn(SLOTS, 2, 256, dtype=torch.bfloat16) for _ in KV_LAYERS]
        self.v = [torch.randn(SLOTS, 2, 256, dtype=torch.bfloat16) for _ in KV_LAYERS]

    def _t(self, layer):
        if layer not in self.full_attention_layer_id_mapping:
            raise ValueError(str(layer))
        return self.full_attention_layer_id_mapping[layer]

    def get_key_buffer(self, layer):
        return self.k[self._t(layer)]

    def get_value_buffer(self, layer):
        return self.v[self._t(layer)]


class FakeAllocator:
    def __init__(self, pages):
        self.free_pages = torch.arange(pages, dtype=torch.int32)


class FakeBackend:
    def __init__(self, pool, allocator, table):
        self.token_to_kv_pool = pool
        self.token_to_kv_pool_allocator = allocator
        self.req_to_token = table


class FakeBatch:
    def __init__(self, rows, out_cache_loc, seq_lens=None):
        self.req_pool_indices = torch.tensor(rows, dtype=torch.int64)
        self.out_cache_loc = out_cache_loc


def fresh(**env):
    log = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    log.close()
    os.environ.update({"SGLANG_HOSTTIER_ENABLE": "1",
                       "SGLANG_HOSTTIER_23M_LOG": log.name,
                       "SGLANG_HOSTTIER_ARM_AFTER": "1",
                       "SGLANG_HOSTTIER_SHRUNK": "",  # isolate from shrunk suite
                       "SGLANG_HOSTTIER_HOT_PAGES": "4",
                       **env})
    module = importlib.import_module("hosttier23m")
    importlib.reload(module)
    return module, log.name


def alloc_slots(allocator, n):
    pages = (n + PAGE - 1) // PAGE
    start = int(allocator.free_pages[0].item()) * PAGE
    allocator.free_pages = allocator.free_pages[pages:]
    return torch.arange(start, start + n, dtype=torch.int64)


class SixteenRequests(unittest.TestCase):
    def setUp(self):
        self.ht, self.log = fresh()
        self.pool = FakePool()
        self.allocator = FakeAllocator(SLOTS // PAGE)
        self.table = torch.zeros(ROWS, CTX, dtype=torch.int32)
        self.backend = FakeBackend(self.pool, self.allocator, self.table)
        # distinct slot ranges per row, like the real allocator
        self.next_slot = 0

    def _prefill_chunk(self, row, start, n_tokens):
        """Append-style chunk: only NEW context positions get slots,
        matching how the runtime extends a request."""
        locs = torch.arange(self.next_slot, self.next_slot + n_tokens,
                            dtype=torch.int64)
        self.next_slot += n_tokens
        self.table[row, start:start + n_tokens] = locs.to(torch.int32)
        self.ht.arm(self.backend, row)
        self.ht.after_write(FakeBatch([row], locs))

    def _prefill(self, row, n_tokens):
        self._prefill_chunk(row, 0, n_tokens)

    def test_sixteen_rows_accounted_independently(self):
        for row in range(16):
            self._prefill(row, 512)
        self.assertEqual(len(self.ht.R), 16)
        for row in range(16):
            st = self.ht.R[row]
            self.assertEqual(st["ctx_written"], 512)
            self.assertEqual(len(st["mirrored"]), 8)  # 512/64

    def test_retirement_isolated_per_row(self):
        for row in range(16):
            self._prefill(row, 1024)  # 16 pages; hot margin 4 -> retire 12
        retired_by_row = {row: len(st["retired"]) for row, st in self.ht.R.items()}
        self.assertTrue(all(n == 12 for n in retired_by_row.values()))
        # columns cleared only below the hot margin, in the owning row
        for row in range(16):
            self.assertTrue((self.table[row, :12 * PAGE] == 0).all(),
                            f"row {row} old columns not cleared")
            hot = self.table[row, 12 * PAGE:16 * PAGE]
            self.assertTrue((hot > 0).all(), f"row {row} hot columns cleared")
        # slots remembered per row are disjoint
        all_slots = torch.cat([s for st in self.ht.R.values()
                               for s in st["page_slots"].values()])
        self.assertEqual(len(torch.unique(all_slots)), len(all_slots))

    def test_restore_targets_only_requesting_row(self):
        self._prefill(0, 512)
        self._prefill(1, 512)
        # retire both rows' old pages
        retired0 = sorted(self.ht.R[0]["retired"])
        retired1 = sorted(self.ht.R[1]["retired"])
        self.assertTrue(retired0 and retired1)
        # row 0 gathers a retired page
        page = retired0[0]
        self.ht.before_gather(torch.tensor([page * PAGE + 3]),
                              FakeBatch([0], torch.tensor([0])))
        self.assertNotIn(page, self.ht.R[0]["retired"])
        self.assertIn(page, self.ht.R[0]["mirrored"])
        # row 1's retirement untouched
        self.assertIn(page, self.ht.R[1]["retired"]) if page in retired1 else None
        self.assertEqual(len(self.ht.R[1]["retired"]), len(retired1))
        # row 0's table columns restored to remembered slots
        slots = self.ht.R[0]["page_slots"][page]
        self.assertTrue((self.table[0, page * PAGE:(page + 1) * PAGE]
                         == slots.to(torch.int32)).all())

    def test_row_budget_respected(self):
        for row in range(20):  # MAX_REQUESTS default 16
            self._prefill(row, 128)
        self.assertEqual(len(self.ht.R), 16)
        # rows 16..19 ran untiered: ctx untouched is fine, no crash

    def test_layer_repeat_dedupe_global(self):
        # the same (first-slot, size) fired 12x must count once
        row = 0
        locs = torch.arange(0, 512, dtype=torch.int64)
        self.table[row, :512] = locs.to(torch.int32)
        self.ht.arm(self.backend, row)
        batch = FakeBatch([row], locs)
        for _ in range(12):
            self.ht.after_write(batch)
        self.assertEqual(self.ht.R[row]["ctx_written"], 512)

    def test_retire_time_remirror_captures_decode_writes(self):
        """The G2 verified=False cause: a page mirrored at prefill, then
        written again (decode/verify/chunk-completion) while still
        unretired, must re-mirror at retirement so the mirror is
        authoritative when restored."""
        # Re-arm the module against THIS test's fakes with a wide margin:
        # G is global, so reset it via reload after env is set.
        os.environ["SGLANG_HOSTTIER_HOT_PAGES"] = "20"
        import hosttier23m
        importlib.reload(hosttier23m)
        hosttier23m.G["armed"] = False  # rebind to this setUp's fakes
        self.ht = hosttier23m
        self.ht.arm(self.backend, row0 := 0)
        row = row0
        self._prefill_chunk(row, 0, 1024)  # 16 pages, none retire (hot=20)
        self.assertEqual(len(self.ht.R[row]["retired"]), 0)
        # writes land in the mirrored-but-unretired page 0
        old_page = 0
        slots = self.table[row, old_page * PAGE:(old_page + 1) * PAGE].long().clone()
        for layer_i in range(len(KV_LAYERS)):
            self.pool.k[layer_i][slots] = 768.0
            self.pool.v[layer_i][slots] = 768.0
        self._prefill_chunk(row, 1024, 1024)  # ctx 2048; hot=20 -> pages 0..11 retire
        self.assertIn(old_page, self.ht.R[row]["retired"])
        # mirror must now hold the decode-written 777 bytes
        for layer_i in range(len(KV_LAYERS)):
            mk, mv = self.ht.G["host_mirror"][KV_LAYERS[layer_i]]
            self.assertTrue((mk[slots].float() == 768.0).all(),
                            "mirror stale: prefill bytes retained")
        # restore path returns the decode-written bytes
        self.ht.before_gather(torch.tensor([old_page * PAGE + 1]),
                              FakeBatch([row], torch.tensor([0])))
        for layer_i in range(len(KV_LAYERS)):
            gk = self.pool.k[layer_i][slots].float()
            self.assertTrue((gk == 768.0).all(), "restore returned stale bytes")

    def test_stats_serialize(self):
        for row in range(3):
            self._prefill(row, 256)
        self.ht.dump_stats()
        import json
        payload = json.loads(Path(self.log).read_text())
        self.assertEqual(payload["requests"], 3)
        self.assertTrue(all("retired" in v for v in payload["per_request"].values()))


if __name__ == "__main__":
    unittest.main()
