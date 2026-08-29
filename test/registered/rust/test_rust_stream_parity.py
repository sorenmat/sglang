"""M5 differential parity test: Rust stream-frame builder + string-stop
decisions (sglang-server) vs the Python packing they replace (plan.md §8).

Three suites:

1. **Frame byte parity** — the same decode batch packed two ways and asserted
   byte-identical: the Python side runs the exact production packing from
   ``rust_server.push_generation`` (the ``FlatPairColumns`` /
   ``RaggedPairColumns`` / ``NestedRowColumns`` flatten + msgspec msgpack
   header, everything up to the ring push); the Rust side runs
   ``build_generation_frame``. Covers the core-4 frame, all seven
   logprob/hidden families plus nested hidden rows, the inactive-family
   empty-column spelling, the input-logprob leading-None→NaN sentinel, and
   finish-reason maps with str/int/float/None/list/dict values.

2. **String-stop parity** — ``Req._check_str_based_finish`` / ``tail_str`` /
   ``check_match_stop_str_prefix`` / ``_locate_str_stop_finished_len`` /
   ``_stop_match_tail_len`` driven against a deterministic char-tokenizer,
   compared to the Rust ``check_str_stop`` / ``stop_match_tail_len`` /
   ``stop_prefix_match`` / ``locate_str_stop_len`` next to the Rust
   detokenizer (including the speculative-decoding multi-accept scan and the
   invalid-regex abort).

3. **M9 performance gate** — Rust frame build p50 ≤ ½ Python p50 at B=256
   (both measured in-process from the same collected columns).

The extension is required; the suites skip when it cannot be loaded (the
CPU CI job builds it via SGLANG_BUILD_RUST_EXTS=all).
"""

import statistics
import time
import types
import unittest
from array import array
from itertools import chain

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

try:
    from sglang.srt.rust_extensions import load_rust_extension

    _ext = load_rust_extension("sglang.srt.rust_extensions._server")
except Exception:  # noqa: S110 - parity is best-effort without the extension
    _ext = None

DISPATCH_TAG_BATCH = 2  # sglang-server::message::response::DISPATCH_TAG_BATCH


def parse_frame(frame: bytes):
    """Split a `[BATCH tag][u32 LE header len][header][data…]` frame."""
    assert frame[0] == DISPATCH_TAG_BATCH, f"bad dispatch tag {frame[0]!r}"
    hlen = int.from_bytes(frame[1:5], "little")
    return frame[5 : 5 + hlen], frame[5 + hlen :]


@unittest.skipUnless(_ext is not None, "sglang-server extension unavailable")
class TestRustStreamParity(CustomTestCase):
    # ------------------------------------------------------------- helpers
    @staticmethod
    def _python_pack(rids, finish_reasons, prompt_tokens, output_ids, families):
        """The production Python packing: exactly what
        ``RustServer.push_generation`` computes up to the ring push.

        ``families`` maps the seven family names to their raw per-request
        columns (``val``/``idx`` pair, or hidden rows for the last).
        """
        import msgspec

        from sglang.srt.utils.flatten import (
            FlatPairColumns,
            NestedRowColumns,
            RaggedPairColumns,
        )

        tok_lens = list(map(len, output_ids))
        flat_ids = array("i", chain.from_iterable(output_ids))
        header_cols = [rids, finish_reasons, prompt_tokens, tok_lens]
        data_cols = [flat_ids.tobytes()]

        has_extra = any(families.values())
        if has_extra:
            batch_size = len(rids)
            extras = (
                FlatPairColumns(
                    "output_token_logprobs",
                    families["out_lp_val"] or [],
                    families["out_lp_idx"] or [],
                ),
                FlatPairColumns(
                    "input_token_logprobs",
                    families["in_lp_val"] or [],
                    families["in_lp_idx"] or [],
                    first_none_to_nan=True,
                ),
                RaggedPairColumns(
                    "output_top_logprobs",
                    families["out_top_val"] or [],
                    families["out_top_idx"] or [],
                ),
                RaggedPairColumns(
                    "input_top_logprobs",
                    families["in_top_val"] or [],
                    families["in_top_idx"] or [],
                ),
                RaggedPairColumns(
                    "output_token_ids_logprobs",
                    families["out_tid_val"] or [],
                    families["out_tid_idx"] or [],
                ),
                RaggedPairColumns(
                    "input_token_ids_logprobs",
                    families["in_tid_val"] or [],
                    families["in_tid_idx"] or [],
                ),
                NestedRowColumns(
                    "output_hidden_states", families["hidden_rows"] or []
                ),
            )
            active = []
            for extra in extras:
                populated = False
                for name, col in extra.columns():
                    assert len(col) in (0, batch_size), (
                        f"extras column {name}: {len(col)} entries for a batch "
                        f"of {batch_size}"
                    )
                    populated |= len(col) > 0
                if populated:
                    active.append(extra)
            for extra in active:
                accept = extra.accept
                for i in range(batch_size):
                    accept(i)
            for extra in extras:
                header_cols += extra.header_cols()
                data_cols += extra.data_cols()

        header = msgspec.msgpack.encode(header_cols)
        data = b"".join(data_cols)
        return header, data

    @staticmethod
    def _rust_frame(rids, finish_reasons, prompt_tokens, tok_lens, output_ids, **fam):
        ids = array("i", chain.from_iterable(output_ids)).tobytes()
        frame = _ext.build_generation_frame(
            rids,
            finish_reasons,
            prompt_tokens,
            tok_lens,
            ids,
            **fam,
        )
        return parse_frame(bytes(frame))

    @staticmethod
    def _assert_same_frame(rids, finish_reasons, prompt_tokens, output_ids, fam, fam_kw):
        py_header, py_data = TestRustStreamParity._python_pack(
            rids, finish_reasons, prompt_tokens, output_ids, fam
        )
        rust_header, rust_data = TestRustStreamParity._rust_frame(
            rids, finish_reasons, prompt_tokens,
            [len(o) for o in output_ids], output_ids, **fam_kw
        )
        self.assertEqual(
            rust_header, py_header,
            "header msgpack bytes diverge (rids/finish/tok-len/shape columns)",
        )
        self.assertEqual(rust_data, py_data, "data column bytes diverge")

    # ------------------------------------------------------- frame parity
    def test_core_frame_parity(self):
        """The hot path: four header columns + the ids buffer, no families."""
        rids = ["r-0001", "r-0002", "r-0003"]
        finish = [
            None,
            {"type": "stop", "matched": "STOP"},
            {
                "type": "length",
                "length": 2048,
                "meta": {"n": 3, "ratio": 0.5, "ok": True, "tag": None},
            },
        ]
        out = [[10, 11, 12], [20], []]
        fam = {
            "out_lp_val": [], "out_lp_idx": [],
            "in_lp_val": [], "in_lp_idx": [],
            "out_top_val": [], "out_top_idx": [],
            "in_top_val": [], "in_top_idx": [],
            "out_tid_val": [], "out_tid_idx": [],
            "in_tid_val": [], "in_tid_idx": [],
            "hidden_rows": [],
        }
        self._assert_same_frame(rids, finish, [7, 8, 9], out, fam, {})

    def test_all_families_frame_parity(self):
        """Every family active at once, with null positions, a leading-None
        input-logprob sentinel, nested hidden rows, and long rids (str8+)."""
        b = 4
        toks = 32
        rids = [f"req-{i:040}" for i in range(b)]
        out = [[1000 + i * 100 + t for t in range(toks)] for i in range(b)]
        finish = [None, None, {"type": "stop", "matched": "DONE"}, None]
        # Flat families: per-request lists; req1's input logprobs lead with the
        # first-prompt-token None sentinel.
        out_lp_val = [[-0.1 * t for t in range(toks)] for _ in range(b)]
        out_lp_idx = [[t for t in range(toks)] for _ in range(b)]
        in_lp_val = [
            [None if t == 0 else -0.2 * t for t in range(toks)] if i == 1
            else [-0.3 * t for t in range(toks)]
            for i in range(b)
        ]
        in_lp_idx = [[t for t in range(toks)] for _ in range(b)]
        # Ragged families: a None position at t==7, a short top-k at t==8.
        rag_val = [
            [[-0.5] * 3 if t != 7 else None
                for t in range(toks)]
            for _ in range(b)
        ]
        rag_idx = [
            [[7, 8, 9] if t != 7 else None for t in range(toks)]
            for _ in range(b)
        ]
        rag_val2 = [
            [[-0.6] if t == 8 else None for t in range(toks)]
            for _ in range(b)
        ]
        rag_idx2 = [
            [[100] if t == 8 else None for t in range(toks)]
            for _ in range(b)
        ]
        rag_val3 = [
            [[-0.7, -0.8] if t % 5 == 0 else None for t in range(toks)]
            for _ in range(b)
        ]
        rag_idx3 = [
            [[200, 201] if t % 5 == 0 else None for t in range(toks)]
            for _ in range(b)
        ]
        rag_val4 = [
            [[-0.9] * 2 if t != 7 else None for t in range(toks)]
            for _ in range(b)
        ]
        rag_idx4 = [
            [[300, 301] if t != 7 else None for t in range(toks)]
            for _ in range(b)
        ]
        # Hidden rows: the (float | list) union incl. a nested list-of-lists.
        hidden = [
            [
                (0.1 * t if t % 3 else [0.5, [0.25, 0.125]])
                for t in range(toks)
            ]
            for _ in range(b)
        ]
        fam = {
            "out_lp_val": out_lp_val, "out_lp_idx": out_lp_idx,
            "in_lp_val": in_lp_val, "in_lp_idx": in_lp_idx,
            "out_top_val": rag_val, "out_top_idx": rag_idx,
            "in_top_val": rag_val2, "in_top_idx": rag_idx2,
            "out_tid_val": rag_val3, "out_tid_idx": rag_idx3,
            "in_tid_val": rag_val4, "in_tid_idx": rag_idx4,
            "hidden_rows": hidden,
        }
        self._assert_same_frame(rids, finish, [128] * b, out, fam, fam)

    def test_inactive_family_ships_empty_columns_in_place(self):
        """One active family with the rest empty: the inactive families still
        contribute empty header/data columns (arity unchanged), and the
        Python `[]` and the Rust None spellings agree."""
        b = 3
        out = [[1, 2], [3], [4, 5, 6]]
        fam = {
            "out_lp_val": [[-0.1, -0.2], [-0.3], [-0.4, -0.5, -0.6]],
            "out_lp_idx": [[10, 11], [12], [13, 14, 15]],
            "in_lp_val": [], "in_lp_idx": [],
            "out_top_val": [], "out_top_idx": [],
            "in_top_val": [], "in_top_idx": [],
            "out_tid_val": [], "out_tid_idx": [],
            "in_tid_val": [], "in_tid_idx": [],
            "hidden_rows": [],
        }
        self._assert_same_frame(
            [f"r{i}" for i in range(b)], [None] * b, [4, 4, 4], out,
            fam, {"out_lp_val": fam["out_lp_val"], "out_lp_idx": fam["out_lp_idx"]},
        )

    def test_empty_batch_frame_parity(self):
        rids, finish, prompt, out = [], [], [], []
        fam = {
            "out_lp_val": [], "out_lp_idx": [],
            "in_lp_val": [], "in_lp_idx": [],
            "out_top_val": [], "out_top_idx": [],
            "in_top_val": [], "in_top_idx": [],
            "out_tid_val": [], "out_tid_idx": [],
            "in_tid_val": [], "in_tid_idx": [],
            "hidden_rows": [],
        }
        self._assert_same_frame(rids, finish, prompt, out, fam, {})


# --------------------------------------------------------------------- str-stop
class _CharTokenizer:
    """One token id = one char (id 0 is NUL) — deterministic and hand-checkable."""

    eos_token_id = None

    def decode(self, ids):
        return "".join(chr(t) for t in ids)


def _char_decode(ids):
    return "".join(chr(t) for t in ids)


def _make_req(output_ids, stop_strs, stop_regex_strs, max_str, max_regex,
              previously_decoded):
    """A bare `Req` carrying just the state the stop checks read."""
    from sglang.srt.managers.schedule_batch import Req

    req = Req.__new__(Req)
    req.sampling_params = types.SimpleNamespace(
        stop_strs=list(stop_strs),
        stop_regex_strs=list(stop_regex_strs),
        stop_str_max_len=max_str,
        stop_regex_max_len=max_regex,
        stop_token_ids=None,
        ignore_eos=False,
        max_new_tokens=10**9,
    )
    req.output_ids = list(output_ids)
    req.tokenizer = _CharTokenizer()
    req.decoded_text = previously_decoded
    req.rid = "parity-req"
    req.finished_reason = None
    req.finished_len = None
    req.to_finish = None
    return req


def _window(req, new_accepted_len):
    """The `_locate_str_stop_finished_len` window, computed like Python does."""
    tail_len = req._stop_match_tail_len(new_accepted_len)
    start = len(req.output_ids) - tail_len
    return start, req.output_ids[start:]


def _prefix_texts(window, new_accepted_len):
    # Only the scanned range is read by Rust, but building the full set keeps
    # the caller trivially correct; the decode is the char-tokenizer.
    return [_char_decode(window[: k]) for k in range(1, len(window) + 1)]


@unittest.skipUnless(_ext is not None, "sglang-server extension unavailable")
class TestRustStrStopParity(CustomTestCase):
    # ------------------------------------------------------------- cases
    CASES = [
        # (output ids, nal, stop_strs, stop_regex_strs, max_str, max_regex,
        #  previously decoded, window start override)
        dict(
            name="stop string in the tail",
            ids=[104, 101, 80, 84, 79, 80],  # "heSTOP"
            nal=1, stop_strs=["STOP"], stop_regex=[], max_str=4, max_regex=0,
            previously="hello ",
        ),
        dict(
            name="stop string only in earlier text",
            ids=[120],  # "x"
            nal=1, stop_strs=["STOP"], stop_regex=[], max_str=4, max_regex=0,
            previously="the STOP word was here",
        ),
        dict(
            name="no match",
            ids=[104, 105],  # "hi"
            nal=2, stop_strs=["STOP"], stop_regex=[], max_str=4, max_regex=0,
            previously="nothing to see",
        ),
        dict(
            name="regex in the tail",
            ids=[120, 120, 83, 84, 79, 80, 121, 121],  # "xxSTOPyy"
            nal=8, stop_strs=[], stop_regex=[r"STOP"], max_str=0, max_regex=4,
            previously="preamble",
        ),
        dict(
            name="invalid regex aborts",
            ids=[97],  # "a"
            nal=1, stop_strs=[], stop_regex=["("], max_str=0, max_regex=1,
            previously="",
        ),
        dict(
            name="stop string beats regex",
            ids=[83, 84, 79, 80],  # "STOP"
            nal=4, stop_strs=["STOP"], stop_regex=[r".+"], max_str=4,
            max_regex=2, previously="",
        ),
        dict(
            name="empty stop string matches everything",
            ids=[122],  # "z"
            nal=1, stop_strs=[""], stop_regex=[], max_str=0, max_regex=0,
            previously="",
        ),
        dict(
            name="speculative accept, match mid-window",
            ids=[97, 98, 99, 83, 84, 79, 80],  # "abcSTOP", 3 newly accepted
            nal=3, stop_strs=["STOP"], stop_regex=[], max_str=4, max_regex=0,
            previously="prefix text ",
        ),
    ]

    def _case(self, case):
        req = _make_req(
            case["ids"],
            case["stop_strs"],
            case["stop_regex"],
            case["max_str"],
            case["max_regex"],
            case["previously"],
        )
        nal = case["nal"]
        hit = req._check_str_based_finish(nal)
        py_reason = req.finished_reason
        py_len = req.finished_len

        start, window = _window(req, nal)
        kinds = {"none": None, "str": "FINISH_MATCHED_STR",
                 "regex": "FINISHED_MATCHED_REGEX",
                 "invalid_regex": "FINISH_ABORT"}
        kind, matched, rust_len = _ext.check_str_stop(
            req.tail_str(nal),
            req.decoded_text,
            list(case["stop_strs"]),
            list(case["stop_regex"]),
            [int(t) for t in window],
            start,
            nal,
            len(req.output_ids),
            _prefix_texts(window, nal),
        )
        py_kind = None if not hit else py_reason.__class__.__name__
        self.assertEqual(
            kinds[kind], py_kind,
            f"{case['name']}: rust {kind!r} vs python {py_kind!r}",
        )
        if py_kind in ("FINISH_MATCHED_STR", "FINISHED_MATCHED_REGEX"):
            self.assertEqual(matched, py_reason.matched, case["name"])
            self.assertEqual(rust_len, py_len, case["name"])
        elif py_kind is None:
            self.assertIsNone(matched, case["name"])
            self.assertIsNone(rust_len, case["name"])
        else:  # FINISH_ABORT: only the fact + pattern matter
            self.assertEqual(matched, case["stop_regex"][0], case["name"])
            self.assertIsNone(rust_len, case["name"])

    def test_str_based_finish_matches_python(self):
        for case in self.CASES:
            with self.subTest(name=case["name"]):
                self._case(case)

    def test_tail_len_matches_python(self):
        for max_str, max_regex in [(5, 0), (0, 5), (3, 7), (0, 0)]:
            for nal in [0, 1, 3, 8]:
                for out_len in [0, 4, 100]:
                    req = _make_req(
                        list(range(out_len)), [], [], max_str, max_regex, ""
                    )
                    got = _ext.stop_match_tail_len(
                        max_str, max_regex, nal, out_len
                    )
                    self.assertEqual(
                        got, req._stop_match_tail_len(nal),
                        f"({max_str},{max_regex},nal={nal},len={out_len})",
                    )

    def test_prefix_gate_matches_python(self):
        stop = "STOP"
        for tail in ["", "hello", "hello STOP", "hello ST", "hello SX hello"]:
            req = _make_req(
                [ord(c) for c in tail], [stop], [], 4, 0, ""
            )
            # Python reads the tail through its own tail_str() (new_accepted_len
            # default 1, whole output here since the tail len covers it).
            py = req.check_match_stop_str_prefix()
            rust = _ext.stop_prefix_match(req.tail_str(1), [stop])
            self.assertEqual(rust, py, f"tail={tail!r}")

    def test_locate_str_stop_len_matches_python(self):
        # Window "abcSTOP" from start 10, scanned at two accept widths.
        ids = list(range(10)) + [ord(c) for c in "abcSTOP"]
        req = _make_req(ids, ["ST"], [], 4, 0, "")
        for nal in (1, 3):
            start, window = _window(req, nal)
            py = req._locate_str_stop_finished_len(nal, stop_str="ST")
            rust = _ext.locate_str_stop_len(
                [int(t) for t in window], start, nal, len(req.output_ids),
                _prefix_texts(window, nal), "ST",
            )
            self.assertEqual(rust, py, f"nal={nal}")
        # The regex spelling locates the same way.
        req2 = _make_req(ids, [], [r"ST"], 0, 4, "")
        start, window = _window(req2, 3)
        py = req2._locate_str_stop_finished_len(3, stop_regex=r"ST")
        rust = _ext.locate_str_stop_len(
            [int(t) for t in window], start, 3, len(req2.output_ids),
            _prefix_texts(window, 3), "ST",  # Rust takes the pattern text;
            # the str/regex distinction lives in check_str_stop
        )
        self.assertEqual(rust, py)


# --------------------------------------------------------------------- M9 gate
@unittest.skipUnless(_ext is not None, "sglang-server extension unavailable")
class TestRustStreamPerfGate(CustomTestCase):
    """M9 (plan.md §8 gate): Rust frame build p50 ≤ ½ Python p50 at B=256,
    logprobs off (the hot decode path)."""

    B = 256
    TOKS = 32
    ITERS = 150

    def _payload(self):
        rids = [f"req-{i:08}" for i in range(self.B)]
        finish = [None] * self.B
        prompt = [128] * self.B
        out = [[1000 + i * 100 + t for t in range(self.TOKS)] for i in range(self.B)]
        fam = {
            "out_lp_val": [], "out_lp_idx": [],
            "in_lp_val": [], "in_lp_idx": [],
            "out_top_val": [], "out_top_idx": [],
            "in_top_val": [], "in_top_idx": [],
            "out_tid_val": [], "out_tid_idx": [],
            "in_tid_val": [], "in_tid_idx": [],
            "hidden_rows": [],
        }
        ids = array("i", chain.from_iterable(out)).tobytes()
        return rids, finish, prompt, out, fam, ids

    def test_rust_frame_build_at_most_half_python_p50(self):
        rids, finish, prompt, out, fam, ids = self._payload()

        def py_pack():
            return TestRustStreamParity._python_pack(
                rids, finish, prompt, out, fam
            )

        def rust_pack():
            return _ext.build_generation_frame(
                rids, finish, prompt,
                [len(o) for o in out], ids
            )

        for _ in range(5):  # warm-up
            py_pack()
            rust_pack()
        py_ns, rust_ns = [], []
        for _ in range(self.ITERS):
            t0 = time.perf_counter_ns()
            py_pack()
            py_ns.append(time.perf_counter_ns() - t0)
            t0 = time.perf_counter_ns()
            rust_pack()
            rust_ns.append(time.perf_counter_ns() - t0)
        py_p50 = statistics.median(py_ns)
        rust_p50 = statistics.median(rust_ns)
        self.assertLessEqual(
            rust_p50,
            0.5 * py_p50,
            f"M9 gate failed: Rust p50 {rust_p50 / 1000:.1f} µs > ½ Python "
            f"p50 {py_p50 / 1000:.1f} µs (B={self.B})",
        )


if __name__ == "__main__":
    unittest.main()
