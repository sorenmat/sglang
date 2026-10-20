"""Tests for the HKV-8 trace-detection rewrite (pure filesystem)."""

import gzip
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "serverless"))
os.environ.setdefault("PHASE0_WORKDIR", tempfile.mkdtemp())

import phase0_job as job  # noqa: E402


class WaitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        job.TRACES = self.tmp
        job._LOG_PREFIX = ""
        # silence log() prints during tests
        job.log = lambda msg: None

    def _write(self, name, mtime=None):
        path = self.tmp / name
        with gzip.open(path, "wt") as handle:
            json.dump({"traceEvents": []}, handle)
        if mtime:
            os.utime(path, (mtime, mtime))
        return path

    def test_suffixless_trace_captured(self):
        before = {self._write("old-TP-0.trace.json.gz")}
        target = self._write("1790-TP-0.trace.json.gz", mtime=time.time() + 5)
        got = job.wait_for_new_trace(before, "w1", timeout=3, poll=0.2)
        self.assertIsNotNone(got)
        self.assertEqual(got.name, "w1-TRACE.trace.json.gz")
        self.assertFalse(target.exists())  # renamed

    def test_decode_suffix_preferred(self):
        before = set()
        self._write("x-TP-0.trace.json.gz", mtime=time.time())
        self._write("x-TP-0-DECODE.trace.json.gz", mtime=time.time() + 1)
        got = job.wait_for_new_trace(before, "w2", timeout=3, poll=0.2)
        self.assertEqual(got.name, "w2-DECODE.trace.json.gz")

    def test_timeout_returns_none(self):
        before = {self._write("pre-existing.trace.json.gz")}
        got = job.wait_for_new_trace(before, "w3", timeout=1, poll=0.2)
        self.assertIsNone(got)


if __name__ == "__main__":
    unittest.main()
