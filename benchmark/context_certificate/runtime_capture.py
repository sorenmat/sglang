"""Opt-in bounded QSA diagnostics; standalone, with no serving-runtime imports.

Disabled wrappers inspect only environment variables/rank, never tensors or files.
Enabled capture synchronizes the GPU and perturbs timing. NPZ files preserve
native logits and selections; they do not establish quality or kernel speed.
"""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import tempfile
import threading
import uuid

import numpy as np


PREFIX = "SGLANG_QSA_CAPTURE_"
_SESSION = None
_SESSION_CONFIG = None


def _text(value, name, limit=4096):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must be a nonempty string of at most {limit} characters")
    return value


def _request_ids(values, rows):
    if not isinstance(values, (list, tuple)) or len(values) != rows:
        raise ValueError("request IDs must be a Python list/tuple aligned with tensor rows")
    return [_text(value, "request_id", 1024) for value in values]


def _graph_guard(tensor):
    if tensor.is_cuda:
        import torch
        with torch.cuda.device(tensor.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("QSA capture requires CUDA graphs disabled; cannot capture inside a CUDA graph")


def _integers(tensor, shape, name):
    import torch
    if tensor.shape != shape or tensor.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be an int32/int64 tensor of shape {shape}")
    return tensor.detach().to(device="cpu", dtype=torch.int64).numpy()


def _bf16_export(tensor):
    import torch
    return tensor.detach().to(dtype=torch.bfloat16).to(device="cpu", dtype=torch.float32).numpy()


class CaptureSession:
    def __init__(self, directory, *, model_id, model_revision, runtime_revision,
                 max_files=8, max_bytes=256 * 1024**2, max_keys=65536,
                 max_events=32, layers=None):
        self.directory = Path(directory)
        self.identity = {name: _text(value, name) for name, value in (
            ("model_id", model_id), ("model_revision", model_revision),
            ("runtime_revision", runtime_revision))}
        for name, value in (("max_files", max_files), ("max_bytes", max_bytes),
                            ("max_keys", max_keys), ("max_events", max_events)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
            setattr(self, name, value)
        self.layers = None if layers is None else frozenset(layers)
        if self.layers is not None and any(type(layer) is not int or layer < 0 for layer in self.layers):
            raise ValueError("layers must contain nonnegative integer IDs")
        self._lock = threading.RLock()

    @contextmanager
    def _locked(self):
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.directory / ".capture.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                os.close(fd)

    def _usage(self):
        files = [path for path in self.directory.iterdir()
                 if path.name != ".capture.lock" and path.is_file()]
        return sum(path.suffix == ".npz" for path in files), sum(path.stat().st_size for path in files)

    def _write_npz(self, arrays, reservation):
        destination = self.directory / f"qsa-{uuid.uuid4().hex}.npz"
        fd, temporary = tempfile.mkstemp(prefix=".qsa-", suffix=".tmp", dir=self.directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez(stream, **arrays)  # Uncompressed: payload size is predictable.
                stream.flush()
                os.fsync(stream.fileno())
            if os.stat(temporary).st_size > reservation:
                raise RuntimeError("NPZ exceeded its pre-copy byte reservation")
            os.link(temporary, destination)  # Atomic publication; never overwrite.
        finally:
            os.unlink(temporary)
        return str(destination)

    def capture_qsa(self, *, q, compressed_cache, page_table, lengths,
                    query_positions, native_scores, native_selected_ids,
                    row_request_ids, layer_id, forward_mode, rank=0):
        if rank != 0:
            return {"status": "nonzero_rank", "files": []}
        if forward_mode not in ("decode", "target_verify"):
            return {"status": "unsupported_forward_mode", "files": []}
        if type(layer_id) is not int or layer_id < 0:
            raise ValueError("layer_id must be a nonnegative integer")
        if self.layers is not None and layer_id not in self.layers:
            return {"status": "filtered_layer", "files": []}
        _graph_guard(q)  # Must precede all synchronization/control-tensor copies.
        import torch
        rows = q.shape[0] if q.ndim else 0
        if (q.ndim != 3 or q.shape[1] not in (4, 8) or q.shape[2] != 128 or rows == 0
                or not q.is_floating_point()):
            raise ValueError("q must be floating [rows,4 or 8,128]")
        if (compressed_cache.ndim != 4 or compressed_cache.shape[1] not in (8, 16, 32, 64)
                or compressed_cache.shape[2:] != (1, 128)
                or not compressed_cache.is_floating_point()):
            raise ValueError("compressed_cache must be floating [pages,page_size,1,128]")
        if q.dtype != torch.bfloat16 or compressed_cache.dtype != torch.bfloat16:
            raise ValueError("initial captures require actual BF16 query/cache tensors; kernel-dependent casts cannot be inferred")
        tensors = (compressed_cache, page_table, lengths, query_positions, native_scores, native_selected_ids)
        if any(tensor.device != q.device for tensor in tensors):
            raise ValueError("all native tensors must share q.device")
        if (page_table.ndim != 2 or page_table.shape[0] != rows
                or page_table.dtype not in (torch.int32, torch.int64)):
            raise ValueError("page_table must be integer [rows,max_pages]")
        for name, tensor in (("lengths", lengths), ("query_positions", query_positions)):
            if tensor.shape != (rows,) or tensor.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"{name} must be an integer [rows] tensor")
        if native_scores.ndim != 2 or native_scores.shape[0] != rows or native_scores.dtype != torch.float32:
            raise ValueError("native_scores must be native FP32 [rows,max_keys]")
        if (native_selected_ids.shape != (rows, 512)
                or native_selected_ids.dtype not in (torch.int32, torch.int64)):
            raise ValueError("native_selected_ids must be integer [rows,512]")
        ids = _request_ids(row_request_ids, rows)
        groups, seen = [], set()
        for row, request_id in enumerate(ids):
            if not groups or request_id != groups[-1][0]:
                if request_id in seen:
                    raise ValueError("rows for each request must be consecutive")
                seen.add(request_id)
                groups.append([request_id, row, row + 1])
            else:
                groups[-1][2] = row + 1
        verification_id = uuid.uuid4().hex
        written, skipped = [], {}
        with self._locked():
            for request_id, start, end in groups:
                count, used_bytes = self._usage()
                reason = "file_budget" if count >= self.max_files else None
                # Even the smallest capture needs this many bytes. Do not copy
                # control tensors when no possible capture can fit.
                if reason is None and used_bytes + 512 * 128 * 4 + 65536 > self.max_bytes:
                    reason = "byte_budget"
                if reason:
                    skipped[reason] = skipped.get(reason, 0) + 1
                    continue
                size = end - start
                if size > 64 or (forward_mode == "decode" and size != 1):
                    raise ValueError("capture permits one decode row or at most 64 verification rows per request")
                valid = _integers(lengths[start:end], (size,), "lengths")
                positions = _integers(query_positions[start:end], (size,), "query_positions")
                if (np.any(valid < 0) or np.any(valid[1:] < valid[:-1])
                        or np.any(positions < 0) or np.any(positions >= np.iinfo(np.int64).max)
                        or np.any(positions[1:] != positions[:-1] + 1)
                        or np.any((positions + 1) // 4 != valid)):
                    raise ValueError("causal lengths must equal (consecutive query_positions+1)//4")
                if np.any(valid < 512):
                    skipped["short_context"] = skipped.get("short_context", 0) + 1
                    continue
                n = int(valid[-1])
                if n > self.max_keys:
                    skipped["key_budget"] = skipped.get("key_budget", 0) + 1
                    continue
                page_size = compressed_cache.shape[1]
                pages = (n + page_size - 1) // page_size
                if pages > page_table.shape[1] or n > native_scores.shape[1]:
                    raise ValueError("native page table or logits do not cover visible keys")
                metadata_bytes = 4 * len(json.dumps(dict(self.identity, request_id=request_id)))
                reservation = n * 128 * 4 + size * (4 * 128 * 4 + n * 4 + 512 * 8 + 16) + metadata_bytes + 65536
                if used_bytes + reservation > self.max_bytes:
                    skipped["byte_budget"] = skipped.get("byte_budget", 0) + 1
                    continue
                # Large GPU-to-CPU transfers begin only after this reservation.
                mappings = _integers(page_table[start:end, :pages], (size, pages), "page_table")
                last_mapping = mappings[-1]
                if np.any(last_mapping < 0) or np.any(last_mapping >= compressed_cache.shape[0]):
                    raise ValueError("page table contains out-of-range physical pages")
                for row, visible in enumerate(valid):
                    prefix_pages = (int(visible) + page_size - 1) // page_size
                    if not np.array_equal(mappings[row, :prefix_pages], last_mapping[:prefix_pages]):
                        raise ValueError("visible prefixes use different physical pages; shared key bank not verified")
                if q.shape[1] == 8 and torch.count_nonzero(q[start:end, 4:]).item() != 0:
                    raise ValueError("extra query heads must be exactly zero")
                physical = torch.as_tensor(last_mapping.copy(), device=compressed_cache.device, dtype=torch.long)
                visible_cache = compressed_cache.index_select(0, physical)[:, :, 0, :].reshape(-1, 128)[:n]
                keys = _bf16_export(visible_cache)
                queries = _bf16_export(q[start:end, :4])
                logits = native_scores[start:end, :n].detach().cpu().numpy()
                selected = _integers(native_selected_ids[start:end], (size, 512), "native_selected_ids")
                if not np.isfinite(keys).all() or not np.isfinite(queries).all():
                    raise ValueError("native BF16 scorer inputs must be finite")
                for row, visible in enumerate(valid):
                    if (not np.isfinite(logits[row, :visible]).all()
                            or np.any(selected[row] < 0) or np.any(selected[row] >= visible)
                            or len(np.unique(selected[row])) != 512):
                        raise ValueError("invalid native logits or selected logical key IDs")
                    threshold = np.partition(logits[row, :visible], -512)[-512]
                    if np.any(logits[row, selected[row]] < threshold):
                        raise ValueError("native selected IDs disagree with native logits")
                metadata = {
                    "schema_version": 1, "source_kind": "model_capture", **self.identity,
                    "forward_mode": forward_mode, "compress_ratio": 4,
                    "score_semantics": "sum_relu_dot_div_sqrt_dim", "top_k": 512,
                    "request_id": request_id, "verification_id": verification_id, "layer_id": layer_id,
                    "request_identity_kind": "caller_supplied_row_request_id",
                    "key_dtype": "bfloat16", "query_dtype": "bfloat16",
                    "keys_stage": "finalized_compressed_index_keys", "queries_stage": "native_index_score_input",
                    "shared_key_bank_verified": True, "shared_key_bank_evidence": "identical visible physical page mappings into one cache snapshot",
                    "input_q_dtype": str(q.dtype), "input_cache_dtype": str(compressed_cache.dtype),
                    "input_query_heads": q.shape[1], "page_size": page_size,
                    "cast_semantics": "actual BF16 query/cache tensors exported losslessly as FP32; no inference of kernel-dependent FP32 casts",
                    "capture_perturbs_timing": True,
                    "event_join": "verification_id identifies this hook call; acceptance events are not linked exactly",
                }
                arrays = dict(metadata=json.dumps(metadata), keys=keys, queries=queries,
                              valid_counts=valid, query_positions=positions,
                              native_scores=logits, native_selected_ids=selected)
                written.append(self._write_npz(arrays, reservation))
        return {"status": "captured" if written else "skipped", "files": written, "skipped": skipped}

    def capture_acceptance(self, *, request_ids, accepted_drafts, draft_positions, rank=0,
                           per_request_metadata=None):
        if rank != 0:
            return {"status": "nonzero_rank"}
        rows = len(request_ids)
        ids = _request_ids(request_ids, rows)
        if not 1 <= rows <= 4096 or len(set(ids)) != rows:
            raise ValueError("acceptance requires 1..4096 distinct request IDs")
        metadata = None
        if per_request_metadata is not None:
            fields = {"forward_iter", "req_pool_idx", "verification_count", "retraction_count",
                      "retracted", "finished", "finished_before_output_extension", "grammar_retained_count"}
            if not isinstance(per_request_metadata, (list, tuple)) or len(per_request_metadata) != rows:
                raise ValueError("per_request_metadata must align with request IDs")
            metadata = []
            for item in per_request_metadata:
                if (not isinstance(item, dict) or item.keys() - fields
                        or any(type(value) not in (int, float, bool, str, type(None)) for value in item.values())):
                    raise ValueError("per-request metadata supports known primitive CPU fields only")
                if len(json.dumps(item, allow_nan=False)) > 4096:
                    raise ValueError("per-request metadata exceeds 4096 characters")
                metadata.append(dict(item))
        if hasattr(accepted_drafts, "is_cuda"):
            _graph_guard(accepted_drafts)
        if hasattr(draft_positions, "is_cuda"):
            _graph_guard(draft_positions)
        with self._locked():
            _, used_bytes = self._usage()
            target = self.directory / "acceptance.jsonl"
            if target.is_symlink():
                raise ValueError("acceptance output must not be a symlink")
            if self.max_events == 0:
                return {"status": "event_budget"}
            reservation = len(json.dumps(ids).encode()) + len(json.dumps(self.identity).encode()) + rows * 128 + 65536
            if metadata is not None:
                reservation += len(json.dumps(metadata).encode())
            if used_bytes + reservation > self.max_bytes:
                return {"status": "byte_budget"}
            events = 0
            if target.exists():
                with target.open("rb") as stream:
                    for _ in stream:
                        events += 1
                        if events >= self.max_events:
                            break
            if events >= self.max_events:
                return {"status": "event_budget"}
            if hasattr(accepted_drafts, "detach"):
                accepted = _integers(accepted_drafts, (rows,), "accepted_drafts").tolist()
            else:
                accepted = list(accepted_drafts)
            if type(draft_positions) is int:
                positions = [draft_positions] * rows
            elif hasattr(draft_positions, "detach"):
                positions = _integers(draft_positions, (rows,), "draft_positions").tolist()
            else:
                positions = list(draft_positions)
            if (len(accepted) != rows or len(positions) != rows
                    or any(type(a) is not int or type(p) is not int or not 0 <= a < p <= 4096
                           for a, p in zip(accepted, positions))):
                raise ValueError("require integer 0 <= accepted_drafts < draft_positions <= 4096 per request")
            event = {
                "schema_version": 1, "source_kind": "model_capture", **self.identity,
                "event_type": "speculative_acceptance", "event_id": uuid.uuid4().hex,
                "request_ids": ids, "accepted_drafts": accepted, "draft_positions": positions,
                "accepted_output_slots_including_bonus": [value + 1 for value in accepted],
                "count_semantics": "Native accepted output slots including correction/bonus; not actual emitted tokens after EOS, grammar or cancellation",
                "event_join": "Request IDs permit association; no exact join to QSA verification_id is recorded",
            }
            if metadata is not None:
                event["per_request_metadata"] = metadata
            payload = (json.dumps(event, separators=(",", ":")) + "\n").encode()
            if len(payload) > reservation:
                raise RuntimeError("acceptance event exceeded its pre-copy byte reservation")
            fd = os.open(target, os.O_CREAT | os.O_APPEND | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "ab") as stream:
                stream.write(payload)
            return {"status": "captured", "path": str(target), "request_count": rows}


def _session():
    global _SESSION, _SESSION_CONFIG
    directory = os.environ.get(PREFIX + "DIR")
    if not directory:
        return None
    if float(os.environ.get("SGLANG_SIMULATE_ACC_LEN", "0")) > 0:
        raise ValueError("disable SGLANG_SIMULATE_ACC_LEN before collecting real captures")
    names = ("MODEL_ID", "MODEL_REVISION", "RUNTIME_REVISION", "MAX_FILES", "MAX_BYTES", "MAX_KEYS", "MAX_EVENTS", "LAYERS")
    config = (directory, *(os.environ.get(PREFIX + name) for name in names))
    if config != _SESSION_CONFIG:
        identity = {name.lower(): os.environ.get(PREFIX + name) for name in names[:3]}
        limits = {name.lower(): int(os.environ.get(PREFIX + name, default)) for name, default in (
            ("MAX_FILES", "8"), ("MAX_BYTES", str(256 * 1024**2)), ("MAX_KEYS", "65536"), ("MAX_EVENTS", "32"))}
        layers = os.environ.get(PREFIX + "LAYERS")
        _SESSION = CaptureSession(directory, **identity, **limits,
                                  layers=None if layers is None else [int(value) for value in layers.split(",")])
        _SESSION_CONFIG = config
    return _SESSION


def capture_qsa(*, q, compressed_cache, page_table, lengths, query_positions,
                native_scores, native_selected_ids, row_request_ids, layer_id,
                forward_mode, rank=0):
    if rank != 0:
        return {"status": "nonzero_rank", "files": []}
    session = _session()
    if session is None:
        return {"status": "disabled", "files": []}
    return session.capture_qsa(q=q, compressed_cache=compressed_cache, page_table=page_table,
                               lengths=lengths, query_positions=query_positions, native_scores=native_scores,
                               native_selected_ids=native_selected_ids, row_request_ids=row_request_ids,
                               layer_id=layer_id, forward_mode=forward_mode, rank=rank)


def capture_acceptance(*, request_ids, accepted_drafts, draft_positions, rank=0,
                       per_request_metadata=None):
    if rank != 0:
        return {"status": "nonzero_rank"}
    session = _session()
    if session is None:
        return {"status": "disabled"}
    return session.capture_acceptance(request_ids=request_ids, accepted_drafts=accepted_drafts,
                                      draft_positions=draft_positions, rank=rank,
                                      per_request_metadata=per_request_metadata)
