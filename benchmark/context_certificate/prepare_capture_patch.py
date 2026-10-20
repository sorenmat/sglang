"""Build a source-pinned diagnostic overlay; never edits a checkout or GPU host."""

import argparse
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


COMMIT = "4ccff141dbe992794f9da6c3aa23535b4f72000d"
INDEXER = "python/sglang/srt/layers/attention/qsa/qsa_indexer.py"
PROCESSOR = "python/sglang/srt/managers/scheduler_components/batch_result_processor.py"
HELPER = "python/sglang/srt/layers/attention/qsa/runtime_capture.py"
# Only these files are changed by the overlay.
SOURCE_HASHES = {
    INDEXER: "a916967133fdb6b06e6094bfe3a9aa1ba275bf7d858d2f5c1bb39f8d0149461f",
    PROCESSOR: "22bd0ccfb1acfa4374a977ac0f104cee55652e6c5449d7c272d6775718828f7c",
}
# Read-only compatibility gates for scorer arithmetic/top-k, page/request
# mapping, MTP identity, graph/rank guards, and acceptance-result fields.
# This is a source contract, not verification of installed libraries/binaries.
DEPENDENCY_HASHES = {
    "python/sglang/srt/layers/attention/qsa/mqa.py": "84a52bb2cf61411862fcd328324cad1d05be264009bed2c12c027b315eeb6f19",
    "python/sglang/srt/layers/attention/qsa/metadata.py": "6486f5d5e10597b94320e16aff889f2381b0c361707ccad8f77430c4d7290e6e",
    "python/sglang/srt/layers/attention/qsa/kernel.py": "7e369f09293fb9b0872c21f0010247ec1e3a696b5ad4809f04d9a730b1031095",
    "python/sglang/srt/layers/attention/qsa/config.py": "c7694203855cf86e08c28da348735b0b808ed4b2c080949f1b1ccda67d2a69dd",
    "python/sglang/srt/layers/attention/qsa/glue.py": "cb8064f4fb56e76e9f04d03ac12be23b76ebcad1eb1662f91a97c7a3a0f4c2d8",
    "python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py": "7cb54a4440a3f6f9619227138398c173993a9f8deec8e6a6be50ce9067d50153",
    "python/sglang/srt/mem_cache/qsa_kv_pool.py": "f52178080fa5487c75bfb7070067c33d8071fdce823a69a6316640393191ff7e",
    "python/sglang/kernels/ops/elementwise/fast_topk.py": "65eeb1c14a111e651819bdac3ecca3a796f06feec4fb5c03f2ea08a5a65d29e4",
    "python/sglang/kernels/jit/csrc/elementwise/fast_topk.cuh": "5d28a70c484fc5820046ca5b4581c37368c31c15764376618cf32274aa9e05a1",
    "python/sglang/srt/model_executor/forward_batch_info.py": "c5465206368475dd2c1887de893679fa171793ee00eddaca8ce6acadc9237627",
    "python/sglang/srt/models/qwen4_exp.py": "a3981e148bc9ede4b84dea700f40831f86211cea29b817e8cc8feca1066e9f72",
    "python/sglang/srt/models/qwen4_exp_mtp.py": "2b2ec09230875279a75ae651a1d9e1d88999bc89748e9d0cb6b4a768ffc0e54e",
    "python/sglang/srt/models/qwen3_5.py": "f7e52f647d8e3dd8c980ae170569585ee118c782df055b21b1e93bbc28b1f9e8",
    "python/sglang/srt/model_executor/runner_utils/capture_mode.py": "539b4f62039a17c5e5e0a0637757d04de3cfda83a5ed37f611c2d5be2e4795db",
    "python/sglang/srt/distributed/parallel_state.py": "01c83b84f3c69757f52afb725b06b04c495fe6fb8cbb4deae661299b736e6d02",
    "python/sglang/srt/managers/schedule_batch.py": "348cd231d705a618c3d4a80a356470f92a430a941c97b429a8b5d16336fd035c",
    "python/sglang/srt/managers/utils.py": "23967b5942bc893d843f39028bb6d091c678125bc4378fa09499804b26f3c7ea",
}

QSA_HOOK = '''        if (_QSA_CAPTURE_ENABLED and forward_batch is not None
                and not self._qsa_capture_is_mtp
                and (forward_batch.forward_mode.is_decode()
                     or forward_batch.forward_mode.is_target_verify())):
            # Diagnostic only: all ranks reject graphs before any host read.
            if (get_is_capture_mode()
                    or getattr(indexer_metadata, "is_cuda_graph", False)
                    or (q.is_cuda and torch.cuda.is_current_stream_capturing())):
                raise RuntimeError("QSA capture requires eager execution; disable CUDA graphs")
            if get_world_rank() == 0:
                base_slots = forward_batch.req_pool_indices.detach().cpu().tolist()
                rids = forward_batch.rids
                if rids is None or len(rids) != len(base_slots) or len(set(base_slots)) != len(base_slots):
                    raise ValueError("QSA capture requires unique request slots and stable request IDs")
                by_slot = dict(zip(base_slots, rids))
                # In paged verify, metadata slots are expanded per query row.
                # token_to_batch_idx is row identity and is NOT this mapping.
                row_slots = indexer_metadata.req_pool_indices.detach().cpu().tolist()
                if len(row_slots) != q.shape[0] or any(slot not in by_slot for slot in row_slots):
                    raise ValueError("QSA capture request mapping does not cover query rows")
                capture_qsa(
                    q=q, compressed_cache=compressed_cache,
                    page_table=compressed_page_table, lengths=compressed_lengths,
                    query_positions=query_positions, native_scores=logits,
                    native_selected_ids=block_indices,
                    row_request_ids=[by_slot[slot] for slot in row_slots],
                    layer_id=self.layer_id,
                    forward_mode=("target_verify" if forward_batch.forward_mode.is_target_verify() else "decode"),
                    rank=0,
                )
'''

ACCEPTANCE_HOOK = '''        if (_QSA_CAPTURE_ENABLED and not batch.spec_algorithm.is_none()
                and result.num_correct_drafts_per_req_cpu is not None
                and get_world_rank() == 0):
            # Already CPU-resident after normalization/copy_done synchronization.
            # Native accepted slots include a bonus; they are not streamed output.
            capture_acceptance(
                request_ids=[req.rid for req in batch.reqs],
                accepted_drafts=result.num_correct_drafts_per_req_cpu,
                draft_positions=result.speculative_num_draft_tokens,
                per_request_metadata=[{
                    "forward_iter": batch.forward_iter,
                    "req_pool_idx": req.req_pool_idx,
                    "verification_count": req.spec_verify_ct,
                    "retraction_count": req.retraction_count,
                    "retracted": req.is_retracted,
                    "finished_before_output_extension": req.finished(),
                    "grammar_retained_count": len(next_token_ids[i]),
                } for i, req in enumerate(batch.reqs)],
                rank=0,
            )
'''


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("pinned source patch anchor is missing or ambiguous")
    return source.replace(old, new, 1)


def patch_source(path, source):
    digest = hashlib.sha256(source.encode()).hexdigest()
    if SOURCE_HASHES.get(path) != digest:
        raise ValueError(f"source revision mismatch: {path}; port the hook explicitly")
    capture_name = "capture_qsa" if path == INDEXER else "capture_acceptance"
    source = replace_once(source, "import torch\n", f'''import os
import torch

from sglang.srt.distributed.parallel_state import get_world_rank
from sglang.srt.layers.attention.qsa.runtime_capture import {capture_name}

_QSA_CAPTURE_ENABLED = bool(os.environ.get("SGLANG_QSA_CAPTURE_DIR"))
''')
    if path == INDEXER:
        source = replace_once(source, "        self.layer_id = int(layer_id)\n",
                              '        self.layer_id = int(layer_id)\n'
                              '        self._qsa_capture_is_mtp = "mtp" in prefix.split(".")\n')
        signature = '''    def select_decode_tokens(
        self,
        q: torch.Tensor,
        compressed_cache: torch.Tensor,
        compressed_page_table: torch.Tensor,
        compressed_lengths: torch.Tensor,
        max_model_len: int,
        query_positions: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> torch.Tensor:
'''
        source = replace_once(source, signature, signature.replace(
            "    ) -> torch.Tensor:",
            "        *,\n        forward_batch=None,\n        indexer_metadata=None,\n    ) -> torch.Tensor:"))
        source = replace_once(source, "        return expand_qsa_block_indices(\n",
                              QSA_HOOK + "        return expand_qsa_block_indices(\n")
        source = replace_once(source,
                              "                indexer_metadata.get_seqlens_int32(),\n            )\n",
                              "                indexer_metadata.get_seqlens_int32(),\n"
                              "                forward_batch=forward_batch,\n"
                              "                indexer_metadata=indexer_metadata,\n            )\n")
    else:
        anchor = "        self.metrics_reporter.num_generated_tokens += len(batch.reqs)\n"
        source = replace_once(source, anchor, ACCEPTANCE_HOOK + "\n" + anchor)
    ast.parse(source, filename=path)
    return source


def check_source_root(root):
    for path, expected in (SOURCE_HASHES | DEPENDENCY_HASHES).items():
        actual = hashlib.sha256((root / path).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"source revision mismatch: {root / path}; refusing this overlay")
    if (root / HELPER).exists():
        raise ValueError(f"helper path already exists: {root / HELPER}")


def prepare(repo, output):
    if output.exists():
        raise ValueError("output already exists; choose a new overlay directory")
    original, patched = {}, {}
    for path in SOURCE_HASHES:
        original[path] = subprocess.run(
            ["git", "show", f"{COMMIT}:{path}"], cwd=repo, check=True,
            capture_output=True, text=True).stdout
        patched[path] = patch_source(path, original[path])
    patched[HELPER] = Path(__file__).with_name("runtime_capture.py").read_text()
    ast.parse(patched[HELPER], filename=HELPER)
    manifest = {
        "status": "CPU-tested diagnostic overlay; not GPU-validated or deployed",
        "source_commit": COMMIT,
        "original_sha256": SOURCE_HASHES,
        "dependency_sha256": DEPENDENCY_HASHES,
        "compatibility_scope": "Listed source files only; installed libraries and binaries are not verified.",
        "overlay_sha256": {path: hashlib.sha256(data.encode()).hexdigest()
                           for path, data in patched.items()},
        "requirements": [
            "Check all modified and dependency source files with --check-source-root before mounting anything.",
            "Run an isolated diagnostic container, with CUDA graphs disabled and simulated acceptance disabled.",
            "Set capture identity and limits before launching the runtime; global rank zero writes.",
            "Capture calls synchronize and copy data; their timings are not serving benchmarks.",
            "Acceptance counts describe native accepted output slots, not actual emitted tokens.",
            "QSA and acceptance events have independent IDs; no exact cross-event join is established.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".capture-overlay-", dir=output.parent) as temp:
        stage = Path(temp) / "overlay"
        stage.mkdir()
        diff = []
        for path, data in patched.items():
            destination = stage / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(data)
            diff.extend(difflib.unified_diff(
                original.get(path, "").splitlines(keepends=True), data.splitlines(keepends=True),
                fromfile=f"a/{path}" if path in original else "/dev/null", tofile=f"b/{path}"))
        (stage / "capture.patch").write_text("".join(diff))
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.rename(stage, output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path, help="Create a new local overlay; no runtime mutation")
    mode.add_argument("--check-source-root", type=Path, help="Read-only runtime source compatibility check")
    args = parser.parse_args()
    try:
        if args.check_source_root:
            check_source_root(args.check_source_root)
            print(json.dumps({"status": "source_hashes_match", "source_commit": COMMIT,
                              "checked_source_files": len(SOURCE_HASHES) + len(DEPENDENCY_HASHES)}))
        else:
            manifest = prepare(args.repo, args.output)
            print(json.dumps({"status": manifest["status"], "output": str(args.output.resolve())}))
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"capture patch: {error}\n")


if __name__ == "__main__":
    main()
