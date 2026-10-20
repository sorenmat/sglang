"""Build the P2.2 host-tier overlay from the pinned runtime source.

Extracts qwen_sparse_attn_backend.py at the pinned commit, inserts the
stage hooks at the two slot-resolution sites, and writes an overlay
directory + manifest for mounting into the container. Refuses to build if
an anchor is missing (source drifted).
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "4ccff141dbe992794f9da6c3aa23535b4f72000d"
BACKEND = "python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py"
HOSTTIER_DST = "python/sglang/srt/layers/attention/qsa/hosttier.py"
HOSTTIER23_DST = "python/sglang/srt/layers/attention/qsa/hosttier23.py"
HOSTTIER23M_DST = "python/sglang/srt/layers/attention/qsa/hosttier23m.py"

HOOK = (
    "        try:\n"
    "            from sglang.srt.layers.attention.qsa.hosttier import stage_physical_slots\n"
    "            stage_physical_slots(slots.long(), self.token_to_kv_pool)\n"
    "        except ImportError:\n"
    "            pass\n"
)

# The extend-path hook was removed: _logical_to_physical is a @staticmethod
# (no `self`), and the extend branch is not on this model's verify path.
# The single hook lives at _forward_paged_attention, proven live by the
# 2026-09-30 probe run (NameError traceback reached it from the main
# backend during a real decode).

def pinned(path):
    return subprocess.check_output(["git", "show", f"{COMMIT}:{path}"], cwd=ROOT).decode()


def sha(data):
    return hashlib.sha256(data.encode()).hexdigest()


def main():
    source = pinned(BACKEND)
    original_hash = sha(source)
    patched = source

    # forward_extend is the live verify path (66 calls in the window-3
    # census, vs 8 for forward_decode); hook it with the same pattern.
    extend_marker = "    def forward_extend(\n"
    eidx = patched.find(extend_marker)
    if eidx == -1:
        sys.exit("forward_extend anchor missing; source drifted")
    eclose = patched.find("\n    ) -> torch.Tensor:\n", eidx)
    if eclose == -1:
        sys.exit("forward_extend signature close not found")
    einsert = eclose + len("\n    ) -> torch.Tensor:\n")
    extend_hook = (
        "        try:\n"
        "            import os as _os\n"
        "            if _os.environ.get(\"SGLANG_HOSTTIER_MODE\") not in (\"P23\", \"P23M\"):\n"
        "                from sglang.srt.layers.attention.qsa.hosttier import stage_physical_slots\n"
        "                _md = getattr(self, \"_resolve_metadata\", None)\n"
        "                if _md is not None and topk_indices is not None:\n"
        "                    stage_physical_slots(\n"
        "                        self._logical_to_physical(topk_indices, _md(forward_batch)).long(),\n"
        "                        self.token_to_kv_pool)\n"
        "        except (ImportError, AttributeError):\n"
        "            pass\n"
    )
    patched = patched[:einsert] + extend_hook + patched[einsert:]

    # forward_decode also carries topk_indices; kept for completeness.
    # The live verify path is forward_decode (proven by the window-3
    # entrypoint probe); _forward_paged_attention only runs during warmup.
    decode_marker = "    def forward_decode(\n"
    didx = patched.find(decode_marker)
    if didx == -1:
        sys.exit("forward_decode anchor missing; source drifted")
    dclose = patched.find("\n    ) -> torch.Tensor:\n", didx)
    if dclose == -1:
        sys.exit("forward_decode signature close not found")
    dinsert = dclose + len("\n    ) -> torch.Tensor:\n")
    decode_hook = (
        "        try:\n"
        "            import os as _os\n"
        "            if _os.environ.get(\"SGLANG_HOSTTIER_MODE\") not in (\"P23\", \"P23M\"):\n"
        "                from sglang.srt.layers.attention.qsa.hosttier import stage_physical_slots\n"
        "                _md = getattr(self, \"_resolve_metadata\", None)\n"
        "                if _md is not None and topk_indices is not None:\n"
        "                    stage_physical_slots(\n"
        "                        self._logical_to_physical(topk_indices, _md(forward_batch)).long(),\n"
        "                        self.token_to_kv_pool)\n"
        "        except (ImportError, AttributeError):\n"
        "            pass\n"
    )
    patched = patched[:dinsert] + decode_hook + patched[dinsert:]

    # paged path: insert the hook after the def line's parameter list closes.
    marker = "    def _forward_paged_attention(\n"
    idx = patched.find(marker)
    if idx == -1:
        sys.exit("paged anchor missing; source drifted")
    close = patched.find("\n    ) -> torch.Tensor:\n", idx)
    if close == -1:
        sys.exit("paged signature close not found")
    insert_at = close + len("\n    ) -> torch.Tensor:\n")
    paged_hook = (
        "        try:\n"
        "            import os as _os\n"
        "            if _os.environ.get(\"SGLANG_HOSTTIER_MODE\") not in (\"P23\", \"P23M\"):\n"
        "                from sglang.srt.layers.attention.qsa.hosttier import stage_physical_slots\n"
        "                _md = getattr(self, \"_resolve_metadata\", None)\n"
        "                if _md is not None and topk_indices is not None:\n"
        "                    stage_physical_slots(\n"
        "                        self._logical_to_physical(topk_indices, _md(forward_batch)).long(),\n"
        "                        self.token_to_kv_pool)\n"
        "        except (ImportError, AttributeError):\n"
        "            pass\n"
    )
    patched = patched[:insert_at] + paged_hook + patched[insert_at:]

    out = ROOT / "artifacts/hosttier-overlay-20260930"
    (out / "python/sglang/srt/layers/attention/qsa").mkdir(parents=True, exist_ok=True)
    (out / "python/sglang/srt/layers/attention").mkdir(parents=True, exist_ok=True)
    backend_dst = out / BACKEND
    backend_dst.parent.mkdir(parents=True, exist_ok=True)
    # Unconditional install at module bottom: probe wraps all backend
    # entrypoints at server start, independent of which paths run.
    patched = patched + (
        "\nfrom sglang.srt.layers.attention.qsa.hosttier import probe_install  # noqa: E402\n"
        "probe_install(globals())\n"
    )
    backend_dst.write_text(patched)
    (out / HOSTTIER_DST).write_text((ROOT / "host_kv/hosttier.py").read_text())
    (out / HOSTTIER23_DST).write_text((ROOT / "host_kv/hosttier23.py").read_text())
    (out / HOSTTIER23M_DST).write_text((ROOT / "host_kv/hosttier23m.py").read_text())
    manifest = {
        "status": "P2.2 correctness-slice overlay; hooks are no-ops unless "
                  "SGLANG_HOSTTIER_ENABLE=1",
        "source_commit": COMMIT,
        "original_sha256": {BACKEND: original_hash},
        "overlay_sha256": {BACKEND: sha(patched),
                           HOSTTIER_DST: sha((ROOT / "host_kv/hosttier.py").read_text()),
                           HOSTTIER23_DST: sha((ROOT / "host_kv/hosttier23.py").read_text()),
                           HOSTTIER23M_DST: sha((ROOT / "host_kv/hosttier23m.py").read_text())},
        "anchors": ["_forward_paged_attention entry (decode/verify; the only hook)",
                    "module-bottom probe_install wrapping every entrypoint"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"overlay": str(out), "patched_bytes": len(patched),
                      "hooks": 2, "original_sha256": original_hash}, indent=2))


if __name__ == "__main__":
    main()
