"""Economics of index screening with MEASURED refinement fractions.

Recomputes the runtime-compatible S=4 budget comparison (32 requests,
256k context, 12 QSA layers, 2 GPUs, 1 TB/s GPU bandwidth, 20 GB/s host
link, 80 tokens/s per request) replacing the budget's ASSUMED 10%
refinement fraction with the fractions measured on the six real H200
captures. Pure arithmetic on existing artifacts; no new claims.
"""

import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GB = 1e9


def measured_fractions():
    fractions = {}
    for path in sorted(glob.glob(str(ROOT / "artifacts/h200-20260929/replay/*.json"))):
        for case in json.load(open(path))["cases"]:
            fractions.setdefault((case["bits"], case["bound_mode"]), []).append(
                case["refine_union_fraction"])
    return {key: (min(v), sum(v) / len(v), max(v)) for key, v in fractions.items()}


def main():
    compact = json.load(open(ROOT / "artifacts/budget-tp2-s4-emitted3.json"))
    original = json.load(open(ROOT / "artifacts/budget-tp2-s4-emitted3-original-index-hot8192.json"))
    t_compact = compact["traffic_bytes_per_pass"]
    t_original = original["traffic_bytes_per_pass"]
    link = compact["inputs"]["host_link_gbs"]
    gpu_bw = compact["inputs"]["gpu_bandwidth_gbs"]
    allowed_ms = compact["conditional_latency_floor_ms_per_pass"]["allowed_at_target"]
    original_index_bytes = t_original["all_gpus_items"]["target_resident_bf16_index_one_shared_scan"]
    compact_scan_bytes = t_compact["all_gpus_items"]["target_compact_index_one_shared_scan"]
    baseline_h2d = t_original["host_to_device_total"]

    print("measured S=4 refine-union fractions (min/mean/max over six captures):")
    for (bits, mode), (lo, mean, hi) in sorted(measured_fractions().items()):
        print(f"  bits={bits} {mode:5s}: {lo:.3f}/{mean:.3f}/{hi:.3f}")

    print(f"\nbaseline (original index on GPU): GPU index scan {original_index_bytes/GB:.2f} GB/pass, "
          f"GPU floor {original['conditional_latency_floor_ms_per_pass']['gpu_traffic_only']:.2f} ms, "
          f"link floor {original['conditional_latency_floor_ms_per_pass']['host_link_payload_only']:.2f} ms")
    report = {"status": "arithmetic on existing artifacts; no new measurements",
              "allowed_ms_per_pass": allowed_ms,
              "measured_fractions": {f"{b}-{m}": v for (b, m), v in measured_fractions().items()},
              "screened_variants": []}
    for bits in (8, 6):
        lo, mean, hi = measured_fractions()[(bits, "joint")]
        for label, f in (("best", lo), ("mean", mean), ("worst", hi)):
            gpu_index = compact_scan_bytes + f * original_index_bytes
            gpu_saving = original_index_bytes - gpu_index
            h2d = baseline_h2d + f * original_index_bytes  # refinement crosses the link
            per_gpu_bytes = (t_original["ideal_TP_per_gpu_total"]
                             - original_index_bytes / 2 + gpu_index / 2)
            gpu_floor_ms = per_gpu_bytes / (gpu_bw * GB) * 1e3
            link_ms = h2d / (link * GB) * 1e3
            variant = {
                "bits": bits, "case": label, "refine_union": round(f, 3),
                "gpu_index_bytes_GB": round(gpu_index / GB, 2),
                "gpu_index_saving_GB_per_pass": round(gpu_saving / GB, 2),
                "gpu_index_saving_percent": round(100 * gpu_saving / original_index_bytes, 1),
                "gpu_floor_ms": round(gpu_floor_ms, 2),
                "link_payload_GB": round(h2d / GB, 2),
                "link_floor_ms_at_20GBps": round(link_ms, 1),
            }
            report["screened_variants"].append(variant)
            print(f"screened bits={bits} {label:5s} f={f:.3f}: GPU index {gpu_index/GB:.2f} GB "
                  f"(saving {gpu_saving/GB:.2f} GB = {100*gpu_saving/original_index_bytes:.1f}%), "
                  f"GPU floor {gpu_floor_ms:.2f} ms, link floor {link_ms:.1f} ms")

    print(f"\nallowed pass time {allowed_ms} ms. Both unscreened scenarios already exceed it; "
          f"screening moves the GPU floor by ~1 ms and moves the LINK floor the wrong way.")
    out = ROOT / "artifacts/h200-20260929/screening-economics.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
