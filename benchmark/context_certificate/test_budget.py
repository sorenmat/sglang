"""Run: python -m unittest discover -s benchmark/context_certificate -p test_budget.py"""

from dataclasses import replace
import unittest

from budget import GB, GIB, Scenario, calculate


class BudgetTest(unittest.TestCase):
    def setUp(self):
        self.scenario = Scenario(conditional_acceptance=.9, refinement_fraction=.1,
                                 kv_union_multiplier=3, kv_hit_rate=.8)

    def test_capacity_anchors_and_ideal_tp(self):
        single = calculate(self.scenario)["memory_GiB"]
        self.assertAlmostEqual(single["host_static_total"], 309.87)
        self.assertEqual(single["host_items"]["target_bf16_kv"], 192)
        self.assertEqual(single["host_items"]["target_bf16_index"], 6)
        self.assertEqual(single["host_items"]["draft_one_qsa_layer_bf16_kv_and_index"], 16.5)
        self.assertAlmostEqual(single["per_gpu_total"], 93.684375)
        two = calculate(replace(self.scenario, gpu_count=2))["memory_GiB"]
        self.assertEqual(two["host_static_total"], single["host_static_total"])
        self.assertAlmostEqual(two["per_gpu_total"], (single["per_gpu_total"] - 4) / 2 + 4)
        cramped = calculate(replace(self.scenario, host_gib=235 * GB / GIB))["memory_GiB"]
        self.assertFalse(cramped["host_static_fits"])
        self.assertLess(cramped["host_headroom_before_unmodeled_buffers"], -90)

    def test_traffic_units_and_position_reuse(self):
        r = calculate(self.scenario)
        traffic = r["traffic_bytes_per_pass"]
        self.assertAlmostEqual(r["speculation"]["emitted_per_request_per_pass_including_bonus"],
                               7.17570463519)
        self.assertEqual(traffic["all_gpus_items"]["target_compact_index_one_shared_scan"], 3.1875 * GIB)
        self.assertEqual(traffic["selected_kv_all_positions_without_read_reuse"], 32 * 12 * 2051 * 2048 * 12)
        self.assertAlmostEqual(traffic["host_to_device_items"]["original_index_union"] / GIB, .6)
        reuse = calculate(replace(self.scenario, reuse_kv_on_gpu=True))["traffic_bytes_per_pass"]
        self.assertEqual(reuse["all_gpus_items"]["target_selected_kv_reads"],
                         traffic["selected_kv_all_positions_without_read_reuse"] / 4)
        two = calculate(replace(self.scenario, gpu_count=2))
        self.assertEqual(two["traffic_bytes_per_pass"]["host_to_device_total"], traffic["host_to_device_total"])
        self.assertAlmostEqual(two["conditional_latency_floor_ms_per_pass"]["gpu_traffic_only"],
                               r["conditional_latency_floor_ms_per_pass"]["gpu_traffic_only"] / 2)
        self.assertAlmostEqual(r["conditional_latency_floor_ms_per_pass"]["host_link_payload_only"],
                               traffic["host_to_device_total"] / (20 * GB) * 1000)
        impossible = calculate(replace(self.scenario, hot_tokens=4096))
        self.assertFalse(impossible["traffic_assumptions"]["kv_hit_rate_within_capacity"])
        too_little_refinement = calculate(replace(self.scenario, refinement_fraction=.001))
        self.assertFalse(too_little_refinement["traffic_assumptions"]
                         ["index_refinement_covers_minimum_selected_records"])
        self.assertAlmostEqual(r["traffic_assumptions"]
                               ["minimum_refinement_union_fraction_without_original_index_cache"],
                               512 / 65536)

    def test_runtime_verification_window_limit(self):
        longer = calculate(self.scenario)
        self.assertFalse(longer["runtime_compatibility"]["passes_inspected_window_guard"])
        self.assertTrue(any("unsupported by the inspected QSA runtime" in warning
                            for warning in longer["warnings"]))
        four = calculate(replace(self.scenario, positions=4, conditional_acceptance=None,
                                 emitted_per_pass=3))
        self.assertTrue(four["runtime_compatibility"]["passes_inspected_window_guard"])
        self.assertFalse(four["runtime_compatibility"]
                         ["requires_pending_index_state_and_write_plan_redesign"])
        self.assertEqual(four["speculation"]["emitted_per_request_per_pass_including_bonus"], 3)
        self.assertFalse(four["fits_conditional_traffic_time_budget"])
        self.assertTrue(any("exceeds the target pass time" in message for message in four["warnings"]))

    def test_original_resident_index_control(self):
        scenario = replace(self.scenario, gpu_count=2, positions=4, conditional_acceptance=None,
                           emitted_per_pass=3, hot_tokens=8192)
        compact = calculate(scenario)
        original = calculate(replace(scenario, index_placement="original-on-gpu", refinement_fraction=0))
        memory = original["memory_GiB"]
        self.assertGreater(memory["per_gpu_total"], compact["memory_GiB"]["per_gpu_total"])
        self.assertAlmostEqual(memory["per_gpu_total"], 51.584375)
        self.assertTrue(memory["gpu_modeled_fits"])
        self.assertEqual(memory["ideal_TP_per_gpu_items"]["target_bf16_index"], 3)
        self.assertEqual(memory["host_static_total"], compact["memory_GiB"]["host_static_total"])
        traffic = original["traffic_bytes_per_pass"]
        self.assertLess(traffic["host_to_device_total"], compact["traffic_bytes_per_pass"]["host_to_device_total"])
        self.assertEqual(traffic["host_to_device_items"]["original_index_union"], 0)
        self.assertEqual(traffic["all_gpus_items"]["target_compact_index_one_shared_scan"], 0)
        self.assertEqual(traffic["all_gpus_items"]["target_resident_bf16_index_one_shared_scan"], 6 * GIB)
        self.assertEqual(traffic["all_gpus_items"]["target_original_index_one_shared_refinement_read"], 0)
        self.assertTrue(original["traffic_assumptions"]["index_refinement_fraction_ignored"])
        self.assertIsNone(original["traffic_assumptions"]["index_refinement_covers_minimum_selected_records"])
        self.assertFalse(any("Refinement union" in warning for warning in original["warnings"]))

    def test_acceptance_edges_and_invalid_inputs(self):
        for p, expected in [(0, 1), (1, 12)]:
            self.assertEqual(calculate(replace(self.scenario, conditional_acceptance=p))
                             ["speculation"]["emitted_per_request_per_pass_including_bonus"], expected)
        for changes in ({"conditional_acceptance": 1.01}, {"kv_hit_rate": -.1},
                        {"refinement_fraction": float("nan")}, {"positions": 0},
                        {"positions": 1.5}, {"gpu_count": 3}, {"weights_total_gib": -1},
                        {"host_link_gbs": 0}, {"conditional_acceptance": None},
                        {"emitted_per_pass": 2}, {"kv_union_multiplier": 13}, {"index_placement": "unknown"},
                        {"conditional_acceptance": None, "emitted_per_pass": 13}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                calculate(replace(self.scenario, **changes))


if __name__ == "__main__":
    unittest.main()
