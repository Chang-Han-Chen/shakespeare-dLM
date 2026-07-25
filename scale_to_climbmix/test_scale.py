"""Correctness tests for the self-contained ClimbMix scaling implementation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from analyze import fit_l1_quadratic
from analyze_followup_isoflop import STUDIES, fit_profiles as fit_followup_profiles
from analyze_refinement import fit_local_profiles
from analyze_scaleup import fit_scaleup_profiles
from config import (
    AR_LARGE_MODEL_SPECS,
    COMPUTE_BUDGETS,
    MASK_ID,
    MAX_STEPS,
    MIN_STEPS,
    MODEL_BY_LABEL,
    MODEL_SPECS,
    REFINEMENT_MODEL_SPECS,
    VOCAB_SIZE,
    is_feasible,
    steps_for,
)
from data import ClimbMixData, corrupt
from curriculum_config import (
    P_AR_VALUES,
    is_feasible as curriculum_is_feasible,
    mixed_steps_for,
    phase_steps_for,
    pure_ar_steps_for,
    realized_flops as curriculum_realized_flops,
)
from model import BlockDiffusionTransformer, make_dual_stream_mask
from train import diffusion_nelbo, wsd_learning_rate
from fixed_steps_config import FIXED_STEP_TARGETS, split_fixed_steps


EXPECTED_COUNTS = {
    "0.14M": 137_824,
    "0.29M": 287_424,
    "0.5M": 504_288,
    "1M": 965_920,
    "2M": 1_985_536,
    "4M": 3_920_256,
    "8M": 7_979_296,
}


class ConfigTests(unittest.TestCase):
    def test_parameter_table(self):
        self.assertEqual(
            {spec.label: spec.n_params for spec in MODEL_SPECS},
            EXPECTED_COUNTS,
        )

    def test_architecture_constraints(self):
        for spec in MODEL_SPECS:
            self.assertGreaterEqual(spec.n_layer, 2)
            self.assertEqual(spec.head_dim, 16)

    def test_experimental_10m_configuration(self):
        spec = MODEL_BY_LABEL["10M"]
        self.assertEqual(spec.n_params, 9_692_032)
        self.assertEqual((spec.n_layer, spec.d_model, spec.n_head), (13, 224, 14))
        self.assertEqual(spec.head_dim, 16)

    def test_first_scaleup_wave_configurations(self):
        expected = {
            "3.4M": (3_422_016, 9, 144, 9),
            "4.4M": (4_411_840, 10, 160, 10),
            "5.6M": (5_550_336, 11, 176, 11),
            "6.9M": (6_886_272, 12, 192, 12),
            "8.5M": (8_502_208, 13, 208, 13),
            "10.3M": (10_296_384, 14, 224, 14),
        }
        for label, (n_params, n_layer, d_model, n_head) in expected.items():
            spec = MODEL_BY_LABEL[label]
            self.assertEqual(spec.n_params, n_params)
            self.assertEqual(
                (spec.n_layer, spec.d_model, spec.n_head),
                (n_layer, d_model, n_head),
            )
            self.assertEqual(spec.head_dim, 16)

    def test_historical_refinement_configurations(self):
        expected = {
            "0.21M": (210_576, 2, 24, 1, 24),
            "0.35M": (350_464, 7, 32, 2, 16),
            "0.45M": (448_800, 2, 48, 3, 16),
            "0.73M": (725_632, 4, 64, 4, 16),
            "1.20M": (1_198_720, 7, 80, 5, 16),
            "1.34M": (1_340_544, 5, 96, 6, 16),
            "1.56M": (1_562_112, 7, 96, 6, 16),
            "1.67M": (1_672_896, 8, 96, 6, 16),
            "2.60M": (2_595_712, 11, 112, 7, 16),
        }
        self.assertEqual(
            {spec.label for spec in REFINEMENT_MODEL_SPECS},
            set(expected),
        )
        for label, (n_params, n_layer, d_model, n_head, head_dim) in expected.items():
            spec = MODEL_BY_LABEL[label]
            self.assertEqual(spec.n_params, n_params)
            self.assertEqual(
                (spec.n_layer, spec.d_model, spec.n_head, spec.head_dim),
                (n_layer, d_model, n_head, head_dim),
            )

    def test_large_ar_configurations(self):
        expected = {
            "43.7M": (43_734_592, 25, 368, 23),
            "54.5M": (54_483_456, 29, 384, 24),
            "68.5M": (68_510_016, 29, 432, 27),
            "85.8M": (85_832_320, 34, 448, 28),
            "107.7M": (107_703_424, 35, 496, 31),
            "134.8M": (134_838_528, 39, 528, 33),
        }
        self.assertEqual(
            {spec.label for spec in AR_LARGE_MODEL_SPECS},
            set(expected),
        )
        for label, (n_params, n_layer, d_model, n_head) in expected.items():
            spec = MODEL_BY_LABEL[label]
            self.assertEqual(
                (spec.n_params, spec.n_layer, spec.d_model, spec.n_head),
                (n_params, n_layer, d_model, n_head),
            )
            self.assertEqual(spec.head_dim, 16)

    def test_feasible_coverage(self):
        expected = [5, 7, 6, 4, 3]
        actual = [
            sum(is_feasible(budget, spec) for spec in MODEL_SPECS)
            for budget in COMPUTE_BUDGETS
        ]
        self.assertEqual(actual, expected)
        for budget in COMPUTE_BUDGETS:
            for spec in MODEL_SPECS:
                if is_feasible(budget, spec):
                    self.assertLessEqual(MIN_STEPS, steps_for(budget, spec))
                    self.assertLessEqual(steps_for(budget, spec), MAX_STEPS)

    def test_compute_accounting(self):
        expected_flops = {
            "0.14M": 1_259_616,
            "0.29M": 2_660_544,
            "0.5M": 6_045_984,
            "1M": 11_581_920,
            "2M": 27_934_368,
            "4M": 59_388_768,
            "8M": 116_135_136,
        }
        self.assertEqual(
            {
                spec.label: spec.training_flops_per_clean_token
                for spec in MODEL_SPECS
            },
            expected_flops,
        )
        for spec in MODEL_SPECS:
            self.assertLess(
                spec.autoregressive_training_flops_per_clean_token,
                spec.training_flops_per_clean_token,
            )

    def test_refinement_profiles_use_immediate_neighbors(self):
        rows = []
        for budget in COMPUTE_BUDGETS:
            for label, n_params, loss in (
                ("1", 1_000_000, 3.0),
                ("2", 2_000_000, 2.0),
                ("4", 4_000_000, 1.0),
                ("8", 8_000_000, 2.0),
            ):
                rows.append(
                    {
                        "budget": budget,
                        "size": label,
                        "n_params": n_params,
                        "val_nelbo": loss,
                        "training_flops_per_clean_token": 12 * n_params,
                    }
                )
        profiles = fit_local_profiles(rows)
        self.assertEqual(len(profiles), len(COMPUTE_BUDGETS))
        for profile in profiles:
            self.assertEqual(profile["support_sizes"], ["2", "4", "8"])
            self.assertAlmostEqual(profile["n_opt"], 4_000_000, delta=1.0)

    def test_curriculum_compute_and_steps(self):
        for budget in COMPUTE_BUDGETS:
            for spec in MODEL_SPECS:
                if not is_feasible(budget, spec):
                    continue
                self.assertGreater(pure_ar_steps_for(budget, spec), steps_for(budget, spec))
                previous_total = 0
                for p_ar in P_AR_VALUES:
                    self.assertTrue(curriculum_is_feasible(budget, spec, p_ar))
                    ar_steps, bd_steps = phase_steps_for(budget, spec, p_ar)
                    total = mixed_steps_for(budget, spec, p_ar)
                    self.assertEqual(ar_steps + bd_steps, total)
                    self.assertGreater(total, previous_total)
                    previous_total = total
                    self.assertLessEqual(
                        curriculum_realized_flops(ar_steps, bd_steps, spec),
                        budget,
                    )

    def test_compute_optimal_fixed_step_targets(self):
        previous_steps = 0
        for target in FIXED_STEP_TARGETS:
            self.assertGreaterEqual(target.total_steps, 150)
            self.assertGreater(target.total_steps, previous_steps)
            self.assertLessEqual(
                target.realized_full_bd_compute,
                target.predicted_compute,
            )
            for p_ar in P_AR_VALUES:
                ar_steps, bd_steps = split_fixed_steps(target.total_steps, p_ar)
                self.assertEqual(ar_steps + bd_steps, target.total_steps)
                self.assertGreater(ar_steps, 0)
                self.assertGreater(bd_steps, 0)
            previous_steps = target.total_steps


class ModelTests(unittest.TestCase):
    def test_attention_mask_structure(self):
        length, block = 8, 2
        mask = make_dual_stream_mask(length, block)[0, 0]
        self.assertTrue(bool(mask[0, 0] and mask[0, 1]))
        self.assertFalse(bool(mask[0, 2]))
        self.assertFalse(bool(mask[0, length]))
        self.assertTrue(bool(mask[4, length]))
        self.assertTrue(bool(mask[4, length + 3]))
        self.assertFalse(bool(mask[4, length + 4]))
        self.assertFalse(bool(mask[length:, :length].any()))

    def test_actual_parameter_counts(self):
        for spec in MODEL_SPECS:
            model = BlockDiffusionTransformer(spec)
            self.assertEqual(model.counted_parameter_count(), spec.n_params)
        experimental = MODEL_BY_LABEL["10M"]
        model = BlockDiffusionTransformer(experimental)
        self.assertEqual(model.counted_parameter_count(), experimental.n_params)

    def test_forward_backward_and_mask_target_exclusion(self):
        model = BlockDiffusionTransformer(MODEL_SPECS[0])
        clean = torch.randint(0, MASK_ID, (2, 256))
        probabilities = torch.full((2, 64), 0.5)
        noisy, masked, token_probability = corrupt(clean, probabilities)
        logits = model(noisy, clean)
        self.assertEqual(tuple(logits.shape), (2, 256, VOCAB_SIZE))
        loss = diffusion_nelbo(logits, clean, masked, token_probability)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_block_32_forward_backward(self):
        model = BlockDiffusionTransformer(MODEL_SPECS[0], block_len=32)
        clean = torch.randint(0, MASK_ID, (2, 256))
        probabilities = torch.full((2, 8), 0.5)
        noisy, masked, token_probability = corrupt(clean, probabilities)
        logits = model(noisy, clean)
        loss = diffusion_nelbo(logits, clean, masked, token_probability)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_autoregressive_forward_is_causal(self):
        model = BlockDiffusionTransformer(MODEL_SPECS[0]).eval()
        tokens_a = torch.randint(0, MASK_ID, (1, 256))
        tokens_b = tokens_a.clone()
        tokens_b[:, 128:] = torch.randint(0, MASK_ID, (1, 128))
        with torch.no_grad():
            logits_a = model.forward_ar(tokens_a)
            logits_b = model.forward_ar(tokens_b)
        self.assertTrue(torch.equal(logits_a[:, :128], logits_b[:, :128]))
    def test_no_clean_target_leakage(self):
        model = BlockDiffusionTransformer(MODEL_SPECS[0]).eval()
        noisy = torch.randint(0, VOCAB_SIZE, (1, 256))
        clean_a = torch.randint(0, MASK_ID, (1, 256))
        clean_b = clean_a.clone()
        clean_b[:, 4:] = torch.randint(0, MASK_ID, (1, 252))
        with torch.no_grad():
            logits_a = model(noisy, clean_a)
            logits_b = model(noisy, clean_b)
        self.assertTrue(torch.equal(logits_a[:, :4], logits_b[:, :4]))


class DataTests(unittest.TestCase):
    def test_one_pass_cross_shard_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenized = root / "tokenized"
            tokenized.mkdir()
            train_entries = []
            for index, values in enumerate(
                (
                    np.arange(10_000, dtype=np.uint16),
                    np.arange(10_000, 20_000, dtype=np.uint16),
                )
            ):
                path = tokenized / f"train_{index}.bin"
                values.tofile(path)
                train_entries.append({"path": path.name, "token_count": len(values)})
            val_path = tokenized / "val.bin"
            np.arange(20_000, dtype=np.uint16).tofile(val_path)
            manifest = {
                "dtype": "uint16",
                "train": train_entries,
                "val": [{"path": val_path.name, "token_count": 20_000}],
            }
            manifest_path = tokenized / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            dataset = ClimbMixData.load(torch.device("cpu"), manifest_path)
            batch = dataset.train_batch(0, 64)
            self.assertEqual(tuple(batch.shape), (64, 256))
            self.assertEqual(int(batch[0, 0]), 0)
            self.assertEqual(int(batch[-1, -1]), 16_383)
            rank_batches = [
                dataset.train_batch(0, 16, rank=rank, world_size=4)
                for rank in range(4)
            ]
            self.assertTrue(torch.equal(torch.cat(rank_batches), batch))
            with self.assertRaises(IndexError):
                dataset.train_batch(1, 64)


class ScheduleAndAnalysisTests(unittest.TestCase):
    def test_wsd_shape(self):
        total, peak = 1000, 0.003
        self.assertAlmostEqual(wsd_learning_rate(49, total, peak), peak)
        self.assertAlmostEqual(wsd_learning_rate(849, total, peak), peak)
        self.assertLess(wsd_learning_rate(900, total, peak), peak)
        self.assertEqual(wsd_learning_rate(999, total, peak), 0.0)

    def test_l1_quadratic(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = x**2 - 0.5 * x + 1.25
        coefficients, support = fit_l1_quadratic(x, y)
        self.assertTrue(np.allclose(coefficients, [1.0, -0.5, 1.25]))
        self.assertEqual(len(support), 3)

    def test_scaleup_fit_uses_local_bracket(self):
        rows = []
        for n_params in (100, 200, 400, 800):
            loss = (np.log10(n_params) - np.log10(200)) ** 2 + 1.0
            if n_params == 800:
                loss = 1.02
            rows.append(
                {
                    "budget": 1e12,
                    "size": str(n_params),
                    "n_params": n_params,
                    "val_nelbo": loss,
                    "training_flops_per_clean_token": 12 * n_params,
                }
            )
        fit = fit_scaleup_profiles(rows)[0]
        self.assertEqual(fit["support_sizes"], ["100", "200", "400"])
        self.assertAlmostEqual(fit["n_opt"], 200.0)

    def test_followup_ar_fit_uses_local_bracket(self):
        rows = []
        for n_params in (100, 200, 400, 800):
            rows.append(
                {
                    "budget": 1e12,
                    "size": str(n_params),
                    "n_params": n_params,
                    "val_ar_ce": (
                        np.log10(n_params) - np.log10(400)
                    )
                    ** 2
                    + 1.0,
                    "flash_causal_training_flops_per_clean_token": (
                        6 * n_params
                    ),
                }
            )
        fit = fit_followup_profiles(rows, STUDIES["ar"])[0]
        self.assertEqual(fit["support_sizes"], ["200", "400", "800"])
        self.assertAlmostEqual(fit["n_opt"], 400.0)
        self.assertAlmostEqual(
            fit["d_opt"],
            1e12 / (6 * 400),
            delta=1e-4,
        )


if __name__ == "__main__":
    unittest.main()
