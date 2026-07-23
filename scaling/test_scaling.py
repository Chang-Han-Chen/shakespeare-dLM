"""Dependency-free correctness tests for the clean scaling implementation."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from analyze import fit_l1_quadratic, fit_loss_law_with_floor
from config import (
    COMPUTE_BUDGETS,
    MAX_STEPS,
    MIN_STEPS,
    MODEL_SPECS,
    VOCAB_SIZE,
    is_feasible,
    steps_for,
)
from curriculum_config import (
    P_AR_VALUES,
    average_training_flops_per_clean_token,
    is_feasible as curriculum_is_feasible,
    realized_flops as curriculum_realized_flops,
    split_phase_steps,
    steps_for as curriculum_steps_for,
)
from curriculum_run_sweep import baseline_learning_rate, run_grid as curriculum_run_grid
from data import corrupt
from model import BlockDiffusionTransformer, make_dual_stream_mask
from train import diffusion_nelbo, optimizer_for, wsd_learning_rate


EXPECTED_COUNTS = {
    "0.002M": 2_232,
    "0.005M": 4_308,
    "0.01M": 11_152,
    "0.02M": 22_488,
    "0.04M": 39_968,
    "0.1M": 114_192,
    "0.2M": 205_504,
    "0.4M": 393_360,
    "0.8M": 781_920,
    "1.6M": 1_591_680,
}


class ConfigTests(unittest.TestCase):
    def test_parameter_table(self):
        self.assertEqual({spec.label: spec.n_params for spec in MODEL_SPECS}, EXPECTED_COUNTS)

    def test_head_dimensions(self):
        actual = {spec.label: spec.head_dim for spec in MODEL_SPECS}
        self.assertEqual(actual["0.002M"], 8)
        self.assertEqual(actual["0.005M"], 6)
        self.assertEqual([actual[label] for label in ("0.01M", "0.02M", "0.04M")], [8, 8, 8])
        self.assertEqual(
            [actual[label] for label in ("0.1M", "0.2M", "0.4M", "0.8M", "1.6M")],
            [16] * 5,
        )

    def test_feasible_coverage(self):
        expected = [5, 5, 7, 8, 6]
        actual = [sum(is_feasible(budget, spec) for spec in MODEL_SPECS) for budget in COMPUTE_BUDGETS]
        self.assertEqual(actual, expected)
        for budget in COMPUTE_BUDGETS:
            for spec in MODEL_SPECS:
                if is_feasible(budget, spec):
                    self.assertLessEqual(MIN_STEPS, steps_for(budget, spec))
                    self.assertLessEqual(steps_for(budget, spec), MAX_STEPS)

    def test_dense_attention_compute_accounting(self):
        by_label = {spec.label: spec for spec in MODEL_SPECS}
        self.assertEqual(by_label["0.01M"].training_flops_per_clean_token, 715_968)
        self.assertEqual(by_label["1.6M"].training_flops_per_clean_token, 31_606_272)
        correction = [
            spec.effective_compute_parameters / spec.n_params
            for spec in MODEL_SPECS
        ]
        self.assertTrue(all(left > right for left, right in zip(correction, correction[1:])))

    def test_curriculum_compute_accounting(self):
        by_label = {spec.label: spec for spec in MODEL_SPECS}
        small = by_label["0.01M"]
        large = by_label["1.6M"]
        self.assertEqual(small.autoregressive_training_flops_per_clean_token, 213_696)
        self.assertEqual(large.autoregressive_training_flops_per_clean_token, 12_682_752)
        for spec in MODEL_SPECS:
            self.assertLess(
                spec.autoregressive_training_flops_per_clean_token,
                spec.training_flops_per_clean_token,
            )
            for p_ar in P_AR_VALUES:
                expected = (
                    p_ar * spec.autoregressive_training_flops_per_clean_token
                    + (1.0 - p_ar) * spec.training_flops_per_clean_token
                )
                self.assertAlmostEqual(
                    average_training_flops_per_clean_token(spec, p_ar),
                    expected,
                )

    def test_curriculum_steps_stay_under_budget(self):
        for p_ar in P_AR_VALUES:
            for budget in COMPUTE_BUDGETS:
                for spec in MODEL_SPECS:
                    if not curriculum_is_feasible(budget, spec, p_ar):
                        continue
                    steps = curriculum_steps_for(budget, spec, p_ar)
                    self.assertLessEqual(
                        curriculum_realized_flops(steps, spec, p_ar),
                        budget,
                    )
                    self.assertGreater(
                        curriculum_realized_flops(steps + 1, spec, p_ar),
                        budget,
                    )
                    ar_steps, bd_steps = split_phase_steps(steps, p_ar)
                    self.assertEqual(ar_steps + bd_steps, steps)
                    self.assertGreater(ar_steps, 0)
                    self.assertGreater(bd_steps, 0)

    def test_curriculum_reuses_one_prior_lr_per_point(self):
        args = SimpleNamespace(budget=None, size=None, p_ar=None)
        runs = curriculum_run_grid(args)
        self.assertEqual(len(runs), 153)
        keys = {
            (run["p_ar"], run["budget"], run["spec"].label)
            for run in runs
        }
        self.assertEqual(len(keys), len(runs))
        for run in runs:
            self.assertEqual(
                run["lr"],
                baseline_learning_rate(run["budget"], run["spec"]),
            )


class MaskTests(unittest.TestCase):
    def test_mask_structure(self):
        length, block = 8, 2
        mask = make_dual_stream_mask(length, block)[0, 0]
        self.assertTrue(bool(mask[0, 0] and mask[0, 1]))
        self.assertFalse(bool(mask[0, 2]))
        self.assertFalse(bool(mask[0, length]))
        self.assertTrue(bool(mask[4, length]))
        self.assertTrue(bool(mask[4, length + 3]))
        self.assertFalse(bool(mask[4, length + 4]))
        self.assertFalse(bool(mask[length:, :length].any()))


class ModelTests(unittest.TestCase):
    def test_actual_parameter_count(self):
        for spec in MODEL_SPECS:
            model = BlockDiffusionTransformer(spec)
            self.assertEqual(model.counted_parameter_count(), spec.n_params)

    def test_forward_and_backward(self):
        spec = MODEL_SPECS[0]
        model = BlockDiffusionTransformer(spec)
        clean = torch.randint(1, VOCAB_SIZE, (2, 256))
        probability = torch.full((2, 64), 0.5)
        noisy, masked, token_probability = corrupt(clean, probability)
        logits = model(noisy, clean)
        self.assertEqual(tuple(logits.shape), (2, 256, VOCAB_SIZE))
        loss = diffusion_nelbo(logits, clean, masked, token_probability)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_autoregressive_forward_and_backward(self):
        spec = MODEL_SPECS[0]
        model = BlockDiffusionTransformer(spec)
        inputs = torch.randint(1, VOCAB_SIZE, (2, 256))
        targets = torch.randint(1, VOCAB_SIZE, (2, 256))
        logits = model.forward_ar(inputs)
        self.assertEqual(tuple(logits.shape), (2, 256, VOCAB_SIZE))
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            targets.reshape(-1),
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_autoregressive_forward_is_causal(self):
        spec = MODEL_SPECS[0]
        model = BlockDiffusionTransformer(spec).eval()
        tokens_a = torch.randint(1, VOCAB_SIZE, (1, 256))
        tokens_b = tokens_a.clone()
        tokens_b[:, 128:] = torch.randint(1, VOCAB_SIZE, (1, 128))
        with torch.no_grad():
            logits_a = model.forward_ar(tokens_a)
            logits_b = model.forward_ar(tokens_b)
        self.assertTrue(torch.equal(logits_a[:, :128], logits_b[:, :128]))

    def test_no_clean_target_leakage(self):
        spec = MODEL_SPECS[0]
        model = BlockDiffusionTransformer(spec).eval()
        noisy = torch.randint(0, VOCAB_SIZE, (1, 256))
        clean_a = torch.randint(1, VOCAB_SIZE, (1, 256))
        clean_b = clean_a.clone()
        clean_b[:, 4:] = torch.randint(1, VOCAB_SIZE, (1, 252))
        with torch.no_grad():
            logits_a = model(noisy, clean_a)
            logits_b = model(noisy, clean_b)
        self.assertTrue(torch.equal(logits_a[:, :4], logits_b[:, :4]))


class OptimizerTests(unittest.TestCase):
    def test_weight_decay_can_be_selected_per_phase(self):
        model = BlockDiffusionTransformer(MODEL_SPECS[0])
        optimizer = optimizer_for(model, 1e-3, weight_decay=0.4)
        self.assertEqual(
            {group["weight_decay"] for group in optimizer.param_groups},
            {0.0, 0.4},
        )


class ScheduleTests(unittest.TestCase):
    def test_wsd_shape(self):
        total, peak = 1000, 0.009
        self.assertAlmostEqual(wsd_learning_rate(49, total, peak), peak)
        self.assertAlmostEqual(wsd_learning_rate(849, total, peak), peak)
        self.assertLess(wsd_learning_rate(900, total, peak), peak)
        self.assertEqual(wsd_learning_rate(999, total, peak), 0.0)

    def test_wsd_can_disable_restart_warmup(self):
        total, peak = 200, 0.009
        self.assertEqual(
            wsd_learning_rate(0, total, peak, warmup_fraction=0.0),
            peak,
        )
        self.assertEqual(
            wsd_learning_rate(169, total, peak, warmup_fraction=0.0),
            peak,
        )
        self.assertLess(
            wsd_learning_rate(171, total, peak, warmup_fraction=0.0),
            peak,
        )
        self.assertEqual(
            wsd_learning_rate(199, total, peak, warmup_fraction=0.0),
            0.0,
        )


class AnalysisTests(unittest.TestCase):
    def test_l1_quadratic_exact_on_clean_data(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        y = x**2 - 0.5 * x + 1.25
        coefficients, support = fit_l1_quadratic(x, y)
        self.assertTrue(np.allclose(coefficients, [1.0, -0.5, 1.25]))
        self.assertEqual(len(support), 3)

    def test_l1_quadratic_minimizes_absolute_residual(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        y = x**2
        y[4] += 3.0
        l1_coefficients, _ = fit_l1_quadratic(x, y)
        l2_coefficients = np.polyfit(x, y, 2)
        l1_error = np.abs(y - np.polyval(l1_coefficients, x)).sum()
        l2_error = np.abs(y - np.polyval(l2_coefficients, x)).sum()
        self.assertLessEqual(l1_error, l2_error + 1e-12)

    def test_loss_law_with_floor_recovers_clean_curve(self):
        compute = np.array([1e13, 3e13, 1e14, 3e14, 1e15])
        expected_floor = 1.2
        expected_amplitude = 0.7
        expected_exponent = -0.2
        losses = (
            expected_floor
            + expected_amplitude * (compute / 1e14) ** expected_exponent
        )
        law = fit_loss_law_with_floor(compute, losses)
        self.assertAlmostEqual(law["asymptote"], expected_floor, places=6)
        self.assertAlmostEqual(
            law["coefficient_at_reference"],
            expected_amplitude,
            places=6,
        )
        self.assertAlmostEqual(law["exponent"], expected_exponent, places=6)
        self.assertAlmostEqual(law["r_squared"], 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
