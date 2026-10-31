"""Unit tests for the E024-E033 modules.   PYTHONPATH=src python -m unittest discover -s tests -v"""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from constellation_denoiser import model_zoo, models_e024, rm_audit, triplets
from constellation_denoiser.core import CONSTELLATION
from constellation_denoiser.losses_new import LOSSES, SymbolLoss, WithLoss


def batch(window=31, n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, window, 2, generator=g) * 0.4


class ActivationTests(unittest.TestCase):
    def test_rm_and_gradients(self):
        x = batch()
        for act in models_e024.ACTIVATIONS:
            spec = {"model": "act_mlp", "window": 31, "hidden": 48, "activation": act, "c_fixed": -0.1}
            model = model_zoo.build(spec, seed=17)
            y = model(x)
            self.assertTrue(torch.is_complex(y), act)
            y.abs().square().mean().backward()
            grads = [p.grad for p in model.parameters() if p.requires_grad]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads), act)
            expected = 4 * 31 * 48 + 4 * 48 + models_e024.ACT_RM[act] * 48
            self.assertEqual(model_zoo.cost(spec).macs, expected, act)
            self.assertEqual(rm_audit.formula(spec)[0], expected, act)
        kerr = {"model": "act_mlp", "window": 31, "hidden": 48, "activation": "kerr_c_train"}
        self.assertEqual(model_zoo.cost(kerr).macs, 6480)      # = complex_one_hidden h48

    def test_kerr_c_train_equals_one_hidden(self):
        x = batch()
        a = model_zoo.build({"model": "act_mlp", "window": 31, "hidden": 16, "activation": "kerr_c_train"}, seed=17)
        b = model_zoo.build({"model": "complex_one_hidden", "window": 31, "hidden": 16}, seed=17)
        with torch.no_grad():
            self.assertLess(float((a(x) - b(x)).abs().max()), 1e-6)

    def test_fixed_c_is_not_trainable(self):
        model = model_zoo.build({"model": "act_mlp", "window": 31, "hidden": 8, "activation": "kerr_c_fixed",
                                 "c_fixed": -0.12}, seed=1)
        names = [n for n, _ in model.named_parameters()]
        self.assertFalse(any(n.endswith(".c") for n in names))
        self.assertAlmostEqual(float(model.act1.c), -0.12, places=6)

    def test_taylor_gets_c_mult(self):
        from constellation_denoiser.lr_schedules import LRConfig, build_optimizer

        model = model_zoo.build({"model": "act_mlp", "window": 31, "hidden": 8, "activation": "kerr_taylor1"}, seed=1)
        opt, base = build_optimizer(model, LRConfig(peak_lr=1e-3, c_mult=30))
        self.assertEqual(len(base), 2)
        self.assertAlmostEqual(base[1], 3e-2)


class RealAndDbpTests(unittest.TestCase):
    def test_real_mlp_matches_real_tanh(self):
        x = batch()
        a = model_zoo.build({"model": "real_mlp", "window": 31, "hidden": 96, "layers": 2, "activation": "tanh"}, seed=17)
        b = model_zoo.build({"model": "real_tanh", "window": 31, "hidden": 96}, seed=17)
        with torch.no_grad():
            self.assertLess(float((a(x) - b(x)).abs().max()), 1e-6)
        self.assertEqual(model_zoo.cost({"model": "real_mlp", "window": 31, "hidden": 96, "layers": 2}).macs, 15360)
        self.assertEqual(model_zoo.cost({"model": "real_mlp", "window": 31, "hidden": 100, "layers": 1}).macs, 6400)

    def test_dbp_variants(self):
        x = batch()
        base = {"window": 31, "fir_len": 31, "n_alpha": 7, "sym": True, "n_steps": 8}
        indep = model_zoo.cost(dict(base, model="phys_dbp")).macs
        self.assertEqual(indep, 592)
        for kind, extra in (("dbp_shared", {}), ("dbp_noise", {"init_noise": 1e-3}), ("dbp_mlp", {"mlp_hidden": 8})):
            spec = dict(base, model=kind, **extra)
            model = model_zoo.build(spec, seed=3)
            y = model(x)
            y.abs().square().mean().backward()
            self.assertTrue(torch.is_complex(y))
            rm = model_zoo.cost(spec).macs
            self.assertEqual(rm, indep + (8 * (4 * 31 + 11) if kind == "dbp_mlp" else 0), kind)
        shared = model_zoo.build(dict(base, model="dbp_shared"), seed=3)
        indep_model = model_zoo.build(dict(base, model="phys_dbp"), seed=3)
        self.assertLess(model_zoo.n_params(shared) * 7, model_zoo.n_params(indep_model))
        with torch.no_grad():                    # identity at start: equals the received centre symbol
            centre = torch.complex(x[:, 15, 0], x[:, 15, 1])
            self.assertLess(float((shared(x) - centre).abs().max()), 1e-6)
        names = [n for n, _ in model_zoo.build(dict(base, model="dbp_noise", init_noise=1e-3), seed=3).named_parameters()]
        self.assertTrue(any(n.startswith("steps.7.") for n in names))   # layer-wise phase prefixes still apply

    def test_rm_formulas_known_points(self):
        cases = [({"model": "complex_one_hidden", "window": 31, "hidden": 48}, 6480),
                 ({"model": "complex_both", "window": 31, "hidden": 48}, 16032),
                 ({"model": "complex_one_hidden", "window": 31, "hidden": 119}, 16065),
                 ({"model": "real_tanh", "window": 31, "hidden": 96}, 15360),
                 ({"model": "phys_kerr", "window": 31, "fir_len": 7, "n_alpha": 31, "sym": True}, 38)]
        for spec, expected in cases:
            self.assertEqual(rm_audit.formula(spec)[1], expected, spec)
            self.assertEqual(model_zoo.cost(spec).macs_merged, expected, spec)
        self.assertEqual(rm_audit.formula({"model": "complex_a", "window": 31, "hidden": 48})[0], 15696)


class LossTests(unittest.TestCase):
    def test_argmax_logits_is_nearest_point(self):
        rng = np.random.default_rng(0)
        y = torch.tensor(rng.normal(size=500) * 0.7 + 1j * rng.normal(size=500) * 0.7, dtype=torch.complex64)
        nearest = torch.tensor(np.abs(y.numpy()[:, None] - CONSTELLATION[None, :]).argmin(axis=1))
        for beta in (0.5, 50.0, 1e4):
            loss = SymbolLoss("ce_dist", beta)
            self.assertTrue(torch.equal(loss.logits(y).argmax(dim=1), nearest))

    def test_losses_finite_and_trainable(self):
        model = model_zoo.build({"model": "complex_one_hidden", "window": 31, "hidden": 8}, seed=2)
        x = batch()
        target = torch.tensor(np.stack([CONSTELLATION.real, CONSTELLATION.imag], 1)[np.arange(64) % 16], dtype=torch.float32)
        for name in LOSSES:
            wrapped = WithLoss(model, SymbolLoss(name, 40.0))
            value = wrapped.training_loss(wrapped(x), target)
            value.backward()
            self.assertTrue(math.isfinite(float(value)), name)
            if name != "mse":
                self.assertIsNotNone(wrapped.loss.log_beta.grad, name)
        real = model_zoo.build({"model": "real_mlp", "window": 31, "hidden": 8, "layers": 1}, seed=2)
        self.assertTrue(math.isfinite(float(WithLoss(real, SymbolLoss("bit_bce", 40.0)).training_loss(real(x), target))))

    def test_bit_bce_small_for_perfect_estimate(self):
        target = torch.tensor(np.stack([CONSTELLATION.real, CONSTELLATION.imag], 1), dtype=torch.float32)
        y = torch.complex(target[:, 0], target[:, 1])
        self.assertLess(float(SymbolLoss("bit_bce", 1e3)(y, target)), 1e-3)
        self.assertLess(float(SymbolLoss("ce_dist", 1e3)(y, target)), 1e-3)


class TripletTests(unittest.TestCase):
    def test_least_squares_recovers_known_coefficients(self):
        self.assertLess(triplets.synthetic_check(seed=1, n=20000, M=3), 1e-8)

    def test_terms_and_rm(self):
        t1 = triplets.triplet_terms(3, "T1")
        self.assertEqual(len(t1), 7)                     # (0, n) for n = -3..3 after merging (m, n) ~ (n, m)
        self.assertTrue(all(m == 0 or n == 0 for m, n in t1))
        t3 = triplets.triplet_terms(3, "T3")
        self.assertTrue(all(abs(m) <= 3 and abs(n) <= 3 and abs(m + n) <= 3 and m <= n for m, n in t3))
        rm = triplets.rm_counts(3, "T3", 7)
        self.assertEqual(rm["rm_sym"], 12 * len(t3) + 28)
        self.assertGreater(rm["rm_naive"], rm["rm_sym"])


class TripletNetTests(unittest.TestCase):
    def test_torch_module_matches_closed_form_prediction(self):
        import tempfile

        from constellation_denoiser.triplets import triplet_terms
        from constellation_denoiser.triplets_ext import TripletNet, triplet_net_macs

        rng = np.random.default_rng(3)
        M, fir, W = 3, 7, 31
        p = len(triplet_terms(M, "T3")) + fir
        coef = (rng.normal(size=p) + 1j * rng.normal(size=p)) * 0.01
        rx = (rng.normal(size=400) + 1j * rng.normal(size=400)) / np.sqrt(2)
        with tempfile.NamedTemporaryFile(suffix=".npz") as handle:
            np.savez(handle.name, coef=coef)
            net = TripletNet(W, M, fir, 0, handle.name)
        expected = triplets.predict(rx, coef, triplet_terms(M, "T3"), fir, False, 4096)[:, 0]   # centres 15 .. 384
        windows = np.lib.stride_tricks.sliding_window_view(rx, W)
        x = torch.tensor(np.stack([windows.real, windows.imag], -1), dtype=torch.float32)
        with torch.no_grad():
            got = net(x).numpy()
        self.assertLess(np.abs(got - expected).max(), 1e-4)
        self.assertEqual(triplet_net_macs(W, M, fir, 8), 12 * len(triplet_terms(M, "T3")) + 28 + 8 * (4 * W + 11))


class SelectionTests(unittest.TestCase):
    def test_pareto_front_and_thinning(self):
        from constellation_denoiser.grids_e024 import pareto_front, thin_by_log

        pts = [("a", 100, 10.0), ("b", 200, 9.0), ("c", 300, 20.0), ("d", 50, 5.0), ("e", 1000, 30.0)]
        self.assertEqual([p[0] for p in pareto_front(pts)], ["d", "a", "c", "e"])
        self.assertEqual(len(thin_by_log(pareto_front(pts), 2)), 2)

    def test_sobol_configs_are_valid(self):
        from constellation_denoiser.grids import load_config
        from constellation_denoiser.grids_e024 import e025_configs
        from constellation_denoiser.lr_schedules import LRConfig

        _, config = load_config("E025")
        for key, pair in config["pairs"].items():
            items = e025_configs(key, pair, config)
            n = pair.get("n_sobol", config["n_sobol"]) + 2
            self.assertEqual(len(items), n, key)
            self.assertEqual(len({i["id"] for i in items}), n)
            for item in items:
                cfg = LRConfig.from_dict(item["lr"])
                self.assertTrue(3e-4 * 0.99 <= cfg.peak_lr <= 1e-2 * 1.01 or item["kind"] != "sobol")
                if cfg.weight_decay > 0:
                    self.assertEqual(cfg.optimizer, "adamw")
            self.assertEqual(items == e025_configs(key, pair, config), True)   # deterministic

    def test_jobs_formula(self):
        from constellation_denoiser.gate import jobs_now

        jobs, info = jobs_now(0)
        self.assertGreaterEqual(jobs, 1)
        self.assertLessEqual(jobs, 40)


if __name__ == "__main__":
    unittest.main()
