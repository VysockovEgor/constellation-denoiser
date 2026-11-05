"""Unit tests for E039-E043.   PYTHONPATH=src python -m unittest discover -s tests -v"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from constellation_denoiser import cascade_e040, model_zoo, rm_group_convention as rg, triplets


class RMConventionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        self.x = (rng.normal(size=200) + 1j * rng.normal(size=200)) / np.sqrt(2)

    def test_counts_match_literal_execution(self):
        for M in (2, 3):
            terms = triplets.triplet_terms(M, "T3")
            coef = np.random.default_rng(M).normal(size=len(terms)) * 0.01 + 0j
            counter = rg.Counter()
            ks = range(40, 140)
            direct = [rg.direct_eval(self.x, terms, coef, k, counter) for k in ks]
            self.assertEqual(counter.count / len(ks), 3 * len(terms))                 # 12 RM per term
            self.assertEqual(rg.rm_repo(terms), 4 * counter.count / len(ks))
            counter = rg.Counter()
            buffered = rg.buffered_eval(self.x, terms, coef, ks, counter)
            self.assertEqual(4 * counter.count / len(ks), rg.rm_buffered(terms))
            self.assertLess(np.abs(np.array(direct) - np.array(buffered)).max(), 1e-12)
            ref = triplets.predict(self.x, coef, terms, 0, False, 4096)[:, 0]         # centres 15 .. 184
            self.assertLess(np.abs(np.array(direct) - ref[np.array(list(ks)) - 15]).max(), 1e-12)

    def test_arxiv_formula(self):
        self.assertEqual(rg.rm_arxiv_2409_20023(2), 4 * (18 + 19 + 4))


class CascadeTests(unittest.TestCase):
    def test_symmetries_of_c(self):
        coef = cascade_e040.TripletCoefficients(4, 16)
        with torch.no_grad():
            coef.base_re.normal_()
            coef.base_im.normal_()
        merged = coef().detach().numpy()
        terms = cascade_e040.stage_terms(4, 16)
        c = {t: w / (1 if t[0] == t[1] else 2) for t, w in zip(terms, merged)}
        full = dict(c)
        full.update({(n, m): v for (m, n), v in c.items()})
        for (m, n), v in full.items():
            self.assertAlmostEqual(abs(full[(-m, -n)] - v), 0, places=6)
            self.assertAlmostEqual(abs(full[(-m, n)] + np.conj(v)), 0, places=6)
            if m == 0 or n == 0:
                self.assertAlmostEqual(v.real, 0, places=6)

    def test_identity_at_start_rm_and_gradients(self):
        stages = [{"type": "T", "M": 4, "K_th": 4, "L": 7}, {"type": "K", "R": 3, "L": 1},
                  {"type": "KT", "M": 2, "K_th": 4, "R": 3, "L": 7}, {"type": "skip"}]
        W = cascade_e040.window_for(stages)
        spec = {"model": "cascade", "window": W, "stages": stages}
        model = model_zoo.build(spec, seed=3)
        x = torch.randn(16, W, 2) * 0.4
        with torch.no_grad():
            centre = torch.complex(x[:, W // 2, 0], x[:, W // 2, 1])
            self.assertLess(float((model(x) - centre).abs().max()), 1e-6)
        expected = 12 * len(cascade_e040.stage_terms(4, 4)) + 28 + (2 + 6) + 12 * len(cascade_e040.stage_terms(2, 4)) + 8 + 28
        self.assertEqual(model_zoo.cost(spec).macs, expected)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        model(x).abs().square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_one_stage_matches_closed_form_features(self):
        stages = [{"type": "T", "M": 2, "K_th": 4, "L": 1}]
        W = cascade_e040.window_for(stages)
        model = cascade_e040.Cascade(W, stages)
        with torch.no_grad():
            model.stages[0].coef.base_im.fill_(0.01)
        rng = np.random.default_rng(1)
        rx = (rng.normal(size=60) + 1j * rng.normal(size=60)) / np.sqrt(2)
        terms = cascade_e040.stage_terms(2, 4)
        weights = model.stages[0].coef().detach().numpy().astype(complex)
        ref = triplets.predict(rx, weights, terms, 0, False, 4096)[:, 0]
        windows = np.lib.stride_tricks.sliding_window_view(rx, W)
        x = torch.tensor(np.stack([windows.real, windows.imag], -1), dtype=torch.float32)
        with torch.no_grad():
            got = model(x).numpy()
        start = 15 - W // 2                                   # predict starts at centre 15
        self.assertLess(np.abs(got[start: start + 10] - ref[:10]).max(), 1e-5)

    def test_shared(self):
        stages = [{"type": "T", "M": 2, "K_th": 4, "L": 7}] * 3
        W = cascade_e040.window_for(stages)
        a = cascade_e040.Cascade(W, stages, shared=True)
        b = cascade_e040.Cascade(W, stages, shared=False)
        n = lambda m: sum(p.numel() for p in m.parameters())
        self.assertLess(n(a), n(b))


if __name__ == "__main__":
    unittest.main()


class CascadeInitTests(unittest.TestCase):
    def test_ridge_init_reproduces_closed_form_stage(self):
        import tempfile

        stages = [{"type": "T", "M": 2, "K_th": 4, "L": 1}]
        terms = cascade_e040.stage_terms(2, 4)
        rng = np.random.default_rng(4)
        sym = cascade_e040.TripletCoefficients(2, 4)
        with torch.no_grad():
            sym.base_re.normal_(0, 0.01)
            sym.base_im.normal_(0, 0.01)
        weights = sym().detach().numpy().astype(complex)          # weights that satisfy the symmetries
        W = cascade_e040.window_for(stages)
        for free in (False, True):
            with tempfile.NamedTemporaryFile(suffix=".npz") as handle:
                np.savez(handle.name, weights=weights)
                model = cascade_e040.Cascade(W, stages, free=free, init_first=handle.name)
            got = model.stages[0].coef().detach().numpy()
            self.assertLess(np.abs(got - weights).max(), 1e-6)
        rx = (rng.normal(size=60) + 1j * rng.normal(size=60)) / np.sqrt(2)
        ref = triplets.predict(rx, weights, terms, 0, False, 4096)[:, 0]
        windows = np.lib.stride_tricks.sliding_window_view(rx, W)
        x = torch.tensor(np.stack([windows.real, windows.imag], -1), dtype=torch.float32)
        with torch.no_grad():
            out = model(x).numpy()
        start = 15 - W // 2
        self.assertLess(np.abs(out[start: start + 10] - ref[:10]).max(), 1e-5)


class PhysicalInitTests(unittest.TestCase):
    def test_physical_taps_and_model(self):
        from constellation_denoiser import e049_physical as ep

        self.assertAlmostEqual(ep.c_step(1), -21.504, places=2)
        taps, energy = ep.physical_taps(ep.c_step(8), 31)
        self.assertGreater(energy, 0.9)
        self.assertLess(np.abs(taps - taps[::-1]).max(), 1e-12)               # h[n] = h[-n]
        c, r2 = ep.fit_c(taps * (0.8 + 0.3j), np.arange(-5, 5, 0.02))
        self.assertAlmostEqual(c, ep.c_step(8), delta=0.03)
        spec = {"model": "dbp_phys", "window": 31, "fir_len": 31, "n_alpha": 7, "sym": True, "n_steps": 8, "c_sign": 1.0,
                "alpha_centre": -0.0019}
        model = model_zoo.build(spec, seed=1)
        full = np.r_[model.steps[0].fir_re.detach().numpy()[1:][::-1], model.steps[0].fir_re.detach().numpy()]
        self.assertLess(np.abs(full - taps.real).max(), 1e-6)
        self.assertEqual(model_zoo.cost(spec).macs, 592)
        self.assertAlmostEqual(float(model.steps[0].alpha[0]), -0.0019 / 8, places=8)
