# Copyright 2024, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: pyzag
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Tests for Gauss-Newton preconditioning of Pyro SVI.

The model here is a miniature of what
:class:`pyzag.stochastic.HierarchicalStatisticalModel` produces -- Normal/HalfNormal
hyper-priors outside a plate, one latent block per plate member inside it, and a
shared noise scale -- but with a closed-form forward map instead of a recursive
solve. That is the structure the estimator exploits, and it keeps the tests fast.

The forward map is deliberately **nonlinear** in the latents, so an exactness
result cannot be an artifact of a linear Jacobian.
"""

import math
import unittest
import warnings

import torch

import pyro
import pyro.distributions as dist
from pyro.poutine.subsample_messenger import _Subsample

from pyzag import preconditioning, stochastic

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)

NB, NT, K = 5, 4, 3


def _model(obs=None, weights=None):
    """Hierarchical model: 2 hyper sites, a per-member block, a shared noise."""
    loc = pyro.sample(
        "p_loc", dist.Normal(torch.zeros(K), 2 * torch.ones(K)).to_event(1)
    )
    scale = pyro.sample("p_scale", dist.HalfNormal(torch.ones(K)).to_event(1))
    eps = pyro.sample("eps", dist.HalfNormal(torch.tensor(1.0)))
    if weights is None:
        weights = 1.0
    with pyro.plate("samples", NB), pyro.poutine.scale_messenger.ScaleMessenger(
        scale=weights
    ):
        theta = pyro.sample("p", dist.Normal(loc, scale).to_event(1))
        pred = 1.7 * (theta**2).sum(-1).expand(NT, NB).unsqueeze(-1)
        with pyro.plate("time", NT):
            pyro.sample("obs", dist.Normal(pred, eps).to_event(1), obs=obs)


class _SVIFixture(unittest.TestCase):
    """Shared setup: a materialized param store and a curvature engine over it."""

    def setUp(self):
        pyro.clear_param_store()
        pyro.set_rng_seed(0)
        self.obs = torch.randn(NT, NB, 1)
        self.guide = pyro.infer.autoguide.AutoDelta(_model)
        # Parameters are created lazily, so take one throwaway step to materialize.
        pyro.infer.SVI(
            _model,
            self.guide,
            pyro.optim.ClippedAdam({"lr": 1e-12}),
            loss=pyro.infer.Trace_ELBO(),
        ).step(self.obs)
        store = pyro.get_param_store()
        self.names = sorted(store.keys())
        self.leaves = [store._params[n] for n in self.names]

    def residual(self):
        return stochastic.gaussian_map_residual(_model, self.guide, self.obs)

    def curvature(self, **kwargs):
        return preconditioning.GaussNewtonCurvature(self.leaves, mode="diag", **kwargs)

    def brute_diag(self, curv, flat):
        """diag(J^T J) the honest way: one one-hot cotangent per residual row."""
        out = torch.zeros(curv.nparam)
        for i in range(flat.numel()):
            e = torch.zeros_like(flat)
            e[i] = 1.0
            out += curv.vjp(flat, e) ** 2
        return out

    def slice_for(self, name):
        off = 0
        for n, leaf in zip(self.names, self.leaves):
            if n == name:
                return slice(off, off + leaf.numel())
            off += leaf.numel()
        raise KeyError(name)


class TestGaussianMapResidual(_SVIFixture):
    def test_reproduces_pyro_log_prob(self):
        """0.5||r||^2 plus the log-normalizers must equal pyro's own log density."""
        gres = self.residual()
        guide_trace = pyro.poutine.trace(self.guide).get_trace(self.obs)
        trace = pyro.poutine.trace(
            pyro.poutine.replay(_model, trace=guide_trace)
        ).get_trace(self.obs)
        trace.compute_log_prob()

        expected, got = 0.0, 0.0
        for name, node in trace.nodes.items():
            if node["type"] != "sample" or isinstance(node["fn"], _Subsample):
                continue
            expected += float(node["log_prob_sum"])
            base = node["fn"]
            while isinstance(base, dist.Independent):
                base = base.base_dist
            r = gres.flat[gres.blocks[name]]
            const = (
                0.5 * math.log(2 / math.pi)
                if isinstance(base, dist.HalfNormal)
                else -0.5 * math.log(2 * math.pi)
            )
            logscale = torch.broadcast_to(base.scale, node["value"].shape).log().sum()
            got += float(-0.5 * (r**2).sum() - logscale + const * r.numel())
        self.assertAlmostEqual(expected, got, places=9)

    def test_records_plate_membership(self):
        gres = self.residual()
        self.assertEqual(gres.plates["p"], ("samples",))
        self.assertEqual(gres.plates["p_loc"], ())
        self.assertEqual(gres.plates["p_scale"], ())
        self.assertEqual(gres.plates["eps"], ())
        self.assertEqual(gres.obs_name, "obs")

    def test_likelihood_alone_misses_the_hyper_parameters(self):
        """Why the prior rows are not optional: the likelihood does not reach the
        hyper-parameters at all, so a likelihood-only curvature would leave them
        silently unpreconditioned."""
        gres = self.residual()
        curv = self.curvature()
        r_obs = gres.flat[gres.blocks[gres.obs_name]]
        grads = torch.autograd.grad(
            r_obs.sum(), self.leaves, retain_graph=True, allow_unused=True
        )
        for name, g in zip(self.names, grads):
            if name.endswith("p_loc") or name.endswith("p_scale"):
                # Either detached from the graph entirely or exactly zero -- which
                # of the two torch reports is an incidental graph-connectivity
                # detail; both mean "no curvature from the data".
                self.assertTrue(
                    g is None or bool((g == 0).all()),
                    f"{name} unexpectedly reached by the likelihood",
                )
        # ...whereas the full residual gives every parameter curvature.
        diag, _ = stochastic.hierarchical_gn_diagonal(curv, gres, self.names, nsub=NT)
        self.assertTrue(bool((diag > 0).all()))

    def test_non_gaussian_site_raises(self):
        def bad_model(obs=None):
            rate = pyro.sample("rate", dist.Gamma(torch.tensor(2.0), torch.tensor(2.0)))
            with pyro.plate("samples", NB):
                with pyro.plate("time", NT):
                    pyro.sample(
                        "obs",
                        dist.Normal(rate.expand(NT, NB, 1), 1.0).to_event(1),
                        obs=obs,
                    )

        pyro.clear_param_store()
        guide = pyro.infer.autoguide.AutoDelta(bad_model)
        with self.assertRaises(TypeError):
            stochastic.gaussian_map_residual(bad_model, guide, self.obs)

    def test_poutine_scale_is_folded_in(self):
        """A per-member likelihood weight must reach the residual as sqrt(w)."""
        pyro.clear_param_store()
        pyro.set_rng_seed(0)
        w = torch.linspace(0.25, 4.0, NB)
        guide = pyro.infer.autoguide.AutoDelta(_model)
        plain = stochastic.gaussian_map_residual(_model, guide, self.obs)
        scaled = stochastic.gaussian_map_residual(_model, guide, self.obs, weights=w)
        a = plain.flat[plain.blocks["obs"]].reshape(NT, NB, 1)
        b = scaled.flat[scaled.blocks["obs"]].reshape(NT, NB, 1)
        self.assertTrue(torch.allclose(b, a * w.sqrt().reshape(1, NB, 1)))


class TestHierarchicalDiagonal(_SVIFixture):
    def test_exact_at_nbatch_fewer_sweeps(self):
        """The headline: plate-slice cotangents are *exact*, not approximate, and
        cost one sweep per time step rather than one per (time, member) row."""
        gres = self.residual()
        curv = self.curvature()
        brute = self.brute_diag(curv, gres.flat)
        diag, sweeps = stochastic.hierarchical_gn_diagonal(
            curv, gres, self.names, nsub=NT
        )
        self.assertTrue(torch.allclose(diag, brute, rtol=1e-9, atol=1e-12))
        self.assertEqual(sweeps, NT)
        self.assertLess(sweeps, gres.flat.numel())

    def test_shared_noise_scale_is_analytic(self):
        """eps is shared across the plate, so grouped cotangents would double count
        it. The analytic column norm is what makes it right -- check both halves."""
        gres = self.residual()
        curv = self.curvature()
        brute = self.brute_diag(curv, gres.flat)
        sl = self.slice_for("AutoDelta.eps")

        diag, _ = stochastic.hierarchical_gn_diagonal(curv, gres, self.names, nsub=NT)
        self.assertTrue(torch.allclose(diag[sl], brute[sl], rtol=1e-9))

        # The same sweeps, but naively letting eps ride along, is wrong.
        axis = gres.obs_plate_axis("samples")
        cots, scale = stochastic._obs_group_cotangents(gres, axis, NT, None)
        naive = torch.zeros(curv.nparam)
        for cot in cots:
            full = torch.zeros_like(gres.flat)
            full[gres.blocks["obs"]] = cot.reshape(-1)
            naive += curv.vjp(gres.flat, full) ** 2
        naive *= scale
        self.assertFalse(torch.allclose(naive[sl], brute[sl], rtol=1e-3))

    def test_naive_row_subsampling_starves_members(self):
        """The failure mode the structure-aware estimator exists to avoid."""
        gres = self.residual()
        curv = self.curvature(nsub=NT, generator=torch.Generator().manual_seed(0))
        curv.refresh(gres.flat)
        starved = int((curv.H <= 0).sum())
        self.assertGreater(starved, 0)
        # ...while the structure-aware estimator, at the same sweep budget, does not.
        curv2 = self.curvature()
        diag, sweeps = stochastic.hierarchical_gn_diagonal(
            curv2, gres, self.names, nsub=NT
        )
        self.assertEqual(sweeps, NT)
        self.assertEqual(int((diag <= 0).sum()), 0)

    def test_subsampled_time_is_unbiased_in_scale(self):
        """Fewer plate-slices still covers every member, just with fewer rows."""
        gres = self.residual()
        curv = self.curvature()
        diag, sweeps = stochastic.hierarchical_gn_diagonal(
            curv, gres, self.names, nsub=2, generator=torch.Generator().manual_seed(0)
        )
        self.assertEqual(sweeps, 2)
        self.assertEqual(int((diag <= 0).sum()), 0)

    def test_full_mode_is_rejected(self):
        gres = self.residual()
        curv = self.curvature()
        curv.mode = "full"
        with self.assertRaises(ValueError):
            stochastic.hierarchical_gn_diagonal(curv, gres, self.names, nsub=NT)

    def test_validate_warns_on_unmodelled_shared_parameter(self):
        """A shared parameter that drives the likelihood directly is outside the
        estimator's assumptions and must be reported, not silently unpreconditioned."""

        def shared_model(obs=None):
            gain = pyro.sample(
                "gain", dist.Normal(torch.tensor(1.0), torch.tensor(1.0))
            )
            eps = pyro.sample("eps", dist.HalfNormal(torch.tensor(1.0)))
            with pyro.plate("samples", NB):
                theta = pyro.sample(
                    "p", dist.Normal(torch.zeros(K), torch.ones(K)).to_event(1)
                )
                pred = gain * (theta**2).sum(-1).expand(NT, NB).unsqueeze(-1)
                with pyro.plate("time", NT):
                    pyro.sample("obs", dist.Normal(pred, eps).to_event(1), obs=obs)

        pyro.clear_param_store()
        pyro.set_rng_seed(0)
        guide = pyro.infer.autoguide.AutoDelta(shared_model)
        pyro.infer.SVI(
            shared_model,
            guide,
            pyro.optim.ClippedAdam({"lr": 1e-12}),
            loss=pyro.infer.Trace_ELBO(),
        ).step(self.obs)
        store = pyro.get_param_store()
        names = sorted(store.keys())
        curv = preconditioning.GaussNewtonCurvature(
            [store._params[n] for n in names], mode="diag"
        )
        gres = stochastic.gaussian_map_residual(shared_model, guide, self.obs)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stochastic.hierarchical_gn_diagonal(
                curv, gres, names, nsub=NT, validate=True
            )
        self.assertTrue(any("shared parameter" in str(w.message) for w in caught))


class TestPyroGaussNewtonOptim(_SVIFixture):
    def make(self, **kwargs):
        opt = stochastic.PyroGaussNewtonOptim(
            torch.optim.SGD,
            {"lr": 1e-2, "momentum": 0.9},
            lambda: stochastic.gaussian_map_residual(_model, self.guide, self.obs),
            nsub=NT,
            **kwargs,
        )
        return opt, stochastic.PreconditionedSVI(
            _model, self.guide, opt, loss=pyro.infer.Trace_ELBO()
        )

    def test_adam_family_base_optimizer_is_rejected(self):
        """SVI's usual optimizer is exactly the one that cannot work here, so the
        rejection must fire through this adapter too -- and lazily, since the
        inner optimizer does not exist until the first step."""
        opt = stochastic.PyroGaussNewtonOptim(
            torch.optim.Adam,
            {"lr": 1e-2},
            lambda: stochastic.gaussian_map_residual(_model, self.guide, self.obs),
            nsub=NT,
        )
        svi = stochastic.PreconditionedSVI(
            _model, self.guide, opt, loss=pyro.infer.Trace_ELBO()
        )
        with self.assertRaises(TypeError):
            svi.step(self.obs)

    def test_svi_accepts_it_and_builds_one_optimizer(self):
        opt, svi = self.make()
        svi.step(self.obs)
        self.assertIsNotNone(opt.inner)
        self.assertEqual(len(opt.inner.param_groups), 1)
        self.assertEqual(len(opt.inner.param_groups[0]["params"]), len(self.names))
        # The base class's per-parameter machinery must stay unused.
        self.assertEqual(len(opt.optim_objs), 0)

    def test_parameters_are_name_sorted(self):
        opt, svi = self.make()
        svi.step(self.obs)
        self.assertEqual(opt._names, sorted(opt._names))
        self.assertEqual(set(opt._names), set(self.names))

    def test_preconditioned_svi_records_the_pre_update_loss(self):
        opt, svi = self.make()
        loss = svi.step(self.obs)
        self.assertAlmostEqual(opt.recorded_loss, loss, places=10)

    def test_plain_svi_warns_and_falls_back_to_compute_once(self):
        """Without the loss the gain-ratio trigger cannot fire; that must be a
        warning and a safe degradation, not a silent one."""
        opt, _ = self.make(rho=0.25)
        svi = pyro.infer.SVI(_model, self.guide, opt, loss=pyro.infer.Trace_ELBO())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(5):
                svi.step(self.obs)
        self.assertTrue(any("no loss was recorded" in str(w.message) for w in caught))
        self.assertEqual(opt.curvature.n_refreshes, 1)

    def test_fixed_preconditioner_refreshes_once(self):
        opt, svi = self.make(rho=None)
        for _ in range(8):
            svi.step(self.obs)
        self.assertEqual(opt.curvature.n_refreshes, 1)
        self.assertEqual(opt.curvature.refresh_steps, [0])
        self.assertEqual(opt.curvature.n_refresh_sweeps, NT)

    def test_gain_ratio_can_trigger_refreshes(self):
        """min_refresh_interval=0 lifts the floor, so an always-stale trigger
        refreshes on every step."""
        refreshed = []
        opt, svi = self.make(
            rho=1e12,
            min_refresh_interval=0,
            on_refresh=lambda **kw: refreshed.append(kw),
        )
        for _ in range(5):
            svi.step(self.obs)
        self.assertEqual(opt.curvature.n_refreshes, 5)
        self.assertEqual([e["step"] for e in refreshed], opt.curvature.refresh_steps)
        self.assertEqual(refreshed[0]["step"], 0)

    def test_min_refresh_interval_spaces_refreshes(self):
        """The counter resets to 0 on refresh, so an interval of k enforces k
        reused steps between refreshes: with the default 1, every other step."""
        opt, svi = self.make(rho=1e12, min_refresh_interval=1)
        for _ in range(6):
            svi.step(self.obs)
        self.assertEqual(opt.curvature.refresh_steps, [0, 2, 4])

    def test_it_actually_optimizes(self):
        opt, svi = self.make(rho=0.25)
        first = svi.step(self.obs)
        for _ in range(40):
            last = svi.step(self.obs)
        self.assertLess(last, first)

    def test_state_roundtrip(self):
        opt, svi = self.make()
        for _ in range(3):
            svi.step(self.obs)
        state = opt.get_state()
        self.assertEqual(state["names"], opt._names)

        # A fresh optimizer that loaded the state must resume from the saved
        # momentum buffer, not from zero -- that is what proves the load worked.
        opt2, svi2 = self.make()
        opt2.set_state(state)
        svi2.step(self.obs)
        loaded = opt2.inner.state_dict()["state"][0]["momentum_buffer"]

        opt3, svi3 = self.make()
        svi3.step(self.obs)
        fresh = opt3.inner.state_dict()["state"][0]["momentum_buffer"]
        self.assertFalse(torch.allclose(loaded, fresh))

    def test_nonfinite_gradient_skips_update_and_forces_refresh(self):
        opt, svi = self.make(rho=None)
        svi.step(self.obs)
        before = [p.detach().clone() for p in opt.curvature.parameters]
        # Poison the cached curvature so the preconditioned gradient blows up.
        opt.curvature.set_H(torch.full_like(opt.curvature.H, float("nan")))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            svi.step(self.obs)
        self.assertTrue(any("non-finite" in str(w.message) for w in caught))
        for a, b in zip(before, opt.curvature.parameters):
            self.assertTrue(torch.allclose(a, b))
        self.assertTrue(opt.curvature._force_refresh)

    def test_per_parameter_optim_args_rejected(self):
        with self.assertRaises(ValueError):
            stochastic.PyroGaussNewtonOptim(
                torch.optim.Adam, lambda _name: {"lr": 1e-2}, lambda: None
            )


if __name__ == "__main__":
    unittest.main()
