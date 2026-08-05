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

"""Test Gauss-Newton gradient preconditioning.

Uses a plain linear least-squares residual ``r(theta) = A theta - b`` -- the
preconditioner only needs a differentiable residual closure, so no recursive
solve is required to exercise it. For a linear residual the Gauss-Newton matrix is
exactly ``A^T A``, so full-mode preconditioning is exact Newton (one step) and the
quadratic model is exact (the gain-ratio trigger should never re-refresh).
"""

from pyzag import preconditioning, reparametrization

import torch

# Ensure test consistency
torch.manual_seed(42)

import warnings
import unittest

torch.set_default_dtype(torch.float64)


class TestGaussNewtonPreconditioner(unittest.TestCase):
    def setUp(self):
        # Seed locally, not off the global RNG. unittest runs methods in
        # alphabetical order, so a globally-seeded fixture makes every test's
        # problem depend on how many tests ran before it -- adding one test then
        # breaks an unrelated one.
        gen = torch.Generator().manual_seed(42)
        self.N, self.p = 20, 4
        self.A = torch.randn(self.N, self.p, generator=gen)
        self.xstar = torch.randn(self.p, generator=gen)
        self.b = self.A @ self.xstar

    def _make(self, mode, nsub, lr=1.0, **kwargs):
        theta = torch.zeros(self.p, requires_grad=True)
        opt = torch.optim.SGD([theta], lr=lr)
        pre = preconditioning.GaussNewtonPreconditioner(
            opt,
            [theta],
            mode=mode,
            nsub=nsub,
            lam=1e-10,
            generator=torch.Generator().manual_seed(0),
            **kwargs,
        )
        return theta, pre

    def _residual(self, theta):
        return self.A @ theta - self.b

    def _loss(self, theta):
        with torch.no_grad():
            r = self._residual(theta)
            return float(0.5 * torch.sum(r * r))

    def test_full_is_exact_newton(self):
        # Full-mode GN on a linear residual is exact Newton: one step -> solution.
        theta, pre = self._make("full", self.N)
        pre.step(lambda: self._residual(theta))
        self.assertTrue(torch.allclose(theta.detach(), self.xstar, atol=1e-6))
        self.assertLess(self._loss(theta), 1e-12)

    def test_full_H_equals_AtA(self):
        theta, pre = self._make("full", self.N)
        pre._refresh(self._residual(theta), torch.ones(self.N))
        self.assertTrue(torch.allclose(pre._H, self.A.T @ self.A, atol=1e-8))

    def test_gain_ratio_avoids_refresh(self):
        # Exact quadratic => gain ratio ~ 1 => refresh only once (at the start).
        theta, pre = self._make("full", self.N)
        for _ in range(10):
            pre.step(lambda: self._residual(theta))
        self.assertEqual(pre.n_refreshes, 1)
        self.assertEqual(pre.n_steps, 10)

    def test_diag_reduces_loss(self):
        theta, pre = self._make("diag", self.N, lr=0.5)
        l0 = self._loss(theta)
        for _ in range(50):
            pre.step(lambda: self._residual(theta))
        self.assertLess(self._loss(theta), 0.5 * l0)

    def test_explicit_sample_indices(self):
        # Passing all rows explicitly reproduces the exact full-mode Newton step.
        theta, pre = self._make("full", 0, sample_indices=list(range(self.N)))
        pre.step(lambda: self._residual(theta))
        self.assertTrue(torch.allclose(theta.detach(), self.xstar, atol=1e-6))

    def test_out_of_range_indices_raise(self):
        theta, pre = self._make("diag", 0, sample_indices=[0, self.N + 5])
        with self.assertRaises(IndexError):
            pre.step(lambda: self._residual(theta))

    # A non-finite residual on the *first* step is covered by
    # `test_non_finite_at_the_initial_point_raises`, and one with a good point
    # behind it by `test_non_finite_residual_backs_out_and_recovers`. The test
    # that used to sit here asserted the pre-fix contract -- warn, return nan,
    # leave theta put -- which is what let a caller in a loop spin forever.

    def test_on_refresh_callback_and_refresh_steps(self):
        events = []
        theta = torch.zeros(self.p, requires_grad=True)
        opt = torch.optim.SGD([theta], lr=0.5)
        pre = preconditioning.GaussNewtonPreconditioner(
            opt,
            [theta],
            mode="diag",
            nsub=self.N,
            lam=1e-6,
            rho=1e12,  # refresh eagerly
            generator=torch.Generator().manual_seed(0),
            on_refresh=lambda **kw: events.append(kw),
        )
        for _ in range(5):
            pre.step(lambda: self._residual(theta))
        self.assertEqual(len(pre.refresh_steps), pre.n_refreshes)
        self.assertEqual([e["step"] for e in events], pre.refresh_steps)
        self.assertEqual(pre.refresh_steps[0], 0)  # first step always refreshes

    def test_fixed_preconditioner_computes_once(self):
        # rho=None: compute the curvature once and never refresh.
        theta = torch.zeros(self.p, requires_grad=True)
        opt = torch.optim.SGD([theta], lr=0.2)
        pre = preconditioning.GaussNewtonPreconditioner(
            opt,
            [theta],
            mode="diag",
            nsub=self.N,
            lam=1e-6,
            rho=None,
            generator=torch.Generator().manual_seed(0),
        )
        l0 = self._loss(theta)
        for _ in range(50):
            pre.step(lambda: self._residual(theta))
        self.assertEqual(pre.n_refreshes, 1)
        self.assertEqual(pre.refresh_steps, [0])
        self.assertLess(self._loss(theta), 0.5 * l0)

    def test_optimizer_is_swappable(self):
        # Swappable *within* the gradient-proportional family: momentum and a
        # different learning rate change the trajectory but not the contract.
        # (Adam is deliberately not swappable here -- see
        # test_adam_family_is_rejected.)
        theta = torch.zeros(self.p, requires_grad=True)
        opt = torch.optim.SGD([theta], lr=0.2, momentum=0.9)
        pre = preconditioning.GaussNewtonPreconditioner(
            opt,
            [theta],
            mode="diag",
            nsub=self.N,
            lam=1e-6,
            # Momentum and LM damping are incompatible -- see
            # test_momentum_and_lam_adapt_warns.
            lam_adapt=False,
            generator=torch.Generator().manual_seed(0),
        )
        l0 = self._loss(theta)
        for _ in range(200):
            pre.step(lambda: self._residual(theta))
        self.assertLess(self._loss(theta), 1e-3 * l0)

    def test_momentum_and_lam_adapt_warns(self):
        """Momentum carries a velocity that damping cannot shorten, so the pair
        stalls in a reject/restore cycle. Warn rather than let it look like a
        convergence failure."""
        theta = torch.zeros(self.p, requires_grad=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            preconditioning.GaussNewtonPreconditioner(
                torch.optim.SGD([theta], lr=0.2, momentum=0.9), [theta], lam_adapt=True
            )
        self.assertTrue(any("momentum" in str(c.message) for c in caught))

    def test_lam_adapt_ignores_roundoff_at_convergence(self):
        """Once converged the loss is flat to within roundoff. A bare
        ``loss > previous`` test then rejects good steps forever on numerical
        noise, so rejection needs a tolerance."""
        theta, pre = self._make("full", self.N)  # exact Newton: converges in one step
        for _ in range(10):
            pre.step(lambda: self._residual(theta))
        self.assertEqual(pre.n_steps, 10)  # every step accepted
        self.assertEqual(pre.curvature.n_rejected, 0)

    def test_lam_adapt_recovers_from_an_overshoot(self):
        """The point of LM: a step that increases the loss is backed out and the
        damping raised, so the run recovers instead of diverging."""
        theta = torch.zeros(self.p, requires_grad=True)
        pre = preconditioning.GaussNewtonPreconditioner(
            torch.optim.SGD([theta], lr=3.0),  # deliberately far too long a step
            [theta],
            mode="diag",
            nsub=self.N,
            lam=1e-8,
            lam_adapt=True,
            generator=torch.Generator().manual_seed(0),
        )
        l0 = self._loss(theta)
        for _ in range(60):
            pre.step(lambda: self._residual(theta))
        self.assertLess(self._loss(theta), l0)
        self.assertGreater(pre.curvature.lam, 1e-8)  # damping actually engaged

    def test_weighted_H_equals_At_W_A(self):
        # W != I: the estimator must form A^T W A, not A^T A.
        w = torch.rand(self.N) + 0.5
        theta = torch.zeros(self.p, requires_grad=True)
        pre = preconditioning.GaussNewtonPreconditioner(
            torch.optim.SGD([theta], lr=0.0),
            [theta],
            mode="full",
            nsub=self.N,
            lam=1e-10,
            weights=w,
            generator=torch.Generator().manual_seed(0),
        )
        pre.step(lambda: self._residual(theta))
        self.assertTrue(
            torch.allclose(pre._H, self.A.T @ torch.diag(w) @ self.A, atol=1e-8)
        )

    def test_subsampled_H_is_unbiased(self):
        # nsub < N: each draw is noisy, but averaging over draws must converge to
        # the full A^T A -- that is what the n/nsampled rescaling is for. A biased
        # estimator would sit off by roughly the N/nsub factor (4x here), so a
        # loose tolerance still discriminates; the seed is local because this
        # file seeds torch once at module scope, making setUp's A order-dependent.
        gen = torch.Generator().manual_seed(0)
        A = torch.randn(self.N, self.p, generator=gen)
        theta = torch.zeros(self.p, requires_grad=True)
        acc = torch.zeros(self.p)
        ndraw = 2000
        for _ in range(ndraw):
            curv = preconditioning.GaussNewtonCurvature(
                [theta], mode="diag", nsub=5, generator=gen
            )
            curv.refresh(A @ theta - A @ self.xstar)
            acc += curv.H
        exact = torch.diagonal(A.T @ A)
        self.assertTrue(torch.allclose(acc / ndraw, exact, rtol=0.1))

    def test_packing_across_several_parameters(self):
        # Multiple parameter tensors of different shapes: the packed offsets must
        # line up, and a parameter the residual never touches must come back as
        # zeros (allow_unused) rather than None.
        a = torch.zeros(2, 3, requires_grad=True)
        b = torch.zeros(4, requires_grad=True)
        unused = torch.zeros(5, requires_grad=True)
        curv = preconditioning.GaussNewtonCurvature(
            [a, b, unused], mode="diag", nsub=10
        )
        self.assertEqual(curv.nparam, 6 + 4 + 5)

        residual = torch.cat([(3.0 * a).reshape(-1), (5.0 * b).reshape(-1)])
        curv.refresh(residual)
        got = curv.H
        self.assertTrue(torch.allclose(got[:6], torch.full((6,), 9.0)))
        self.assertTrue(torch.allclose(got[6:10], torch.full((4,), 25.0)))
        self.assertTrue(torch.allclose(got[10:], torch.zeros(5)))

        # write_grads must scatter back into the original shapes.
        curv.write_grads(torch.arange(15, dtype=torch.float64))
        self.assertEqual(tuple(a.grad.shape), (2, 3))
        self.assertEqual(tuple(b.grad.shape), (4,))
        self.assertTrue(
            torch.allclose(b.grad, torch.arange(6, 10, dtype=torch.float64))
        )

    def test_grouped_cotangents_match_exact_rows(self):
        # Cotangent groups are exact when the grouped rows have disjoint parameter
        # support -- here a block-diagonal residual, one block per "specimen".
        nrow, nblock, k = 5, 6, 3
        th = torch.zeros(nblock, k, requires_grad=True)
        coef = torch.randn(nrow, nblock, k)
        residual = (coef * th).sum(-1)

        exact = preconditioning.GaussNewtonCurvature(
            [th], mode="diag", nsub=nrow * nblock
        )
        exact.refresh(residual)

        grouped = preconditioning.GaussNewtonCurvature(
            [th],
            mode="diag",
            cotangents=[
                torch.nn.functional.one_hot(torch.tensor(i), nrow)
                .to(residual.dtype)
                .unsqueeze(-1)
                .expand(nrow, nblock)
                for i in range(nrow)
            ],
        )
        grouped.refresh(residual)
        self.assertTrue(torch.allclose(exact.H, grouped.H))
        self.assertEqual(exact.n_refresh_sweeps, nrow * nblock)
        self.assertEqual(grouped.n_refresh_sweeps, nrow)

    def _badly_scaled(self):
        """A quadratic whose curvature spans six orders of magnitude."""
        D = torch.tensor([1e-3, 1.0, 1e3])
        xstar = torch.tensor([5.0, 5.0, 5.0])
        return D, xstar

    def _run_scaled(self, opt_cls, lr, apply, niter=200, lam=1e-10):
        D, xstar = self._badly_scaled()
        theta = torch.zeros(3, requires_grad=True)
        opt = opt_cls([theta], lr=lr)
        pre = preconditioning.GaussNewtonPreconditioner(
            opt, [theta], mode="diag", nsub=3, lam=lam, rho=None, apply=apply
        )
        # r = sqrt(D) * (theta - xstar)  =>  J^T J = diag(D), loss = 0.5 (theta-xstar)^T D (theta-xstar)
        for _ in range(niter):
            pre.step(lambda: torch.sqrt(D) * (theta - xstar))
        return float(0.5 * (D * (theta.detach() - xstar) ** 2).sum())

    def test_adam_is_invariant_to_gradient_preconditioning(self):
        """The reason Adam-family optimizers are rejected outright.

        Adam divides each coordinate by its own running gradient RMS, so scaling
        the gradient by a fixed diagonal leaves its step unchanged however badly
        scaled the problem is -- preconditioning it is a no-op, not a weaker
        effect. (Invariance is exact in the limit; the residual ~1e-7
        disagreement is Adam's ``eps`` in ``m / (sqrt(v) + eps)``, whose relative
        weight shifts when the gradient is rescaled.)

        This is measured directly, bypassing the constructor guard, so the
        premise behind the guard stays pinned rather than merely asserted.
        """
        D, xstar = self._badly_scaled()

        def run(precondition):
            theta = torch.zeros(3, requires_grad=True)
            opt = torch.optim.Adam([theta], lr=0.05)
            for _ in range(200):
                opt.zero_grad()
                (0.5 * (D * (theta - xstar) ** 2).sum()).backward()
                if precondition:
                    theta.grad = theta.grad / D  # the exact GN diagonal here
                opt.step()
            return float(0.5 * (D * (theta.detach() - xstar) ** 2).sum())

        plain, preconditioned = run(False), run(True)
        self.assertLess(abs(plain - preconditioned) / abs(plain), 1e-6)

    def test_adam_family_is_rejected(self):
        """The no-op must fail loudly at construction, not silently at run time."""
        theta = torch.zeros(self.p, requires_grad=True)
        for cls in (torch.optim.Adam, torch.optim.AdamW, torch.optim.RMSprop):
            with self.assertRaises(TypeError):
                preconditioning.GaussNewtonPreconditioner(cls([theta], lr=0.1), [theta])
        # ...and can be overridden by a caller who knows what they are doing.
        preconditioning.GaussNewtonPreconditioner(
            torch.optim.Adam([theta], lr=0.1), [theta], check_optimizer=False
        )

    def test_sgd_is_accepted_and_gets_the_newton_step(self):
        """SGD's step is proportional to its gradient, so at lr=1 the update is
        exactly the damped Gauss-Newton step."""
        D, xstar = self._badly_scaled()
        theta = torch.zeros(3, requires_grad=True)
        pre = preconditioning.GaussNewtonPreconditioner(
            torch.optim.SGD([theta], lr=1.0),
            [theta],
            mode="diag",
            nsub=3,
            lam=1e-12,
            rho=None,
        )
        pre.step(lambda: torch.sqrt(D) * (theta - xstar))
        self.assertTrue(torch.allclose(theta.detach(), xstar, atol=1e-6))

    def test_recovers_when_the_residual_cannot_be_evaluated(self):
        """A stiff model driven out of its valid region raises instead of
        returning a number. That is the limiting case of too long a step, so the
        run must back out and damp rather than die."""
        theta = torch.zeros(self.p, requires_grad=True)
        pre = preconditioning.GaussNewtonPreconditioner(
            torch.optim.SGD([theta], lr=1.0),
            [theta],
            mode="diag",
            nsub=self.N,
            lam=1e-8,
            lam_adapt=True,
            generator=torch.Generator().manual_seed(0),
        )
        # theta0 is the last point the residual was successfully evaluated at;
        # the step taken from it lands on theta1, which is never validated until
        # the next evaluation -- so theta0 is what recovery must return to.
        theta0 = theta.detach().clone()
        pre.step(lambda: self._residual(theta))
        lam_before = pre.curvature.lam

        def explode():
            raise RuntimeError("inner solve did not converge")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = pre.step(explode)
        self.assertNotEqual(out, out)  # nan
        self.assertTrue(any("could not be evaluated" in str(c.message) for c in caught))
        self.assertTrue(torch.allclose(theta.detach(), theta0))  # rolled back
        self.assertGreater(pre.curvature.lam, lam_before)  # damped harder
        self.assertEqual(pre.curvature.n_failed, 1)
        # ...and the run carries on from there, with the harder damping.
        pre.step(lambda: self._residual(theta))
        self.assertLess(self._loss(theta), self._loss_at(theta0))
        self.assertFalse(torch.allclose(theta.detach(), theta0))

    def test_non_finite_residual_backs_out_and_recovers(self):
        """A residual that *returns* nan is the same situation as one that
        raises, and needs the same recovery. It used to get none: the step reset
        the anchor and returned, leaving theta sitting at the dead point, so
        every later step recomputed nan and skipped again. The run spun out its
        whole iteration budget without moving while ``lam`` never rose -- seen on
        the NEML2 calibration as 79 wasted iterations that then reported the
        stale iteration-0 loss as if it were a converged result."""
        theta = torch.zeros(self.p, requires_grad=True)
        pre = preconditioning.GaussNewtonPreconditioner(
            torch.optim.SGD([theta], lr=1.0),
            [theta],
            mode="diag",
            nsub=self.N,
            lam=1e-8,
            lam_adapt=True,
            generator=torch.Generator().manual_seed(0),
        )
        theta0 = theta.detach().clone()
        pre.step(lambda: self._residual(theta))
        lam_before = pre.curvature.lam

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = pre.step(lambda: self._residual(theta) * float("nan"))
        self.assertNotEqual(out, out)  # nan
        self.assertTrue(any("backing out" in str(c.message) for c in caught))
        self.assertTrue(torch.allclose(theta.detach(), theta0))  # rolled back
        self.assertGreater(pre.curvature.lam, lam_before)  # damped harder
        self.assertEqual(pre.curvature.n_failed, 1)
        # ...and it moves again afterwards instead of spinning at the dead point.
        pre.step(lambda: self._residual(theta))
        self.assertFalse(torch.allclose(theta.detach(), theta0))
        self.assertLess(self._loss(theta), self._loss_at(theta0))

    def test_two_consecutive_failures_still_have_somewhere_to_land(self):
        """A rollback invalidates the quadratic model, but the point it rewinds
        *to* is the best-established one available -- the parameters are sitting
        on it. Clearing both together left the step after a rejection with no
        fallback, so a second failure raised 'no previous point' while a known
        good point was loaded in the parameters. Observed on the NEML2
        calibration, where the run alternates rejection and recovery."""
        theta = torch.zeros(self.p, requires_grad=True)
        pre = preconditioning.GaussNewtonPreconditioner(
            torch.optim.SGD([theta], lr=1.0),
            [theta],
            mode="diag",
            nsub=self.N,
            lam=1e-8,
            lam_adapt=True,
            generator=torch.Generator().manual_seed(0),
        )
        theta0 = theta.detach().clone()
        pre.step(lambda: self._residual(theta))  # establish a good point
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pre.step(lambda: self._residual(theta) * float("nan"))
            pre.step(lambda: self._residual(theta) * float("nan"))  # used to raise
        self.assertTrue(torch.allclose(theta.detach(), theta0))
        self.assertEqual(pre.curvature.n_failed, 2)
        # ...and the run is still able to make progress from there.
        pre.step(lambda: self._residual(theta))
        self.assertLess(self._loss(theta), self._loss_at(theta0))

    def test_nan_loss_is_rejected_rather_than_accepted(self):
        """``nan > tol`` is False, so a non-finite objective used to read as an
        *accepted* step. It then landed in the anchor, after which every later
        comparison against nan was False too and the damping froze for good.
        This is the path SVI reaches, where the ELBO arrives from outside."""
        theta = torch.zeros(self.p, requires_grad=True)
        curv = preconditioning.GaussNewtonCurvature([theta], mode="diag", nsub=self.N)
        curv.refresh(self._residual(theta))
        curv.note_step(
            torch.zeros(self.p),
            torch.zeros(self.p),
            1.0,
            theta=theta.detach().clone(),
        )
        lam_before = curv.lam
        self.assertTrue(curv.adapt_damping(float("nan"), None))
        self.assertGreater(curv.lam, lam_before)

    def test_non_finite_at_the_initial_point_raises(self):
        """With no anchor there is nothing to fall back to. Returning nan
        forever would hide a bad starting guess behind a silent no-op."""
        theta = torch.zeros(self.p, requires_grad=True)
        pre = preconditioning.GaussNewtonPreconditioner(
            torch.optim.SGD([theta], lr=1.0), [theta], nsub=self.N
        )
        with self.assertRaises(ValueError):
            pre.step(lambda: self._residual(theta) * float("nan"))

    def _loss_at(self, vec):
        with torch.no_grad():
            r = self.A @ vec - self.b
            return float(0.5 * torch.sum(r * r))

    def test_first_evaluation_failure_propagates(self):
        """With no good point to fall back to there is nothing to recover to;
        swallowing the error would hide a broken model behind a nan."""
        theta = torch.zeros(self.p, requires_grad=True)
        pre = preconditioning.GaussNewtonPreconditioner(
            torch.optim.SGD([theta], lr=1.0), [theta], nsub=self.N
        )
        with self.assertRaises(RuntimeError):
            pre.step(lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    def test_full_mode_warns_when_rank_deficient(self):
        """Sampling fewer rows than parameters leaves the full-mode H singular.
        Marquardt damping papers over it, so the warning is the only signal --
        and it lives on a branch nothing else exercises."""
        theta = torch.zeros(self.p, requires_grad=True)
        curv = preconditioning.GaussNewtonCurvature(
            [theta], mode="full", nsub=self.p - 1
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            curv.refresh(self._residual(theta))
        self.assertTrue(any("rank deficient" in str(c.message) for c in caught))

    def test_set_H_rejects_wrong_shape(self):
        theta = torch.zeros(self.p, requires_grad=True)
        curv = preconditioning.GaussNewtonCurvature([theta], mode="diag")
        with self.assertRaises(ValueError):
            curv.set_H(torch.zeros(self.p, self.p))
        curv.set_H(torch.ones(self.p), sweeps=3)
        self.assertEqual(curv.n_refreshes, 1)
        self.assertEqual(curv.n_refresh_sweeps, 3)


if __name__ == "__main__":
    unittest.main()


class TestCurvatureRescale(unittest.TestCase):
    """The second lever: a static, curvature-derived reparametrization.

    Conditioned by changing coordinates rather than by preconditioning the
    gradient, which is what makes it work with optimizers that reject the
    preconditioner.

    The fixture matters. Whitening helps when the **parameter magnitudes** are
    heterogeneous, not merely when the curvature is: Adam already takes roughly
    equal steps in every coordinate, which is close to ideal when every parameter
    is O(1), and whitening would only spoil it. So the solution here spans four
    decades -- as the real calibration does, where a log rate coefficient of
    -8.7 sits beside a hardening ordinate of 300 -- with column magnitudes scaled
    inversely, so each parameter's sensitivity reflects its own scale.
    """

    def setUp(self):
        self.N, self.p = 20, 4
        gen = torch.Generator().manual_seed(3)
        self.xstar = torch.tensor([1e-2, 1.0, 1e2, 1e3])
        self.A = torch.randn(self.N, self.p, generator=gen) / self.xstar
        self.b = self.A @ self.xstar

    def _residual(self, theta):
        return self.A @ theta - self.b

    def _loss(self, theta):
        with torch.no_grad():
            r = self._residual(theta)
            return float(0.5 * torch.sum(r * r))

    def test_scale_is_inverse_sqrt_curvature(self):
        theta = torch.zeros(self.p, requires_grad=True)
        raw = preconditioning.gauss_newton_rescalers(
            [("theta", theta)],
            lambda: self._residual(theta),
            nsub=self.N,
            normalize=False,
        )["theta"].scale
        expected = torch.rsqrt(torch.diagonal(self.A.T @ self.A))
        self.assertTrue(torch.allclose(raw, expected, rtol=1e-8))

    def test_default_is_the_exact_estimate(self):
        """A frozen scale should be right by default and cheap only on request:
        the caller has no way to know what nsub their problem needs."""
        theta = torch.zeros(self.p, requires_grad=True)
        est = preconditioning.CurvatureEstimator([theta])
        self.assertIsNone(est.nsub)
        H, sweeps = est.estimate(self._residual(theta))
        self.assertEqual(sweeps, self.N)  # every row
        self.assertTrue(torch.allclose(H, torch.diagonal(self.A.T @ self.A), rtol=1e-9))
        # The preconditioner re-estimates and damps, so it stays cheap.
        self.assertEqual(preconditioning.GaussNewtonCurvature([theta]).nsub, 8)

    def test_guard_rejects_an_ill_conditioned_estimate(self):
        """The failure this guard exists for is silent in ``H``.

        An under-sampled row set can leave one parameter's curvature orders of
        magnitude low; ``H`` still looks reasonable but ``1/sqrt(H)`` hands that
        parameter a runaway step. Measured on the real calibration, nsub=8 gave
        an H 17% off and a scale wrong by 1e12.
        """
        theta = torch.zeros(self.p, requires_grad=True)
        # A residual that barely touches the last coordinate.
        A = self.A.clone()
        A[:, -1] *= 1e-9

        with self.assertRaises(ValueError) as ctx:
            preconditioning.gauss_newton_rescalers(
                [("theta", theta)], lambda: A @ theta - self.b
            )
        msg = str(ctx.exception)
        self.assertIn("cond_max", msg)
        self.assertIn("theta", msg)
        # ...and it is a guard, not a hard ceiling: a caller who knows the spread
        # is real can say so.
        preconditioning.gauss_newton_rescalers(
            [("theta", theta)], lambda: A @ theta - self.b, cond_max=1e30
        )

    def test_guard_still_names_unconstrained_parameters(self):
        theta = torch.zeros(self.p, requires_grad=True)
        unused = torch.zeros(2, requires_grad=True)
        with self.assertRaises(ValueError) as ctx:
            preconditioning.gauss_newton_rescalers(
                [("theta", theta), ("unused", unused)],
                lambda: self._residual(theta),
            )
        self.assertIn("unused", str(ctx.exception))
        self.assertIn("non-positive", str(ctx.exception))

    def test_normalization_keeps_the_profile_and_fixes_the_level(self):
        """``1/sqrt(H)`` carries units of theta/residual, so its absolute level is
        a property of the residual's units, not of the parameters -- on the real
        calibration it sits ~920x below the hand-picked range widths. Normalizing
        hands the learning rate back its usual meaning while leaving the entire
        relative profile, which is the part the curvature actually measures."""
        theta = torch.zeros(self.p, requires_grad=True)
        kw = dict(nsub=self.N)
        raw = preconditioning.gauss_newton_rescalers(
            [("theta", theta)], lambda: self._residual(theta), normalize=False, **kw
        )["theta"].scale
        norm = preconditioning.gauss_newton_rescalers(
            [("theta", theta)], lambda: self._residual(theta), normalize=True, **kw
        )["theta"].scale
        self.assertAlmostEqual(float(torch.log(norm).mean()), 0.0, places=10)
        # Same shape, different level: the ratio is one constant.
        ratio = raw / norm
        self.assertTrue(torch.allclose(ratio, ratio[0].expand_as(ratio), rtol=1e-10))

    def test_round_trip_and_std_dev(self):
        s = reparametrization.CurvatureRescale(torch.tensor([2.0, 0.5]), offset=1.0)
        x = torch.tensor([3.0, -4.0])
        self.assertTrue(torch.allclose(s.reverse(s.forward(x)), x))
        self.assertTrue(
            torch.allclose(
                s.forward_std_dev(x), torch.tensor([6.0, -2.0]).abs() * torch.sign(x)
            )
        )
        self.assertTrue(torch.allclose(s.reverse_std_dev(s.forward_std_dev(x)), x))

    def test_bounds_clamp_in_natural_units(self):
        s = reparametrization.CurvatureRescale(
            torch.tensor([1.0]), lb=torch.tensor([-1.0]), ub=torch.tensor([1.0])
        )
        self.assertAlmostEqual(float(s.forward(torch.tensor([5.0]))), 1.0)
        self.assertAlmostEqual(float(s.forward(torch.tensor([-5.0]))), -1.0)
        with self.assertRaises(ValueError):
            reparametrization.CurvatureRescale(
                torch.tensor([1.0]), lb=torch.tensor([0.0])
            )

    def test_whitening_is_not_a_no_op_for_adam(self):
        """The whole point of the second lever.

        Adam is invariant to gradient preconditioning (see
        test_adam_is_invariant_to_gradient_preconditioning), but a
        reparametrization moves its moment estimates into the new coordinates
        too -- so this one does reach it, and helps.
        """

        def run(whiten, niter=150, lr=0.05):
            theta = torch.zeros(self.p, requires_grad=True)
            scale = torch.ones(self.p)
            if whiten:
                probe = torch.zeros(self.p, requires_grad=True)
                scale = preconditioning.gauss_newton_rescalers(
                    [("p", probe)], lambda: self._residual(probe), nsub=self.N
                )["p"].scale
            opt = torch.optim.Adam([theta], lr=lr)
            for _ in range(niter):
                opt.zero_grad()
                # theta is the *scaled* coordinate; the residual sees theta*scale.
                r = self.A @ (theta * scale) - self.b
                (0.5 * torch.sum(r * r)).backward()
                opt.step()
            with torch.no_grad():
                r = self.A @ (theta * scale) - self.b
                return float(0.5 * torch.sum(r * r))

        plain, whitened = run(False), run(True)
        self.assertLess(whitened, 0.5 * plain)

    def test_installs_through_reparameterizer(self):
        """End to end: build scalers from curvature, install them with the
        existing Reparameterizer, and optimize with Adam."""

        class Tiny(torch.nn.Module):
            def __init__(self, p):
                super().__init__()
                self.theta = torch.nn.Parameter(torch.zeros(p))

        model = Tiny(self.p)
        resid = lambda: self.A @ model.theta - self.b  # noqa: E731
        scalers = preconditioning.gauss_newton_rescalers(
            model.named_parameters(), resid, nsub=self.N
        )
        self.assertEqual(set(scalers), {"theta"})
        reparametrization.Reparameterizer(scalers, error_not_provided=True)(model)

        l0 = float(0.5 * torch.sum(resid() ** 2))
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        for _ in range(200):
            opt.zero_grad()
            (0.5 * torch.sum(resid() ** 2)).backward()
            opt.step()
        # Two orders of magnitude is an unambiguous "it works"; the point of this
        # test is that the pieces compose, not how fast Adam happens to converge.
        self.assertLess(float(0.5 * torch.sum(resid() ** 2)), 1e-2 * l0)
