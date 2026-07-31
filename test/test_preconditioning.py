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

from pyzag import preconditioning

import torch

# Ensure test consistency
torch.manual_seed(42)

import warnings
import unittest

torch.set_default_dtype(torch.float64)


class TestGaussNewtonPreconditioner(unittest.TestCase):
    def setUp(self):
        self.N, self.p = 20, 4
        self.A = torch.randn(self.N, self.p)
        self.xstar = torch.randn(self.p)
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

    def test_nonfinite_residual_warns_and_skips(self):
        theta, pre = self._make("diag", self.N)
        before = theta.detach().clone()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = pre.step(
                lambda: torch.full((self.N,), float("nan"), requires_grad=True)
            )
        self.assertTrue(any("non-finite" in str(c.message) for c in caught))
        self.assertNotEqual(out, out)  # NaN
        self.assertTrue(torch.allclose(theta.detach(), before))  # unchanged

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
        opt = torch.optim.Adam([theta], lr=0.2)
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
        # The same preconditioner works with a different base optimizer (Adam).
        theta = torch.zeros(self.p, requires_grad=True)
        opt = torch.optim.Adam([theta], lr=0.2)
        pre = preconditioning.GaussNewtonPreconditioner(
            opt,
            [theta],
            mode="diag",
            nsub=self.N,
            lam=1e-6,
            generator=torch.Generator().manual_seed(0),
        )
        l0 = self._loss(theta)
        for _ in range(200):
            pre.step(lambda: self._residual(theta))
        self.assertLess(self._loss(theta), 1e-3 * l0)


if __name__ == "__main__":
    unittest.main()
