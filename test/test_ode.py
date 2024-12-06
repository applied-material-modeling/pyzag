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

"""Test solving ODEs as specialized nonlinear recursive equations"""

from pyzag import ode, nonlinear, chunktime

import itertools

import torch

# Ensure test consistency
torch.manual_seed(42)

import unittest

torch.set_default_dtype(torch.float64)


class LogisticODE(torch.nn.Module):
    def __init__(self, r, K):
        super().__init__()
        self.r = torch.tensor(r)
        self.K = torch.tensor(K)

    def forward(self, t, y):
        return (
            self.r * (1.0 - y / self.K) * y,
            (self.r - (2 * self.r * y) / self.K)[..., None],
        )

    def exact(self, t):
        y0 = self.y0(t.shape[-2])
        return (
            self.K
            * torch.exp(self.r * t)
            * y0[None, ...]
            / (self.K + (torch.exp(self.r * t) - 1) * y0[None, ...])
        )

    def y0(self, nbatch):
        return torch.linspace(0, 1, nbatch).reshape(nbatch, 1)


class TestBackwardEulerTimeIntegrationLogistic(unittest.TestCase):
    def setUp(self):
        self.model = ode.BackwardEulerODE(LogisticODE(1.0, 1.0))

        self.nbatch = 5
        self.ntime = 100

        self.times = (
            torch.linspace(0, 1, self.ntime)
            .unsqueeze(-1)
            .expand(-1, self.nbatch)
            .unsqueeze(-1)
        )
        self.y = torch.rand((self.ntime, self.nbatch, 1))
        self.y0 = self.model.ode.y0(self.nbatch)

    def test_shapes(self):
        nchunk = 8
        R, J = self.model(
            self.y[: nchunk + self.model.lookback],
            self.times[: nchunk + self.model.lookback],
        )

        self.assertEqual(R.shape, (nchunk, self.nbatch, 1))
        self.assertEqual(J.shape, (1 + self.model.lookback, nchunk, self.nbatch, 1, 1))

    def test_integrate_forward(self):
        nchunk = 8

        for method in [
            chunktime.BidiagonalThomasFactorization,
            chunktime.BidiagonalPCRFactorization,
            chunktime.BidiagonalHybridFactorization(2),
        ]:
            solver = nonlinear.RecursiveNonlinearEquationSolver(
                self.model,
                step_generator=nonlinear.StepGenerator(nchunk),
                direct_solve_operator=method,
            )

            nres = solver.solve(self.y0, self.ntime, self.times)
            eres = self.model.ode.exact(self.times)

            self.assertEqual(nres.shape, eres.shape)
            self.assertTrue(torch.allclose(nres, eres, rtol=1e-3))


class LinearSystem(torch.nn.Module):
    """Linear system of equations"""

    def __init__(self, n):
        super().__init__()
        self.n = n
        Ap = torch.rand((n, n))
        self.A = Ap.transpose(0, 1) * Ap

        self.vals, self.vecs = torch.linalg.eigh(self.A)

    def forward(self, t, y):
        return torch.matmul(self.A.unsqueeze(0).unsqueeze(0), y.unsqueeze(-1)).squeeze(
            -1
        ), self.A.unsqueeze(0).unsqueeze(0).expand(
            t.shape[0], t.shape[1], self.n, self.n
        )

    def exact(self, t):
        c0 = torch.linalg.solve(
            self.vecs.unsqueeze(0).expand(t.shape[1], -1, -1),
            self.y0(t.shape[1]),
        )
        soln = torch.zeros((t.shape[:-1] + (self.n,)))

        for i in range(self.n):
            soln += c0[:, i, None] * torch.exp(self.vals[i] * t) * self.vecs[:, i]

        return soln

    def y0(self, nbatch):
        return torch.outer(torch.linspace(-1, 1, nbatch), torch.linspace(1, 2, self.n))


class TestIntegrateLinear(unittest.TestCase):
    def setUp(self):
        self.n = [1, 4, 5]
        self.nchunk = [1, 3, 5]
        self.method = [ode.BackwardEulerODE, ode.ForwardEulerODE]
        self.linear_method = [
            chunktime.BidiagonalThomasFactorization,
            chunktime.BidiagonalPCRFactorization,
            chunktime.BidiagonalHybridFactorization(2),
        ]

        self.nbatch = [1, 5]
        self.ninit = [None, 2]
        self.ntime = 100

        self.ref_times = torch.linspace(0, 0.1, self.ntime)

    def test_integrate_forward(self):
        for n, nchunk, ninit, method, linear_method, nbatch in itertools.product(
            self.n,
            self.nchunk,
            self.ninit,
            self.method,
            self.linear_method,
            self.nbatch,
        ):
            with self.subTest(
                n=n,
                nchunk=nchunk,
                ninit=ninit,
                method=method,
                linear_method=linear_method,
                nbatch=nbatch,
            ):
                model = method(LinearSystem(n))
                y0 = model.ode.y0(nbatch)
                times = self.ref_times.unsqueeze(-1).expand(-1, nbatch).unsqueeze(-1)

                if ninit is None or ninit <= nchunk:
                    nn = 0
                else:
                    nn = ninit

                solver = nonlinear.RecursiveNonlinearEquationSolver(
                    model,
                    step_generator=nonlinear.StepGenerator(nchunk, first_block_size=nn),
                    direct_solve_operator=linear_method,
                )

                nres = solver.solve(y0, self.ntime, times)
                eres = model.ode.exact(times)

                self.assertEqual(nres.shape, eres.shape)

                self.assertTrue(torch.allclose(nres, eres, rtol=1e-2))
