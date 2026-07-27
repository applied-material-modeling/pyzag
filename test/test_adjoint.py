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

"""Test adjoint gradient calculations"""

from pyzag import ode, nonlinear, chunktime
from pyzag.operators.dense import DenseBlockJacobian

import itertools

import torch

# Ensure test consistency
torch.manual_seed(42)

import unittest

torch.set_default_dtype(torch.float64)


def _factorizations():
    """Bidiagonal factorizations to sweep in the adjoint gradient checks.

    Thomas is the default; PCR and Hybrid exercise the parallel-cyclic-reduction
    path (including multi-window reduction for non-power-of-two chunk sizes such
    as nchunk=5, 7), whose adjoint was previously never gradient-checked.
    """
    return [
        chunktime.BidiagonalThomasFactorization,
        chunktime.BidiagonalPCRFactorization,
        chunktime.BidiagonalHybridFactorization(min_size=2),
    ]


def _adjoint_vs_ad(model, params, y0, ntime, times, nchunk, nn, fact):
    """Compare the adjoint gradient (using ``fact`` as the direct solve) against
    the plain-autograd gradient (using the default Thomas direct solve) for a
    shared ``model``.

    The factorization is varied only in the *adjoint* solver: it changes the
    internal linear solves, not the mathematical solution, so the true gradient
    is factorization-independent. The autograd reference always uses the default
    Thomas direct solve, since PCR/Hybrid deliberately run their forward under
    ``torch.no_grad`` and are not differentiable through plain autograd.
    """
    ref_solver = nonlinear.RecursiveNonlinearEquationSolver(
        model,
        step_generator=nonlinear.StepGenerator(nchunk, first_block_size=nn),
    )
    res1 = nonlinear.solve(ref_solver, y0, ntime, times)
    val1 = torch.linalg.norm(res1)
    val1.backward()
    derivs1 = [p.grad.clone() for p in params]

    model.zero_grad()

    adj_solver = nonlinear.RecursiveNonlinearEquationSolver(
        model,
        step_generator=nonlinear.StepGenerator(nchunk, first_block_size=nn),
        direct_solve_operator=fact,
    )
    res2 = nonlinear.solve_adjoint(adj_solver, y0, ntime, times)
    val2 = torch.linalg.norm(res2)
    val2.backward()
    derivs2 = [p.grad for p in params]

    return val1, val2, derivs1, derivs2


class SimpleLinearSystem(torch.nn.Module):
    """Linear system of equations"""

    def __init__(self, n):
        super().__init__()
        self.n = n
        Ap = torch.rand((n, n))
        self.A = torch.nn.Parameter(Ap.transpose(0, 1) * Ap)

    def forward(self, t, y):
        return torch.matmul(self.A.unsqueeze(0).unsqueeze(0), y.unsqueeze(-1)).squeeze(
            -1
        ), self.A.unsqueeze(0).unsqueeze(0).expand(
            t.shape[0], t.shape[1], self.n, self.n
        )

    def y0(self, nbatch):
        return torch.outer(torch.linspace(-1, 1, nbatch), torch.linspace(1, 2, self.n))


class ComplexLinearSystem(torch.nn.Module):
    """Linear system of equations"""

    def __init__(self, n, f=1.0):
        super().__init__()
        self.n = n
        self.A = torch.nn.Parameter(torch.rand((n, n)))
        self.f = f

    def forward(self, t, y):
        return torch.matmul(
            self.A.unsqueeze(0).unsqueeze(0) * torch.cos(t * self.f).unsqueeze(-1),
            y.unsqueeze(-1),
        ).squeeze(-1), self.A.unsqueeze(0).unsqueeze(0).expand(
            t.shape[0], t.shape[1], self.n, self.n
        ) * torch.cos(
            t * self.f
        ).unsqueeze(
            -1
        )

    def y0(self, nbatch):
        return torch.outer(torch.linspace(-1, 1, nbatch), torch.linspace(1, 2, self.n))


class TestAllLinearODEs(unittest.TestCase):
    def setUp(self):
        self.n = [1, 3, 6]
        self.nbatch = [1, 3, 5]
        self.nchunk = [1, 2, 5, 7]
        self.ninit = [None, 2]

        self.ode = [SimpleLinearSystem, ComplexLinearSystem]
        self.method = [ode.BackwardEulerODE, ode.ForwardEulerODE]

        self.ntime = 100

        self.ref_time = torch.linspace(0, 1, self.ntime)

    def test_all(self):
        for n, nbatch, nchunk, code, method, ninit in itertools.product(
            self.n, self.nbatch, self.nchunk, self.ode, self.method, self.ninit
        ):
            with self.subTest(
                n=n, nbatch=nbatch, nchunk=nchunk, code=code, method=method, ninit=ninit
            ):
                times = (
                    self.ref_time.clone().unsqueeze(-1).expand(-1, nbatch).unsqueeze(-1)
                )
                model = method(code(n))
                y0 = model.ode.y0(nbatch)
                if ninit is None or ninit <= nchunk:
                    nn = 0
                else:
                    nn = ninit
                solver = nonlinear.RecursiveNonlinearEquationSolver(
                    model,
                    step_generator=nonlinear.StepGenerator(nchunk, first_block_size=nn),
                )

                res1 = nonlinear.solve(solver, y0, self.ntime, times)
                val1 = torch.linalg.norm(res1)
                val1.backward()
                derivs1 = [p.grad for p in solver.func.parameters()]

                solver.zero_grad()
                res2 = nonlinear.solve_adjoint(solver, y0, self.ntime, times)
                val2 = torch.linalg.norm(res2)
                val2.backward()
                derivs2 = [p.grad for p in solver.func.parameters()]

                self.assertTrue(torch.isclose(val1, val2))
                for p1, p2 in zip(derivs1, derivs2):
                    self.assertTrue(torch.allclose(p1, p2))


class BernoulliODE(torch.nn.Module):
    """Linear system of equations"""

    def __init__(self, n, N):
        super().__init__()
        self.n = n
        self.P = torch.nn.Parameter(torch.rand(n) / 100.0)
        self.Q = torch.nn.Parameter(torch.rand(n) / 100.0)
        self.N = torch.tensor(N)

    def forward(self, t, y):
        yd = self.Q * y**self.N - self.P * y
        Jd = torch.diag_embed(self.Q * self.N * y ** (self.N - 1.0) - self.P)

        return yd, Jd

    def y0(self, nbatch):
        return torch.outer(torch.linspace(-1, 1, nbatch), torch.linspace(1, 2, self.n))


class TestComplexODE(unittest.TestCase):
    def setUp(self):
        self.n = [1, 3, 6]
        self.nbatch = [1, 3, 5]
        self.nchunk = [1, 2, 5, 7]
        self.ninit = [None, 2]

        self.ode = [lambda n: BernoulliODE(n, 3.0)]
        self.method = [ode.BackwardEulerODE]

        self.ntime = 100

        self.ref_time = torch.linspace(0, 1, self.ntime)

    def test_all(self):
        for n, nbatch, nchunk, code, method, ninit in itertools.product(
            self.n, self.nbatch, self.nchunk, self.ode, self.method, self.ninit
        ):
            with self.subTest(
                n=n, nbatch=nbatch, nchunk=nchunk, code=code, method=method, ninit=ninit
            ):
                times = (
                    self.ref_time.clone().unsqueeze(-1).expand(-1, nbatch).unsqueeze(-1)
                )
                model = method(code(n))
                y0 = model.ode.y0(nbatch)
                if ninit is None or ninit <= nchunk:
                    nn = 0
                else:
                    nn = ninit
                solver = nonlinear.RecursiveNonlinearEquationSolver(
                    model,
                    step_generator=nonlinear.StepGenerator(nchunk, first_block_size=nn),
                )

                res1 = nonlinear.solve(solver, y0, self.ntime, times)
                val1 = torch.linalg.norm(res1)
                val1.backward()
                derivs1 = [p.grad for p in solver.func.parameters()]

                solver.zero_grad()
                res2 = nonlinear.solve_adjoint(solver, y0, self.ntime, times)
                val2 = torch.linalg.norm(res2)
                val2.backward()
                derivs2 = [p.grad for p in solver.func.parameters()]

                self.assertTrue(torch.isclose(val1, val2))
                for p1, p2 in zip(derivs1, derivs2):
                    self.assertTrue(torch.allclose(p1, p2, rtol=1e-4))


class GeneralNonlinearEquation(
    torch.nn.Module, nonlinear.NonlinearFunctionOperatorFactory
):
    def __init__(self, n, f):
        super().__init__()
        self.n = n
        self.A = torch.nn.Parameter(torch.rand((n, n)) / 100.0)
        self.B = torch.nn.Parameter(torch.rand((n, n)) / 100.0)
        self.f = f
        from pyzag.ode import DenseODEWrapper

        self._wrapper = DenseODEWrapper()

    @property
    def lookback(self):
        return 1

    @property
    def wrapper(self):
        return self._wrapper

    def make_operator(self, prev_solution, forces, inverse_operator):
        return nonlinear.ChunkOp(self, prev_solution, forces, inverse_operator)

    def evaluate_raw(self, x_full, forces):
        x = x_full
        t = forces[0]
        R = (
            torch.matmul(
                self.A.unsqueeze(0).unsqueeze(0)
                * torch.cos(t[1:] * self.f).unsqueeze(-1),
                x[1:].unsqueeze(-1),
            ).squeeze(-1)
            + torch.matmul(
                self.B.unsqueeze(0).unsqueeze(0)
                * torch.sin(t[:-1] * self.f).unsqueeze(-1),
                x[:-1].unsqueeze(-1),
            ).squeeze(-1)
            - 0.1
        )

        J2 = self.A.unsqueeze(0).unsqueeze(0).expand(
            t.shape[0] - 1, t.shape[1], self.n, self.n
        ) * torch.cos(t[1:] * self.f).unsqueeze(-1)

        J1 = self.B.unsqueeze(0).unsqueeze(0).expand(
            t.shape[0] - 1, t.shape[1], self.n, self.n
        ) * torch.sin(t[:-1] * self.f).unsqueeze(-1)

        return R, DenseBlockJacobian(diag=J2, sub=J1)

    def y0(self, nbatch):
        return torch.outer(torch.linspace(-1, 1, nbatch), torch.linspace(1, 2, self.n))


class TestGeneralFunction(unittest.TestCase):
    def setUp(self):
        self.n = [1, 3, 6]
        self.nbatch = [1, 3, 5]
        self.nchunk = [1, 2, 5, 7]
        self.ninit = [None, 2]

        self.ntime = 100

        self.ref_time = torch.linspace(0, 0.01, self.ntime)

        self.f = 0.5

    def test_all(self):
        for n, nbatch, nchunk, ninit in itertools.product(
            self.n, self.nbatch, self.nchunk, self.ninit
        ):
            with self.subTest(n=n, nbatch=nbatch, nchunk=nchunk, ninit=ninit):
                times = (
                    self.ref_time.clone().unsqueeze(-1).expand(-1, nbatch).unsqueeze(-1)
                )
                model = GeneralNonlinearEquation(n, self.f)
                y0 = model.y0(nbatch)
                if ninit is None or ninit <= nchunk:
                    nn = 0
                else:
                    nn = ninit
                solver = nonlinear.RecursiveNonlinearEquationSolver(
                    model,
                    step_generator=nonlinear.StepGenerator(nchunk, first_block_size=nn),
                )

                res1 = nonlinear.solve(solver, y0, self.ntime, times)
                val1 = torch.linalg.norm(res1)
                val1.backward()
                derivs1 = [p.grad for p in solver.func.parameters()]

                solver.zero_grad()
                res2 = nonlinear.solve_adjoint(solver, y0, self.ntime, times)
                val2 = torch.linalg.norm(res2)
                val2.backward()
                derivs2 = [p.grad for p in solver.func.parameters()]

                self.assertTrue(torch.isclose(val1, val2))
                for p1, p2 in zip(derivs1, derivs2):
                    self.assertTrue(torch.allclose(p1, p2))


class AdjointSolveModule(torch.nn.Module):
    def __init__(self, solver):
        super().__init__()
        self.solver = solver

    def forward(self, y0, n, *forces):
        # Uses custom adjoint backward via solve_adjoint
        return nonlinear.solve_adjoint(self.solver, y0, n, *forces)


class TestGeneralFunctionJacRev(unittest.TestCase):
    def setUp(self):
        self.n = [1]
        self.nbatch = [1]
        self.nchunk = [1]
        self.ninit = [None]

        self.ntime = 100

        self.ref_time = torch.linspace(0, 0.01, self.ntime)

        self.f = 0.5

    def test_all(self):
        for n, nbatch, nchunk, ninit in itertools.product(
            self.n, self.nbatch, self.nchunk, self.ninit
        ):
            with self.subTest(n=n, nbatch=nbatch, nchunk=nchunk, ninit=ninit):
                times = (
                    self.ref_time.clone().unsqueeze(-1).expand(-1, nbatch).unsqueeze(-1)
                )
                model = GeneralNonlinearEquation(n, self.f)
                y0 = model.y0(nbatch)
                if ninit is None or ninit <= nchunk:
                    nn = 0
                else:
                    nn = ninit
                solver = nonlinear.RecursiveNonlinearEquationSolver(
                    model,
                    step_generator=nonlinear.StepGenerator(nchunk, first_block_size=nn),
                )

                params = dict(solver.named_parameters())
                jac_ad = torch.func.jacrev(torch.func.functional_call, argnums=1)(
                    solver, params, (y0, self.ntime, times)
                )

                wrapper = AdjointSolveModule(solver)
                params = dict(wrapper.named_parameters())
                buffers = dict(wrapper.named_buffers())

                def model_response(params, buffers, y0, n, *forces):
                    return torch.func.functional_call(
                        wrapper, (params, buffers), (y0, n, *forces)
                    )

                jac_adjoint = torch.func.jacrev(model_response, argnums=0)(
                    params, buffers, y0, self.ntime, times
                )

                for k in jac_ad.keys():
                    mod_key = "solver." + k
                    self.assertTrue(mod_key in jac_adjoint)
                    self.assertTrue(
                        torch.allclose(
                            jac_adjoint[mod_key], jac_ad[k], rtol=1e-4, atol=1e-6
                        )
                    )


class TestPCRHybridAdjoint(unittest.TestCase):
    """Adjoint gradients through the PCR and Hybrid factorizations.

    The existing sweeps above validate the adjoint only through the default
    Thomas factorization. This focused sweep exercises the PCR and Hybrid
    factorizations, including non-power-of-two chunk sizes (nchunk=3, 5, 7) that
    trigger multi-window parallel cyclic reduction — the path where the reduced
    subdiagonal (``B_red``) is written back and consumed by the next window. Each
    adjoint gradient is checked against the plain-autograd (Thomas) reference.
    """

    def setUp(self):
        self.n = [3, 6]
        self.nbatch = [1, 4]
        self.nchunk = [3, 5, 7, 8]
        self.factorization = _factorizations()
        self.ntime = 25
        self.ref_time = torch.linspace(0, 1, self.ntime)

    def _times(self, nbatch, scale=1.0):
        return (
            self.ref_time.clone().unsqueeze(-1).expand(-1, nbatch).unsqueeze(-1) * scale
        )

    def test_linear_ode(self):
        for n, nbatch, nchunk, fact in itertools.product(
            self.n, self.nbatch, self.nchunk, self.factorization
        ):
            with self.subTest(n=n, nbatch=nbatch, nchunk=nchunk, fact=fact):
                model = ode.BackwardEulerODE(SimpleLinearSystem(n))
                y0 = model.ode.y0(nbatch)
                val1, val2, derivs1, derivs2 = _adjoint_vs_ad(
                    model,
                    list(model.parameters()),
                    y0,
                    self.ntime,
                    self._times(nbatch),
                    nchunk,
                    0,
                    fact,
                )
                self.assertTrue(torch.isclose(val1, val2))
                for p1, p2 in zip(derivs1, derivs2):
                    self.assertTrue(torch.allclose(p1, p2))

    def test_general_nonlinear(self):
        for n, nbatch, nchunk, fact in itertools.product(
            self.n, self.nbatch, self.nchunk, self.factorization
        ):
            with self.subTest(n=n, nbatch=nbatch, nchunk=nchunk, fact=fact):
                model = GeneralNonlinearEquation(n, 0.5)
                y0 = model.y0(nbatch)
                val1, val2, derivs1, derivs2 = _adjoint_vs_ad(
                    model,
                    list(model.parameters()),
                    y0,
                    self.ntime,
                    self._times(nbatch, scale=0.01),
                    nchunk,
                    0,
                    fact,
                )
                self.assertTrue(torch.isclose(val1, val2))
                for p1, p2 in zip(derivs1, derivs2):
                    self.assertTrue(torch.allclose(p1, p2, rtol=1e-4))
