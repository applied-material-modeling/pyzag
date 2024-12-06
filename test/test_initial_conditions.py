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

from pyzag import ode, nonlinear

import itertools

import torch

# Ensure test consistency
torch.manual_seed(42)

import unittest

torch.set_default_dtype(torch.float64)


class SimpleLinearSystem(torch.nn.Module):
    """Linear system of equations"""

    def __init__(self, n, make_parameter=True):
        super().__init__()
        self.n = n
        Ap = torch.rand((n, n))
        if make_parameter:
            self.A = torch.nn.Parameter(Ap.transpose(0, 1) * Ap)
        else:
            self.A = Ap.transpose(0, 1) * Ap

    def forward(self, t, y):
        return torch.matmul(self.A.unsqueeze(0).unsqueeze(0), y.unsqueeze(-1)).squeeze(
            -1
        ), self.A.unsqueeze(0).unsqueeze(0).expand(
            t.shape[0], t.shape[1], self.n, self.n
        )

    def y0(self, nbatch):
        return torch.outer(torch.linspace(-1, 1, nbatch), torch.linspace(1, 2, self.n))


class TestInitialConditionDerivatives(unittest.TestCase):
    def setUp(self):
        self.n = 3
        self.nbatch = 6
        self.nchunk = 5

        self.ode = SimpleLinearSystem
        self.method = ode.BackwardEulerODE

        self.ntime = 100

        self.ref_time = torch.linspace(0, 1, self.ntime)

    def test_only_parameters_ic(self):
        times = (
            self.ref_time.clone().unsqueeze(-1).expand(-1, self.nbatch).unsqueeze(-1)
        )
        model = self.method(self.ode(self.n))

        class Dummy(torch.nn.Module):
            def __init__(self, model, nbatch, nchunk):
                super().__init__()
                self.y0 = model.ode.y0(nbatch)
                self.solver = nonlinear.RecursiveNonlinearEquationSolver(
                    model,
                    step_generator=nonlinear.StepGenerator(nchunk),
                )

            def forward(self, ntime, times, adjoint=False):
                if adjoint:
                    nonlinear.solve_adjoint(self.solver, self.y0, ntime, times)
                return nonlinear.solve(self.solver, self.y0, ntime, times)

        dmodel = Dummy(model, self.nbatch, self.nchunk)

        res1 = dmodel.forward(self.ntime, times)
        val1 = torch.linalg.norm(res1)
        val1.backward()
        derivs1 = [p.grad for p in dmodel.parameters()]

        dmodel.zero_grad()
        res2 = dmodel.forward(self.ntime, times, adjoint=True)
        val2 = torch.linalg.norm(res2)
        val2.backward()
        derivs2 = [p.grad for p in dmodel.parameters()]

        self.assertEqual(len(derivs1), 1)
        self.assertEqual(len(derivs2), 1)

        self.assertTrue(torch.isclose(val1, val2))
        for p1, p2 in zip(derivs1, derivs2):
            self.assertTrue(torch.allclose(p1, p2))

    def test_mixed_parameters_ic(self):
        times = (
            self.ref_time.clone().unsqueeze(-1).expand(-1, self.nbatch).unsqueeze(-1)
        )
        model = self.method(self.ode(self.n))

        class Dummy(torch.nn.Module):
            def __init__(self, model, nbatch, nchunk):
                super().__init__()
                self.y0 = torch.nn.Parameter(model.ode.y0(nbatch))
                self.solver = nonlinear.RecursiveNonlinearEquationSolver(
                    model,
                    step_generator=nonlinear.StepGenerator(nchunk),
                )

            def forward(self, ntime, times, adjoint=False):
                if adjoint:
                    nonlinear.solve_adjoint(self.solver, self.y0, ntime, times)
                return nonlinear.solve(self.solver, self.y0, ntime, times)

        dmodel = Dummy(model, self.nbatch, self.nchunk)

        res1 = dmodel.forward(self.ntime, times)
        val1 = torch.linalg.norm(res1)
        val1.backward()
        derivs1 = [p.grad for p in dmodel.parameters()]

        dmodel.zero_grad()
        res2 = dmodel.forward(self.ntime, times, adjoint=True)
        val2 = torch.linalg.norm(res2)
        val2.backward()
        derivs2 = [p.grad for p in dmodel.parameters()]

        self.assertEqual(len(derivs1), 2)
        self.assertEqual(len(derivs2), 2)

        self.assertTrue(torch.isclose(val1, val2))
        for p1, p2 in zip(derivs1, derivs2):
            self.assertTrue(torch.allclose(p1, p2))

    def test_ic_only(self):
        times = (
            self.ref_time.clone().unsqueeze(-1).expand(-1, self.nbatch).unsqueeze(-1)
        )
        model = self.method(self.ode(self.n, make_parameter=False))

        class Dummy(torch.nn.Module):
            def __init__(self, model, nbatch, nchunk):
                super().__init__()
                self.y0 = torch.nn.Parameter(model.ode.y0(nbatch))
                self.solver = nonlinear.RecursiveNonlinearEquationSolver(
                    model,
                    step_generator=nonlinear.StepGenerator(nchunk),
                )

            def forward(self, ntime, times, adjoint=False):
                if adjoint:
                    nonlinear.solve_adjoint(self.solver, self.y0, ntime, times)
                return nonlinear.solve(self.solver, self.y0, ntime, times)

        dmodel = Dummy(model, self.nbatch, self.nchunk)

        res1 = dmodel.forward(self.ntime, times)
        val1 = torch.linalg.norm(res1)
        val1.backward()
        derivs1 = [p.grad for p in dmodel.parameters()]

        dmodel.zero_grad()
        res2 = dmodel.forward(self.ntime, times, adjoint=True)
        val2 = torch.linalg.norm(res2)
        val2.backward()
        derivs2 = [p.grad for p in dmodel.parameters()]

        self.assertEqual(len(derivs1), 1)
        self.assertEqual(len(derivs2), 1)

        self.assertTrue(torch.isclose(val1, val2))
        for p1, p2 in zip(derivs1, derivs2):
            self.assertTrue(torch.allclose(p1, p2))
