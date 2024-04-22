from pyzag import ode, nonlinear, chunktime

import torch

# Ensure test consistency
torch.manual_seed(42)

import unittest

torch.set_default_dtype(torch.float64)


class LinearSystem(torch.nn.Module):
    """Linear system of equations"""

    def __init__(self, n):
        super().__init__()
        self.n = n
        Ap = torch.rand((n, n))
        self.A = torch.nn.Parameter(Ap.transpose(0, 1) * Ap)

        self.vals, self.vecs = torch.linalg.eigh(self.A)

    def forward(self, t, y):
        return torch.matmul(self.A.unsqueeze(0).unsqueeze(0), y.unsqueeze(-1)).squeeze(
            -1
        ), self.A.unsqueeze(0).unsqueeze(0).expand(
            t.shape[0], t.shape[1], self.n, self.n
        )

    def y0(self, nbatch):
        return torch.outer(torch.linspace(-1, 1, nbatch), torch.linspace(1, 2, self.n))


class TestSimpleAdjointLinearBackward(unittest.TestCase):
    def setUp(self):
        self.n = 4
        self.model = ode.BackwardEulerODE(LinearSystem(self.n))

        self.nbatch = 5
        self.ntime = 100

        self.nchunk = 6

        self.times = (
            torch.linspace(0, 1, self.ntime)
            .unsqueeze(-1)
            .expand(-1, self.nbatch)
            .unsqueeze(-1)
        )

        self.y0 = self.model.ode.y0(self.nbatch)

        self.solver = nonlinear.RecursiveNonlinearEquationSolver(
            self.model,
            self.y0,
            step_generator=nonlinear.StepGenerator(self.nchunk),
        )

    def test_parameters(self):
        self.assertEqual(1, len(list(self.model.parameters())))

    def test_adjoint(self):
        res1 = self.solver.solve(self.ntime, self.times)
        val1 = torch.linalg.norm(res1)
        val1.backward()
        deriv1 = self.solver.func.ode.A.grad

        self.solver.zero_grad()
        res2 = nonlinear.solve_adjoint(self.solver, self.ntime, self.times)
        val2 = torch.linalg.norm(res2)
        val2.backward()
        deriv2 = self.solver.func.ode.A.grad

        self.assertTrue(torch.isclose(val1, val2))
        self.assertTrue(torch.allclose(deriv1, deriv2))


class TestSimpleAdjointLinearForward(unittest.TestCase):
    def setUp(self):
        self.n = 4
        self.model = ode.ForwardEulerODE(LinearSystem(self.n))

        self.nbatch = 5
        self.ntime = 100

        self.nchunk = 6

        self.times = (
            torch.linspace(0, 1, self.ntime)
            .unsqueeze(-1)
            .expand(-1, self.nbatch)
            .unsqueeze(-1)
        )

        self.y0 = self.model.ode.y0(self.nbatch)

        self.solver = nonlinear.RecursiveNonlinearEquationSolver(
            self.model,
            self.y0,
            step_generator=nonlinear.StepGenerator(self.nchunk),
        )

    def test_parameters(self):
        self.assertEqual(1, len(list(self.model.parameters())))

    def test_adjoint(self):
        res1 = self.solver.solve(self.ntime, self.times)
        val1 = torch.linalg.norm(res1)
        val1.backward()
        deriv1 = self.solver.func.ode.A.grad

        self.solver.zero_grad()
        res2 = nonlinear.solve_adjoint(self.solver, self.ntime, self.times)
        val2 = torch.linalg.norm(res2)
        val2.backward()
        deriv2 = self.solver.func.ode.A.grad

        self.assertTrue(torch.isclose(val1, val2))
        self.assertTrue(torch.allclose(deriv1, deriv2))
