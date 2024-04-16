from pyzag import ode, nonlinear

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

        print(soln.shape)

        for i in range(self.n):
            soln += c0[:, i, None] * torch.exp(self.vals[i] * t) * self.vecs[:, i]

        return soln

    def y0(self, nbatch):
        return torch.outer(torch.linspace(-1, 1, nbatch), torch.linspace(1, 2, self.n))


class TestBackwardEulerTimeIntegrationLinear(unittest.TestCase):
    def setUp(self):
        self.n = 4
        self.model = ode.BackwardEulerODE(LinearSystem(self.n))

        self.nbatch = 5
        self.ntime = 100

        self.times = (
            torch.linspace(0, 1, self.ntime)
            .unsqueeze(-1)
            .expand(-1, self.nbatch)
            .unsqueeze(-1)
        )

    def test_integrate_forward(self):
        nchunk = 8

        for method in ["thomas", "pcr", "hybrid"]:
            solver = nonlinear.RecursiveNonlinearEquationSolver(
                self.model,
                self.model.ode.y0(self.nbatch),
                block_size=nchunk,
                direct_solve_method=method,
            )

            nres = solver.solve(self.ntime, self.times)
            eres = self.model.ode.exact(self.times)

            self.assertEqual(nres.shape, eres.shape)
            self.assertTrue(torch.allclose(nres, eres, rtol=1e-2))


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

        for method in ["thomas", "pcr", "hybrid"]:
            solver = nonlinear.RecursiveNonlinearEquationSolver(
                self.model, self.y0, block_size=nchunk, direct_solve_method=method
            )

            nres = solver.solve(self.ntime, self.times)
            eres = self.model.ode.exact(self.times)

            self.assertEqual(nres.shape, eres.shape)
            self.assertTrue(torch.allclose(nres, eres, rtol=1e-3))
