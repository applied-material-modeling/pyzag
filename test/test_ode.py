from pyzag import ode, nonlinear

import torch

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

    def exact(self, t, y0):
        return (
            self.K
            * torch.exp(self.r * t)
            * y0[None, ...]
            / (self.K + (torch.exp(self.r * t) - 1) * y0[None, ...])
        )


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

        self.y0 = torch.linspace(0, 1, self.nbatch).reshape(self.nbatch, 1)

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
        solver = nonlinear.RecursiveNonlinearEquationSolver(
            self.model, self.y0, block_size=nchunk
        )
        nres = solver.solve(self.ntime, self.times)
        eres = self.model.ode.exact(self.times, self.y0)

        self.assertEqual(nres.shape, eres.shape)
        self.assertTrue(torch.allclose(nres, eres, rtol=1e-3))
