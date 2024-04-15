from pyzag import ode

import torch

import unittest

torch.set_default_dtype(torch.float64)


class VanderPolODE(torch.nn.Module):
    """
    From Manichev et al 2019

      x0 = [-1, 1]
      t = [0, 4.2*mu)
    """

    def __init__(self, mu):
        super().__init__()
        self.mu = torch.tensor(mu)

    def forward(self, t, y):
        f = torch.empty(y.shape)
        f[..., 0] = y[..., 1]
        f[..., 1] = -y[..., 0] + self.mu * (1.0 - y[..., 0] ** 2.0) * y[..., 1]

        df = torch.empty(y.shape + y.shape[-1:])
        df[..., 0, 0] = 0
        df[..., 0, 1] = 1
        df[..., 1, 0] = -1 - 2.0 * self.mu * y[..., 0] * y[..., 1]
        df[..., 1, 1] = self.mu * (1.0 - y[..., 0] ** 2.0)

        return f, df


class TestBackwardEulerTimeIntegration(unittest.TestCase):
    def setUp(self):
        self.mu = 1.0  # Not stiff
        self.model = ode.BackwardEulerODE(VanderPolODE(self.mu))

        self.nbatch = 5
        self.ntime = 100

        self.times = (
            torch.linspace(0, 1, self.ntime).unsqueeze(-1).expand(-1, self.nbatch)
        )
        self.y = torch.rand((self.ntime, self.nbatch, 2))

    def test_shapes(self):
        nchunk = 8
        R, J = self.model(
            self.y[: nchunk + self.model.lookback],
            self.times[: nchunk + self.model.lookback],
        )

        self.assertEquals(R.shape, (nchunk, self.nbatch, 2))
        self.assertEquals(J.shape, (1 + self.model.lookback, nchunk, self.nbatch, 2, 2))
