"""Test the reparameterization functions"""

from pyzag import reparametrization

import torch

# Ensure test consistency
torch.manual_seed(42)

import unittest

torch.set_default_dtype(torch.float64)


class TestRangeRescale(unittest.TestCase):
    def setUp(self):
        self.lb = torch.tensor(1.5)
        self.ub = torch.tensor(5.0)
        self.fn = reparametrization.RangeRescale(self.lb, self.ub)

    def test_forward(self):
        y = torch.tensor([0.2, 0.7, 0.5])
        self.assertTrue(torch.allclose(self.fn(y), y * (self.ub - self.lb) + self.lb))

    def test_backward(self):
        y = torch.tensor([0.2, 0.7, 0.5])
        self.assertTrue(torch.allclose(self.fn.reverse(self.fn(y)), y))
