"""Test the various predictor strategies"""

from pyzag import nonlinear

import torch

# Ensure test consistency
torch.manual_seed(42)

import unittest

torch.set_default_dtype(torch.float64)


class BasePredictor(unittest.TestCase):
    def setUp(self):
        self.ntime = 100
        self.nbatch = 10
        self.nstate = 3
        self.data = torch.rand((self.ntime, self.nbatch, self.nstate))


class TestFullTrajectoryPredictor(BasePredictor):
    def setUp(self):
        super().setUp()
        self.predictor = nonlinear.FullTrajectoryPredictor(self.data)

    def test_prediction(self):
        k = 21
        dk = 10
        self.assertTrue(
            torch.allclose(
                self.predictor.predict(self.data, k, dk),
                self.data[k : k + dk],
            )
        )


class TestZeroPredictor(BasePredictor):
    def setUp(self):
        self.predictor = nonlinear.ZeroPredictor()
        super().setUp()

    def test_prediction(self):
        k = 21
        dk = 10
        self.assertTrue(
            torch.allclose(
                self.predictor.predict(self.data, k, dk),
                torch.zeros_like(self.data[k - dk : k]),
            )
        )


class TestPreviousStepsPredictor(BasePredictor):
    def setUp(self):
        self.predictor = nonlinear.PreviousStepsPredictor()
        super().setUp()

    def test_prediction(self):
        k = 21
        dk = 10
        self.assertTrue(
            torch.allclose(
                self.predictor.predict(self.data, k, dk),
                self.data[k - dk : k],
            )
        )

    def test_prediction_not_enough_steps(self):
        k = 5
        dk = 10
        self.assertTrue(
            torch.allclose(
                self.predictor.predict(self.data, k, dk),
                torch.zeros((dk, self.nbatch, self.nstate)),
            )
        )
