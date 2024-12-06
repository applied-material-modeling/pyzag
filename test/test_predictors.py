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
