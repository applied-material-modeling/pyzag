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

"""Test various time chunking step generators"""

from pyzag import nonlinear

import torch

# Ensure test consistency
torch.manual_seed(42)

import unittest

torch.set_default_dtype(torch.float64)


class TestBasicStepper(unittest.TestCase):
    def setUp(self):
        self.nchunk = 9
        self.ntime = 100
        self.stepper = nonlinear.StepGenerator(self.nchunk)

    def test_forward(self):
        steps = [(i, j) for i, j in self.stepper(self.ntime)]
        should = [1]
        should += list(range(1, self.ntime, self.nchunk))[1:] + [self.ntime]
        dsteps = [(i, j) for i, j in zip(should[:-1], should[1:])]

        self.assertEqual(steps[0][0], 1)
        self.assertEqual(steps[-1][1], self.ntime)
        self.assertEqual(steps, dsteps)

    def test_reverse(self):
        steps = [(i, j) for i, j in self.stepper(self.ntime).reverse()]
        rev = [
            (self.ntime - k2, self.ntime - k1) for k1, k2 in self.stepper(self.ntime)
        ][:-1]
        if rev[-1][0] != 1:
            rev += [(1, rev[-1][0])]

        self.assertEqual(steps[0][1], self.ntime - 1)
        self.assertEqual(steps[-1][0], 1)
        self.assertEqual(steps, rev)


class TestOffsetStepper(unittest.TestCase):
    def setUp(self):
        self.nchunk = 9
        self.ntime = 100
        self.offset = 4
        self.stepper = nonlinear.StepGenerator(self.nchunk, self.offset)

    def test_forward(self):
        steps = [(i, j) for i, j in self.stepper(self.ntime)]
        should = [1, 1 + self.offset]
        should += list(range(should[-1], self.ntime, self.nchunk))[1:] + [self.ntime]
        dsteps = [(i, j) for i, j in zip(should[:-1], should[1:])]

        self.assertEqual(steps[0][0], 1)
        self.assertEqual(steps[-1][1], self.ntime)
        self.assertEqual(steps, dsteps)

    def test_reverse(self):
        steps = [(i, j) for i, j in self.stepper(self.ntime).reverse()]
        rev = [
            (self.ntime - k2, self.ntime - k1) for k1, k2 in self.stepper(self.ntime)
        ][:-1]
        if rev[-1][0] != 1:
            rev += [(1, rev[-1][0])]

        self.assertEqual(steps[0][1], self.ntime - 1)
        self.assertEqual(steps[-1][0], 1)
        self.assertEqual(steps, rev)


class TestInitialOffsetStepper(unittest.TestCase):
    def setUp(self):
        self.nchunk = 9
        self.ntime = 100
        self.offset = 4
        self.stepper = nonlinear.InitialOffsetStepGenerator(self.nchunk, [1, 2, 4, 8])

    def test_reverse(self):
        steps = [(i, j) for i, j in self.stepper(self.ntime).reverse()]
        rev = [
            (self.ntime - k2, self.ntime - k1) for k1, k2 in self.stepper(self.ntime)
        ][:-1]
        if rev[-1][0] != 1:
            rev += [(1, rev[-1][0])]

        self.assertEqual(steps[0][1], self.ntime - 1)
        self.assertEqual(steps[-1][0], 1)
        self.assertEqual(steps, rev)
