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
