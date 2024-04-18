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

        self.assertEqual(steps, dsteps)

    def test_reverse(self):
        steps = [(i, j) for i, j in self.stepper(self.ntime).reverse()]
        should = [1]
        should += list(range(1, self.ntime, self.nchunk))[1:] + [self.ntime]
        dsteps = [
            (1 + self.ntime - i, 1 + self.ntime - j)
            for i, j in zip(should[::-1][:-1], should[::-1][1:])
        ]

        self.assertEqual(steps, dsteps)


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

        self.assertEqual(steps, dsteps)

    def test_reverse(self):
        steps = [(i, j) for i, j in self.stepper(self.ntime).reverse()]
        should = [1, 1 + self.offset]
        should += list(range(should[-1], self.ntime, self.nchunk))[1:] + [self.ntime]
        dsteps = [
            (1 + self.ntime - i, 1 + self.ntime - j)
            for i, j in zip(should[::-1][:-1], should[::-1][1:])
        ]

        self.assertEqual(steps, dsteps)
