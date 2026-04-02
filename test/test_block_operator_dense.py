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

"""Test dense block-operator algebra."""

import unittest

import torch

from pyzag.operators.dense import (
    DenseBlockLUFactorizedOperator,
    DenseBlockOperator,
    DenseBlockOperatorBuilder,
)

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)


class _DensePackedOperatorTestMixin:
    operator_cls = None

    def setUp(self):
        self.nblk = 7
        self.sbat = 5
        self.sblk = 6
        self.nrhs = 4

        A = torch.rand(self.nblk, self.sbat, self.sblk, self.sblk)
        B = torch.rand(self.nblk, self.sbat, self.sblk, self.sblk)

        I = torch.eye(self.sblk, dtype=A.dtype).reshape(1, 1, self.sblk, self.sblk)
        self.A_data = A + 2.0 * I
        self.B_data = B + 2.0 * I

        self.x = torch.rand(self.nblk, self.sbat, self.sblk)
        self.X = torch.rand(self.nblk, self.sbat, self.sblk, self.nrhs)

        self.A = self.operator_cls(self.A_data)
        self.B = self.operator_cls(self.B_data)

    def test_properties(self):
        self.assertEqual(self.A.device, self.A_data.device)
        self.assertEqual(self.A.dtype, self.A_data.dtype)
        self.assertEqual(self.A.nblk, self.nblk)
        self.assertEqual(self.A.batch_size, self.sbat)
        self.assertEqual(self.A.block_shape, (self.sblk, self.sblk))
        self.assertEqual(len(self.A), self.nblk)

    def test_matvec(self):
        one = torch.matmul(self.A_data, self.x.unsqueeze(-1)).squeeze(-1)
        two = self.A.matvec(self.x)
        self.assertTrue(torch.allclose(one, two))

    def test_t_matvec(self):
        one = torch.matmul(self.A_data.transpose(-1, -2), self.x.unsqueeze(-1)).squeeze(
            -1
        )
        two = self.A.t_matvec(self.x)
        self.assertTrue(torch.allclose(one, two))

    def test_solve(self):
        b = torch.matmul(self.A_data, self.x.unsqueeze(-1)).squeeze(-1)
        one = torch.linalg.solve(self.A_data, b.unsqueeze(-1)).squeeze(-1)
        two = self.A.solve(b)
        self.assertTrue(torch.allclose(one, two))
        self.assertTrue(torch.allclose(two, self.x))

    def test_clone(self):
        C = self.A.clone()
        self.assertTrue(torch.allclose(C.data, self.A.data))
        self.assertIsNot(C, self.A)
        self.assertIsNot(C.data, self.A.data)

    def test_block(self):
        blk = self.A.block(3)
        self.assertEqual(blk.nblk, 1)
        self.assertEqual(blk.batch_size, self.sbat)
        self.assertEqual(blk.block_shape, (self.sblk, self.sblk))
        self.assertTrue(torch.allclose(blk.data, self.A.data[3:4]))

    def test_window(self):
        win = self.A.window(2, 5)
        self.assertEqual(win.nblk, 3)
        self.assertEqual(win.batch_size, self.sbat)
        self.assertEqual(win.block_shape, (self.sblk, self.sblk))
        self.assertTrue(torch.allclose(win.data, self.A.data[2:5]))


class TestDenseBlockOperator(_DensePackedOperatorTestMixin, unittest.TestCase):
    operator_cls = DenseBlockOperator


class TestDenseBlockLUFactorizedOperator(
    _DensePackedOperatorTestMixin, unittest.TestCase
):
    operator_cls = DenseBlockLUFactorizedOperator


class TestDenseBlockOperatorViews(unittest.TestCase):
    def setUp(self):
        self.nblk = 7
        self.sbat = 5
        self.sblk = 6
        self.data = torch.rand(self.nblk, self.sbat, self.sblk, self.sblk)
        self.op = DenseBlockOperator(self.data)

    def test_pad_front(self):
        out = self.op.pad_front(2)
        self.assertEqual(out.nblk, self.nblk + 2)
        self.assertTrue(torch.allclose(out.data[2:], self.data))
        self.assertTrue(torch.allclose(out.data[:2], torch.zeros_like(out.data[:2])))

    def test_trim_front(self):
        out = self.op.trim_front(2)
        self.assertEqual(out.nblk, self.nblk - 2)
        self.assertTrue(torch.allclose(out.data, self.data[2:]))

    def test_update_window(self):
        repl_data = torch.rand(3, self.sbat, self.sblk, self.sblk)
        repl = DenseBlockOperator(repl_data)

        out = self.op.clone()
        out.update_window(2, 5, repl)

        expected = self.data.clone()
        expected[2:5] = repl_data
        self.assertTrue(torch.allclose(out.data, expected))


class TestDenseBlockLUFactorizedOperatorReduceBlock(unittest.TestCase):
    def setUp(self):
        self.nblk = 8
        self.sbat = 5
        self.sblk = 6

        A = torch.rand(self.nblk, self.sbat, self.sblk, self.sblk)
        B = torch.rand(self.nblk - 1, self.sbat, self.sblk, self.sblk)
        rhs = torch.rand(self.nblk, self.sbat, self.sblk)

        I = torch.eye(self.sblk, dtype=A.dtype).reshape(1, 1, self.sblk, self.sblk)
        self.A_data = A + 2.0 * I
        self.B_data = B
        self.rhs = rhs

        self.A = DenseBlockLUFactorizedOperator(self.A_data)
        self.B = DenseBlockOperator(self.B_data)

    def test_reduce_block_unpadded_B_shapes(self):
        B_red, rhs_red = self.A.reduce_block(self.B, self.rhs)
        self.assertIsInstance(B_red, DenseBlockOperator)
        self.assertEqual(B_red.nblk, self.nblk - 1)
        self.assertEqual(rhs_red.shape, (self.nblk - 1, self.sbat, self.sblk))

    def test_reduce_block_padded_B_shapes(self):
        B_pad = self.B.pad_front(1)
        B_red, rhs_red = self.A.reduce_block(B_pad, self.rhs)
        self.assertIsInstance(B_red, DenseBlockOperator)
        self.assertEqual(B_red.nblk, self.nblk - 1)
        self.assertEqual(rhs_red.shape, (self.nblk - 1, self.sbat, self.sblk))

    def test_reduce_block_wrong_rhs_nblk(self):
        with self.assertRaises(ValueError):
            self.A.reduce_block(self.B, self.rhs[:-1])

    def test_reduce_block_wrong_B_nblk(self):
        bad_B = DenseBlockOperator(self.B_data[:-1])
        with self.assertRaises(ValueError):
            self.A.reduce_block(bad_B, self.rhs)


class TestDenseBlockOperatorBuilder(unittest.TestCase):
    def setUp(self):
        self.nblk = 7
        self.sbat = 5
        self.sblk = 6
        self.builder = DenseBlockOperatorBuilder()

        self.J = torch.rand(2, self.nblk, self.sbat, self.sblk, self.sblk)

    def test_make_forward_blocks(self):
        A_ops, B_ops = self.builder.make_forward_blocks(self.J)

        self.assertIsInstance(A_ops, DenseBlockLUFactorizedOperator)
        self.assertIsInstance(B_ops, DenseBlockOperator)

        self.assertEqual(A_ops.nblk, self.nblk)
        self.assertEqual(B_ops.nblk, self.nblk - 1)

        self.assertTrue(torch.allclose(A_ops.data, self.J[1]))
        self.assertTrue(torch.allclose(B_ops.data, self.J[0, 1:]))

    def test_make_adjoint_blocks(self):
        A_ops, B_ops = self.builder.make_adjoint_blocks(self.J)

        self.assertIsInstance(A_ops, DenseBlockLUFactorizedOperator)
        self.assertIsInstance(B_ops, DenseBlockOperator)

        self.assertEqual(A_ops.nblk, self.nblk - 1)
        self.assertEqual(B_ops.nblk, self.nblk - 2)

        self.assertTrue(torch.allclose(A_ops.data, self.J[1, 1:].transpose(-1, -2)))
        self.assertTrue(torch.allclose(B_ops.data, self.J[0, 1:-1].transpose(-1, -2)))
