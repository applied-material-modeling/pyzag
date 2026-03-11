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
    DenseBlockOperator,
    DenseBlockLUFactorizedOperator,
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

    def test_matmat(self):
        one = torch.matmul(self.A_data, self.X)
        two = self.A.matmat(self.X)
        self.assertTrue(torch.allclose(one, two))

    def test_t_matmat(self):
        one = torch.matmul(self.A_data.transpose(-1, -2), self.X)
        two = self.A.t_matmat(self.X)
        self.assertTrue(torch.allclose(one, two))

    def test_solve(self):
        b = torch.matmul(self.A_data, self.x.unsqueeze(-1)).squeeze(-1)
        one = torch.linalg.solve(self.A_data, b.unsqueeze(-1)).squeeze(-1)
        two = self.A.solve(b)
        self.assertTrue(torch.allclose(one, two))
        self.assertTrue(torch.allclose(two, self.x))

    def test_t_solve(self):
        b = torch.matmul(self.A_data.transpose(-1, -2), self.x.unsqueeze(-1)).squeeze(
            -1
        )
        one = torch.linalg.solve(
            self.A_data.transpose(-1, -2), b.unsqueeze(-1)
        ).squeeze(-1)
        two = self.A.t_solve(b)
        self.assertTrue(torch.allclose(one, two))
        self.assertTrue(torch.allclose(two, self.x))

    def test_solve_mat(self):
        B_rhs = torch.matmul(self.A_data, self.X)
        one = torch.linalg.solve(self.A_data, B_rhs)
        two = self.A.solve_mat(B_rhs)
        self.assertTrue(torch.allclose(one, two))
        self.assertTrue(torch.allclose(two, self.X))

    def test_t_solve_mat(self):
        B_rhs = torch.matmul(self.A_data.transpose(-1, -2), self.X)
        one = torch.linalg.solve(self.A_data.transpose(-1, -2), B_rhs)
        two = self.A.t_solve_mat(B_rhs)
        self.assertTrue(torch.allclose(one, two))
        self.assertTrue(torch.allclose(two, self.X))

    def test_compose(self):
        C = self.A.compose(self.B)
        one = torch.matmul(self.A_data, self.B_data)
        self.assertIsInstance(C, DenseBlockOperator)
        self.assertTrue(torch.allclose(C.data, one))

    def test_add(self):
        C = self.A.add(self.B)
        self.assertIsInstance(C, DenseBlockOperator)
        self.assertTrue(torch.allclose(C.data, self.A_data + self.B_data))

    def test_sub(self):
        C = self.A.sub(self.B)
        self.assertIsInstance(C, DenseBlockOperator)
        self.assertTrue(torch.allclose(C.data, self.A_data - self.B_data))

    def test_neg(self):
        C = self.A.neg()
        self.assertIsInstance(C, DenseBlockOperator)
        self.assertTrue(torch.allclose(C.data, -self.A_data))

    def test_inv_compose(self):
        C = self.A.inv_compose(self.B)
        one = torch.linalg.solve(self.A_data, self.B_data)
        self.assertIsInstance(C, DenseBlockOperator)
        self.assertTrue(torch.allclose(C.data, one))

    def test_t_inv_compose(self):
        C = self.A.t_inv_compose(self.B)
        one = torch.linalg.solve(self.A_data.transpose(-1, -2), self.B_data)
        self.assertIsInstance(C, DenseBlockOperator)
        self.assertTrue(torch.allclose(C.data, one))

    def test_clone(self):
        C = self.A.clone()
        self.assertTrue(torch.allclose(C.data, self.A.data))
        self.assertIsNot(C, self.A)
        self.assertIsNot(C.data, self.A.data)

    def test_slice_blocks(self):
        C = self.A.slice_blocks(1, 5, 2)
        self.assertEqual(C.nblk, 2)
        self.assertTrue(torch.allclose(C.data, self.A_data[1:5:2]))

    def test_empty_like(self):
        C = self.A.empty_like(3)
        self.assertEqual(C.nblk, 3)
        self.assertEqual(C.batch_size, self.sbat)
        self.assertEqual(C.block_shape, (self.sblk, self.sblk))
        self.assertEqual(C.dtype, self.A.dtype)
        self.assertEqual(C.device, self.A.device)
        self.assertEqual(C.data.shape, (3, self.sbat, self.sblk, self.sblk))


class TestDenseBlockOperator(_DensePackedOperatorTestMixin, unittest.TestCase):
    operator_cls = DenseBlockOperator


class TestDenseBlockLUFactorizedOperator(
    _DensePackedOperatorTestMixin, unittest.TestCase
):
    operator_cls = DenseBlockLUFactorizedOperator


class TestDenseBlockOperatorBuilder(unittest.TestCase):
    def setUp(self):
        self.nblk = 7
        self.sbat = 5
        self.sblk = 6
        self.builder = DenseBlockOperatorBuilder()

        self.J = torch.rand(2, self.nblk, self.sbat, self.sblk, self.sblk)

    def test_make_forward_blocks(self):
        A_ops, B_ops = self.builder.make_forward_blocks(self.J)

        self.assertIsInstance(A_ops, DenseBlockOperator)
        self.assertIsInstance(B_ops, DenseBlockOperator)

        self.assertEqual(A_ops.nblk, self.nblk)
        self.assertEqual(B_ops.nblk, self.nblk - 1)

        self.assertTrue(torch.allclose(A_ops.data, self.J[1]))
        self.assertTrue(torch.allclose(B_ops.data, self.J[0, 1:]))

    def test_make_adjoint_blocks(self):
        A_ops, B_ops = self.builder.make_adjoint_blocks(self.J)

        self.assertIsInstance(A_ops, DenseBlockOperator)
        self.assertIsInstance(B_ops, DenseBlockOperator)

        self.assertEqual(A_ops.nblk, self.nblk - 1)
        self.assertEqual(B_ops.nblk, self.nblk - 2)

        self.assertTrue(torch.allclose(A_ops.data, self.J[1, 1:].transpose(-1, -2)))
        self.assertTrue(torch.allclose(B_ops.data, self.J[0, 1:-1].transpose(-1, -2)))
