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


class _LowerBidiagonalSolveMixin:
    diag_operator_cls = None

    def setUp(self):
        self.nblk = 7
        self.sbat = 5
        self.sblk = 6

        A = torch.rand(self.nblk, self.sbat, self.sblk, self.sblk)
        B = torch.rand(self.nblk - 1, self.sbat, self.sblk, self.sblk)

        I = torch.eye(self.sblk, dtype=A.dtype).reshape(1, 1, self.sblk, self.sblk)
        self.A_data = A + 2.0 * I
        self.B_data = B
        self.x = torch.rand(self.nblk, self.sbat, self.sblk)

        self.A = self.diag_operator_cls(self.A_data)
        self.B = DenseBlockOperator(self.B_data)

    def _make_lower_bidiagonal_rhs(self, A_data, B_data, x):
        rhs = torch.matmul(A_data, x.unsqueeze(-1)).squeeze(-1)
        rhs[1:] = rhs[1:] + torch.matmul(B_data, x[:-1].unsqueeze(-1)).squeeze(-1)
        return rhs

    def test_solve_lower_bidiagonal(self):
        rhs = self._make_lower_bidiagonal_rhs(self.A_data, self.B_data, self.x)
        y = self.A.solve_lower_bidiagonal(self.B, rhs)
        self.assertTrue(torch.allclose(y, self.x))

    def test_solve_lower_bidiagonal_wrong_nblk(self):
        rhs = self._make_lower_bidiagonal_rhs(self.A_data, self.B_data, self.x)
        bad_B = DenseBlockOperator(self.B_data[:-1])
        with self.assertRaises((ValueError, RuntimeError, IndexError)):
            self.A.solve_lower_bidiagonal(bad_B, rhs)


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


class TestDenseBlockOperatorLowerBidiagonal(
    _LowerBidiagonalSolveMixin, unittest.TestCase
):
    diag_operator_cls = DenseBlockOperator


class TestDenseBlockLUFactorizedOperatorLowerBidiagonal(
    _LowerBidiagonalSolveMixin, unittest.TestCase
):
    diag_operator_cls = DenseBlockLUFactorizedOperator


class TestLowerBidiagonalSolveEquivalence(unittest.TestCase):
    def test_dense_and_lu_factorized_match(self):
        nblk, sbat, sblk = 7, 5, 6

        A = torch.rand(nblk, sbat, sblk, sblk)
        B = torch.rand(nblk - 1, sbat, sblk, sblk)
        I = torch.eye(sblk, dtype=A.dtype).reshape(1, 1, sblk, sblk)

        A_data = A + 2.0 * I
        B_data = B
        x = torch.rand(nblk, sbat, sblk)

        rhs = torch.matmul(A_data, x.unsqueeze(-1)).squeeze(-1)
        rhs[1:] = rhs[1:] + torch.matmul(B_data, x[:-1].unsqueeze(-1)).squeeze(-1)

        A_dense = DenseBlockOperator(A_data)
        A_lu = DenseBlockLUFactorizedOperator(A_data)
        B_op = DenseBlockOperator(B_data)

        y_dense = A_dense.solve_lower_bidiagonal(B_op, rhs)
        y_lu = A_lu.solve_lower_bidiagonal(B_op, rhs)

        self.assertTrue(torch.allclose(y_dense, x))
        self.assertTrue(torch.allclose(y_lu, x))
        self.assertTrue(torch.allclose(y_dense, y_lu))
