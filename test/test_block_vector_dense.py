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

"""Test the dense block-vector implementation."""

from __future__ import annotations

import unittest

import torch

from pyzag.operators.dense import DenseBlockVector

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)


class TestDenseBlockVector(unittest.TestCase):
    def setUp(self):
        self.nblk = 5
        self.sbat = 3
        self.sblk = 4
        self.data = torch.rand(self.nblk, self.sbat, self.sblk)
        self.v = DenseBlockVector(self.data)

    def test_properties(self):
        self.assertEqual(self.v.nblk, self.nblk)
        self.assertEqual(self.v.batch_size, self.sbat)
        self.assertEqual(self.v.block_size, self.sblk)
        self.assertEqual(self.v.dtype, self.data.dtype)
        self.assertEqual(self.v.device, self.data.device)
        self.assertEqual(len(self.v), self.nblk)

    def test_clone(self):
        u = self.v.clone()
        self.assertIsInstance(u, DenseBlockVector)
        self.assertTrue(torch.allclose(u.data, self.v.data))
        self.assertIsNot(u.data, self.v.data)

    def test_norm(self):
        result = self.v.norm(dim=-1)
        expected = torch.norm(self.data, dim=-1)
        self.assertTrue(torch.allclose(result, expected))
        self.assertIsInstance(result, torch.Tensor)

    def test_flatten(self):
        result = self.v.flatten()
        expected = self.data.transpose(0, 1).flatten(1)
        self.assertEqual(result.shape, (self.sbat, self.nblk * self.sblk))
        self.assertTrue(torch.allclose(result, expected))

    def test_flatten_norm_is_cross_block_l2(self):
        # The per-batch convergence scalar used by the line search.
        result = self.v.flatten().norm(dim=-1)
        expected = torch.norm(self.data.transpose(0, 1).flatten(1), dim=-1)
        self.assertEqual(result.shape, (self.sbat,))
        self.assertTrue(torch.allclose(result, expected))

    def test_where_partial_mask(self):
        other_data = torch.rand_like(self.data)
        other = DenseBlockVector(other_data)
        mask = torch.tensor([True, False, True])
        result = self.v.where(mask, other)
        self.assertIsInstance(result, DenseBlockVector)
        # batches 0 and 2: take self; batch 1: take other
        self.assertTrue(torch.allclose(result.data[:, 0], self.data[:, 0]))
        self.assertTrue(torch.allclose(result.data[:, 1], other_data[:, 1]))
        self.assertTrue(torch.allclose(result.data[:, 2], self.data[:, 2]))

    def test_where_all_true_mask(self):
        other = DenseBlockVector(torch.rand_like(self.data))
        mask = torch.ones(self.sbat, dtype=torch.bool)
        result = self.v.where(mask, other)
        self.assertTrue(torch.allclose(result.data, self.data))

    def test_where_all_false_mask(self):
        other_data = torch.rand_like(self.data)
        other = DenseBlockVector(other_data)
        mask = torch.zeros(self.sbat, dtype=torch.bool)
        result = self.v.where(mask, other)
        self.assertTrue(torch.allclose(result.data, other_data))

    def test_scale_batches(self):
        factor = torch.tensor([0.5, 2.0, 3.0])
        result = self.v.scale_batches(factor)
        self.assertIsInstance(result, DenseBlockVector)
        for b in range(self.sbat):
            self.assertTrue(
                torch.allclose(result.data[:, b], self.data[:, b] * factor[b])
            )

    def test_scale_batches_identity(self):
        factor = torch.ones(self.sbat)
        result = self.v.scale_batches(factor)
        self.assertTrue(torch.allclose(result.data, self.data))

    def test_flip(self):
        u = self.v.flip(0)
        self.assertIsInstance(u, DenseBlockVector)
        self.assertTrue(torch.allclose(u.data, self.data.flip(0)))

    def test_neg(self):
        u = -self.v
        self.assertIsInstance(u, DenseBlockVector)
        self.assertTrue(torch.allclose(u.data, -self.data))

    def test_neg_method(self):
        u = self.v.neg()
        self.assertTrue(torch.allclose(u.data, -self.data))

    def test_add(self):
        other = DenseBlockVector(torch.rand_like(self.data))
        result = self.v + other
        self.assertIsInstance(result, DenseBlockVector)
        self.assertTrue(torch.allclose(result.data, self.data + other.data))

    def test_sub(self):
        other = DenseBlockVector(torch.rand_like(self.data))
        result = self.v - other
        self.assertIsInstance(result, DenseBlockVector)
        self.assertTrue(torch.allclose(result.data, self.data - other.data))

    def test_mul_scalar(self):
        result = self.v * 2.5
        self.assertIsInstance(result, DenseBlockVector)
        self.assertTrue(torch.allclose(result.data, self.data * 2.5))

    def test_rmul_scalar(self):
        result = 2.5 * self.v
        self.assertIsInstance(result, DenseBlockVector)
        self.assertTrue(torch.allclose(result.data, 2.5 * self.data))

    def test_getitem_slice(self):
        u = self.v[1:4]
        self.assertIsInstance(u, DenseBlockVector)
        self.assertEqual(u.nblk, 3)
        self.assertTrue(torch.allclose(u.data, self.data[1:4]))

    def test_getitem_single_slice(self):
        u = self.v[0:1]
        self.assertEqual(u.nblk, 1)
        self.assertTrue(torch.allclose(u.data, self.data[0:1]))

    def test_getitem_negative_slice(self):
        u = self.v[-1:]
        self.assertEqual(u.nblk, 1)
        self.assertTrue(torch.allclose(u.data, self.data[-1:]))

    def test_getitem_int(self):
        u = self.v[2]
        self.assertIsInstance(u, DenseBlockVector)
        self.assertEqual(u.nblk, 1)
        self.assertTrue(torch.allclose(u.data, self.data[2:3]))

    def test_setitem_slice(self):
        repl_data = torch.rand(2, self.sbat, self.sblk)
        repl = DenseBlockVector(repl_data)
        v = self.v.clone()
        v[1:3] = repl
        expected = self.data.clone()
        expected[1:3] = repl_data
        self.assertTrue(torch.allclose(v.data, expected))

    def test_cat(self):
        a = DenseBlockVector(torch.rand(2, self.sbat, self.sblk))
        b = DenseBlockVector(torch.rand(3, self.sbat, self.sblk))
        result = DenseBlockVector.cat([a, b], dim=0)
        self.assertIsInstance(result, DenseBlockVector)
        self.assertEqual(result.nblk, 5)
        self.assertTrue(torch.allclose(result.data, torch.cat([a.data, b.data], dim=0)))

    def test_zeros(self):
        v = DenseBlockVector.zeros(4, 3, 5, torch.float64, torch.device("cpu"))
        self.assertEqual(v.nblk, 4)
        self.assertEqual(v.batch_size, 3)
        self.assertEqual(v.block_size, 5)
        self.assertTrue(torch.allclose(v.data, torch.zeros(4, 3, 5)))

    def test_zeros_like(self):
        u = DenseBlockVector.zeros_like(self.v)
        self.assertEqual(u.nblk, self.v.nblk)
        self.assertEqual(u.batch_size, self.v.batch_size)
        self.assertEqual(u.block_size, self.v.block_size)
        self.assertTrue(torch.allclose(u.data, torch.zeros_like(self.data)))

    def test_empty(self):
        v = DenseBlockVector.empty(4, 3, 5, torch.float64, torch.device("cpu"))
        self.assertEqual(v.nblk, 4)
        self.assertEqual(v.batch_size, 3)
        self.assertEqual(v.block_size, 5)


if __name__ == "__main__":
    unittest.main()
