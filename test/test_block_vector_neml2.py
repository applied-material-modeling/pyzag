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

"""Test the NEML2 block-vector implementation.

Mirrors test_block_vector_dense.py but for the multi-group NEML2 backend.
Uses both a single-group (DENSE) layout for direct parity with dense, and a
multi-group (BLOCK + DENSE) layout to exercise per-group arithmetic.
"""

from __future__ import annotations

import unittest

import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)

from pyzag.operators.neml2 import AxisLayout, NEML2BlockVector


def _single_group_layout(block_size: int = 4) -> "AxisLayout":
    """Single-group DENSE layout with one scalar-ish variable of given size."""
    return AxisLayout([["x"]], [[]], [[block_size]], [AxisLayout.IStructure.DENSE])


def _multi_group_layout(grains: int = 5, np_: int = 10, ns_: int = 12) -> "AxisLayout":
    """Two-group BLOCK+DENSE layout: group 0 per-grain (intmd=grains, base=np),
    group 1 global (DENSE, base=ns)."""
    return AxisLayout(
        [["x_grain"], ["x_global"]],
        [[grains], []],
        [[np_], [ns_]],
        [AxisLayout.IStructure.BLOCK, AxisLayout.IStructure.DENSE],
    )


class TestNEML2BlockVectorSingleGroup(unittest.TestCase):
    """Mirrors TestDenseBlockVector with a single-group DENSE layout —
    should behave identically to the dense backend."""

    def setUp(self):
        self.nblk = 5
        self.sbat = 3
        self.sblk = 4
        self.layout = _single_group_layout(self.sblk)
        self.data = torch.rand(self.nblk, self.sbat, self.sblk)
        self.v = NEML2BlockVector([self.data], self.layout, [0])

    def test_properties(self):
        self.assertEqual(self.v.nblk, self.nblk)
        self.assertEqual(self.v.batch_size, self.sbat)
        self.assertEqual(self.v.block_size, self.sblk)
        self.assertEqual(self.v.dtype, self.data.dtype)
        self.assertEqual(self.v.device, self.data.device)
        self.assertEqual(len(self.v), self.nblk)

    def test_clone(self):
        u = self.v.clone()
        self.assertIsInstance(u, NEML2BlockVector)
        self.assertTrue(torch.allclose(u.raw_tensors[0], self.v.raw_tensors[0]))
        self.assertIsNot(u.raw_tensors[0], self.v.raw_tensors[0])

    def test_norm(self):
        # Single group -> norm should equal the per-block L2.
        result = self.v.norm(dim=-1)
        expected = torch.norm(self.data, dim=-1)
        self.assertTrue(torch.allclose(result, expected))

    def test_flat_norm(self):
        result = self.v.flat_norm()
        expected = torch.norm(self.data.transpose(0, 1).flatten(1), dim=-1)
        self.assertEqual(result.shape, (self.sbat,))
        self.assertTrue(torch.allclose(result, expected))

    def test_where_partial_mask(self):
        other_data = torch.rand_like(self.data)
        other = NEML2BlockVector([other_data], self.layout, [0])
        mask = torch.tensor([True, False, True])
        result = self.v.where(mask, other)
        self.assertTrue(torch.allclose(result.raw_tensors[0][:, 0], self.data[:, 0]))
        self.assertTrue(torch.allclose(result.raw_tensors[0][:, 1], other_data[:, 1]))
        self.assertTrue(torch.allclose(result.raw_tensors[0][:, 2], self.data[:, 2]))

    def test_scale_batches(self):
        factor = torch.tensor([0.5, 2.0, 3.0])
        result = self.v.scale_batches(factor)
        for b in range(self.sbat):
            self.assertTrue(
                torch.allclose(result.raw_tensors[0][:, b], self.data[:, b] * factor[b])
            )

    def test_flip(self):
        u = self.v.flip(0)
        self.assertTrue(torch.allclose(u.raw_tensors[0], self.data.flip(0)))

    def test_neg(self):
        u = -self.v
        self.assertTrue(torch.allclose(u.raw_tensors[0], -self.data))

    def test_add(self):
        other = NEML2BlockVector([torch.rand_like(self.data)], self.layout, [0])
        result = self.v + other
        self.assertTrue(
            torch.allclose(result.raw_tensors[0], self.data + other.raw_tensors[0])
        )

    def test_sub(self):
        other = NEML2BlockVector([torch.rand_like(self.data)], self.layout, [0])
        result = self.v - other
        self.assertTrue(
            torch.allclose(result.raw_tensors[0], self.data - other.raw_tensors[0])
        )

    def test_mul_scalar(self):
        result = self.v * 2.5
        self.assertTrue(torch.allclose(result.raw_tensors[0], self.data * 2.5))

    def test_rmul_scalar(self):
        result = 2.5 * self.v
        self.assertTrue(torch.allclose(result.raw_tensors[0], 2.5 * self.data))

    def test_getitem_slice(self):
        u = self.v[1:4]
        self.assertEqual(u.nblk, 3)
        self.assertTrue(torch.allclose(u.raw_tensors[0], self.data[1:4]))

    def test_getitem_int(self):
        u = self.v[2]
        self.assertEqual(u.nblk, 1)
        self.assertTrue(torch.allclose(u.raw_tensors[0], self.data[2:3]))

    def test_setitem_slice(self):
        repl_data = torch.rand(2, self.sbat, self.sblk)
        repl = NEML2BlockVector([repl_data], self.layout, [0])
        v = self.v.clone()
        v[1:3] = repl
        expected = self.data.clone()
        expected[1:3] = repl_data
        self.assertTrue(torch.allclose(v.raw_tensors[0], expected))

    def test_cat(self):
        a = NEML2BlockVector([torch.rand(2, self.sbat, self.sblk)], self.layout, [0])
        b = NEML2BlockVector([torch.rand(3, self.sbat, self.sblk)], self.layout, [0])
        result = NEML2BlockVector.cat([a, b], dim=0)
        self.assertEqual(result.nblk, 5)
        self.assertTrue(
            torch.allclose(
                result.raw_tensors[0],
                torch.cat([a.raw_tensors[0], b.raw_tensors[0]], dim=0),
            )
        )

    def test_zeros_not_in_interface(self):
        # The shape-only zeros() constructor was removed from the BlockVector
        # interface (it cannot express the multi-group layout from a scalar
        # block_size). Use zeros_with_layout / zeros_like instead.
        self.assertFalse(hasattr(NEML2BlockVector, "zeros"))
        self.assertFalse(hasattr(NEML2BlockVector, "empty"))

    def test_zeros_with_layout(self):
        v = NEML2BlockVector.zeros_with_layout(
            4, 3, self.layout, torch.float64, torch.device("cpu")
        )
        self.assertEqual(v.nblk, 4)
        self.assertEqual(v.batch_size, 3)
        self.assertEqual(v.block_size, self.sblk)
        self.assertTrue(torch.allclose(v.raw_tensors[0], torch.zeros(4, 3, self.sblk)))

    def test_zeros_like(self):
        u = NEML2BlockVector.zeros_like(self.v)
        self.assertEqual(u.nblk, self.v.nblk)
        self.assertTrue(torch.allclose(u.raw_tensors[0], torch.zeros_like(self.data)))


class TestNEML2BlockVectorMultiGroup(unittest.TestCase):
    """Multi-group (BLOCK + DENSE) — exercises the per-group arithmetic
    and max-across-groups norm reduction."""

    def setUp(self):
        self.nblk = 4
        self.sbat = 2
        self.grains = 5
        self.np_ = 10
        self.ns_ = 12
        self.layout = _multi_group_layout(self.grains, self.np_, self.ns_)
        # Group 0: BLOCK per-grain (nblk, B, grains, np)
        self.data_p = torch.rand(self.nblk, self.sbat, self.grains, self.np_)
        # Group 1: DENSE global (nblk, B, ns)
        self.data_s = torch.rand(self.nblk, self.sbat, self.ns_)
        self.v = NEML2BlockVector([self.data_p, self.data_s], self.layout, [1, 0])

    def test_properties(self):
        self.assertEqual(self.v.nblk, self.nblk)
        self.assertEqual(self.v.batch_size, self.sbat)
        # block_size = sum(intmd × base) = 5*10 + 12 = 62
        self.assertEqual(self.v.block_size, self.grains * self.np_ + self.ns_)

    def test_clone(self):
        u = self.v.clone()
        self.assertIsNot(u.raw_tensors[0], self.v.raw_tensors[0])
        self.assertIsNot(u.raw_tensors[1], self.v.raw_tensors[1])
        self.assertTrue(torch.allclose(u.raw_tensors[0], self.v.raw_tensors[0]))
        self.assertTrue(torch.allclose(u.raw_tensors[1], self.v.raw_tensors[1]))

    def test_add(self):
        other_p = torch.rand_like(self.data_p)
        other_s = torch.rand_like(self.data_s)
        other = NEML2BlockVector([other_p, other_s], self.layout, [1, 0])
        result = self.v + other
        self.assertTrue(torch.allclose(result.raw_tensors[0], self.data_p + other_p))
        self.assertTrue(torch.allclose(result.raw_tensors[1], self.data_s + other_s))

    def test_neg(self):
        u = -self.v
        self.assertTrue(torch.allclose(u.raw_tensors[0], -self.data_p))
        self.assertTrue(torch.allclose(u.raw_tensors[1], -self.data_s))

    def test_norm_combined_l2_across_groups(self):
        result = self.v.norm(dim=-1)
        # Combined L2 over the whole (multi-group) state: sqrt(sum_g ||g||^2),
        # matching the dense backend and NEML2's own residual norm.
        p_flat = self.data_p.flatten(start_dim=-2)  # (nblk, B, grains*np)
        s_flat = self.data_s  # (nblk, B, ns)
        n_p = torch.norm(p_flat, dim=-1)
        n_s = torch.norm(s_flat, dim=-1)
        expected = torch.stack([n_p**2, n_s**2], dim=0).sum(dim=0).sqrt()
        self.assertTrue(torch.allclose(result, expected))

    def test_flat_norm_per_batch(self):
        result = self.v.flat_norm()
        self.assertEqual(result.shape, (self.sbat,))
        # Combined L2 across groups: per-group flatten + norm, then sqrt of the
        # sum of squares.
        n_p = torch.norm(self.data_p.transpose(0, 1).flatten(1), dim=-1)
        n_s = torch.norm(self.data_s.transpose(0, 1).flatten(1), dim=-1)
        expected = torch.stack([n_p**2, n_s**2], dim=0).sum(dim=0).sqrt()
        self.assertTrue(torch.allclose(result, expected))

    def test_where_broadcasts_over_groups(self):
        other_p = torch.rand_like(self.data_p)
        other_s = torch.rand_like(self.data_s)
        other = NEML2BlockVector([other_p, other_s], self.layout, [1, 0])
        mask = torch.tensor([True, False])
        result = self.v.where(mask, other)
        self.assertTrue(torch.allclose(result.raw_tensors[0][:, 0], self.data_p[:, 0]))
        self.assertTrue(torch.allclose(result.raw_tensors[0][:, 1], other_p[:, 1]))
        self.assertTrue(torch.allclose(result.raw_tensors[1][:, 0], self.data_s[:, 0]))
        self.assertTrue(torch.allclose(result.raw_tensors[1][:, 1], other_s[:, 1]))

    def test_to_av_round_trip(self):
        # to_av -> from_av should preserve data byte-for-byte.
        av = self.v.to_av()
        v2 = NEML2BlockVector.from_av(av)
        self.assertTrue(torch.allclose(v2.raw_tensors[0], self.v.raw_tensors[0]))
        self.assertTrue(torch.allclose(v2.raw_tensors[1], self.v.raw_tensors[1]))
        self.assertEqual(v2.intmd_dims, self.v.intmd_dims)

    def test_zeros_with_layout_multi_group(self):
        v = NEML2BlockVector.zeros_with_layout(
            self.nblk, self.sbat, self.layout, torch.float64, torch.device("cpu")
        )
        self.assertEqual(
            v.raw_tensors[0].shape, (self.nblk, self.sbat, self.grains, self.np_)
        )
        self.assertEqual(v.raw_tensors[1].shape, (self.nblk, self.sbat, self.ns_))
        self.assertTrue((v.raw_tensors[0] == 0).all())
        self.assertTrue((v.raw_tensors[1] == 0).all())


if __name__ == "__main__":
    unittest.main()
