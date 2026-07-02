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

"""Test the NEML2 block-operator algebra (matvec, t_matvec, solve, Schur)."""

from __future__ import annotations

import unittest

import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)

from pyzag.operators.neml2 import (
    AssembledMatrix,
    AxisLayout,
    NEML2BlockVector,
    NEML2SolvableBlockOperator,
    Tensor,
    _am_to_flat,
    _av_to_flat,
    _split_flat_to_av,
    _transpose_am,
)


def _flat_solve_oracle(op, rhs):
    """Correctness oracle: materialize the full flat matrix and LU-solve.

    Relocated from ``NEML2SolvableBlockOperator`` (production no longer carries
    an oracle-only solve path). Independently cross-validates the Schur solve:
    materialization uses ``_am_to_flat`` (the same embedding that drives
    ``NEML2BlockJacobian.to_dense``). Supports the 2-group BLOCK+DENSE and
    1-group cases with a single intmd dim on the BLOCK group.
    """
    M = _am_to_flat(op.am)
    rhs_flat = _av_to_flat(rhs.to_av())
    x_flat = torch.linalg.solve(M, rhs_flat.unsqueeze(-1)).squeeze(-1)
    return NEML2BlockVector.from_av(_split_flat_to_av(x_flat, op.am.col_layout))


def _single_group_layout(block_size: int = 4) -> "AxisLayout":
    return AxisLayout([["x"]], [[]], [[block_size]], [AxisLayout.IStructure.DENSE])


def _multi_group_layout(grains: int = 5, np_: int = 10, ns_: int = 12) -> "AxisLayout":
    return AxisLayout(
        [["x_grain"], ["x_global"]],
        [[grains], []],
        [[np_], [ns_]],
        [AxisLayout.IStructure.BLOCK, AxisLayout.IStructure.DENSE],
    )


def _build_single_group_op(nblk: int, sbat: int, sblk: int):
    """Build a NEML2SolvableBlockOperator with a single DENSE group, diagonally-dominant
    so LU is well-conditioned."""
    layout = _single_group_layout(sblk)
    A_raw = torch.rand(nblk, sbat, sblk, sblk)
    I = torch.eye(sblk).reshape(1, 1, sblk, sblk)
    A_raw = A_raw + 2.0 * I
    # AssembledMatrix expects a 2D list of tensors indexed by [row group][col group].
    am = AssembledMatrix(
        layout,
        layout,
        [
            [Tensor(A_raw, 2, 0)]
        ],  # one row group, one col group; dynamic_dim=2, intmd_dim=0
    )
    return NEML2SolvableBlockOperator(am), A_raw


def _build_multi_group_op(
    nblk: int, sbat: int, grains: int = 5, np_: int = 6, ns_: int = 8
):
    """Build a 2-group BLOCK+DENSE NEML2SolvableBlockOperator (Taylor-like)."""
    layout = _multi_group_layout(grains, np_, ns_)
    # A_pp: (nblk, B, grains, np, np), diagonally dominant per-grain.
    I_p = torch.eye(np_).reshape(1, 1, 1, np_, np_)
    A_pp = torch.rand(nblk, sbat, grains, np_, np_) + 2.0 * I_p
    # A_ps: (nblk, B, grains, np, ns)
    A_ps = torch.rand(nblk, sbat, grains, np_, ns_) * 0.1
    # A_sp: (nblk, B, grains, ns, np)
    A_sp = torch.rand(nblk, sbat, grains, ns_, np_) * 0.1
    # A_ss: (nblk, B, ns, ns), diagonally dominant.
    I_s = torch.eye(ns_).reshape(1, 1, ns_, ns_)
    A_ss = torch.rand(nblk, sbat, ns_, ns_) + 2.0 * I_s
    am = AssembledMatrix(
        layout,
        layout,
        [
            [Tensor(A_pp, 2, 1), Tensor(A_ps, 2, 1)],
            [Tensor(A_sp, 2, 1), Tensor(A_ss, 2, 0)],
        ],
    )
    return NEML2SolvableBlockOperator(am), (A_pp, A_ps, A_sp, A_ss)


class TestNEML2SolvableBlockOperatorSingleGroup(unittest.TestCase):
    """Single-group operator — should behave identically to dense."""

    def setUp(self):
        self.nblk = 7
        self.sbat = 5
        self.sblk = 6
        self.op, self.A_raw = _build_single_group_op(self.nblk, self.sbat, self.sblk)

        self.x_data = torch.rand(self.nblk, self.sbat, self.sblk)
        layout = _single_group_layout(self.sblk)
        self.x = NEML2BlockVector([self.x_data], layout, [0])

    def test_properties(self):
        self.assertEqual(self.op.nblk, self.nblk)
        self.assertEqual(self.op.batch_size, self.sbat)
        self.assertEqual(self.op.dtype, self.A_raw.dtype)
        self.assertEqual(self.op.device, self.A_raw.device)

    def test_matvec(self):
        expected = torch.matmul(self.A_raw, self.x_data.unsqueeze(-1)).squeeze(-1)
        result = self.op.matvec(self.x)
        self.assertTrue(torch.allclose(result.raw_tensors[0], expected, atol=1e-10))

    def test_t_matvec(self):
        expected = torch.matmul(
            self.A_raw.transpose(-1, -2), self.x_data.unsqueeze(-1)
        ).squeeze(-1)
        result = self.op.t_matvec(self.x)
        self.assertTrue(torch.allclose(result.raw_tensors[0], expected, atol=1e-10))

    def test_solve(self):
        # b = A @ x; verify solve(b) recovers x.
        b_data = torch.matmul(self.A_raw, self.x_data.unsqueeze(-1)).squeeze(-1)
        layout = _single_group_layout(self.sblk)
        b = NEML2BlockVector([b_data], layout, [0])
        result = self.op.solve(b)
        self.assertTrue(torch.allclose(result.raw_tensors[0], self.x_data, atol=1e-10))


class TestNEML2SolvableBlockOperatorMultiGroup(unittest.TestCase):
    """Multi-group BLOCK+DENSE — exercises Schur complement."""

    def setUp(self):
        self.nblk = 4
        self.sbat = 2
        self.grains = 5
        self.np_ = 6
        self.ns_ = 8
        self.op, self.A_blocks = _build_multi_group_op(
            self.nblk, self.sbat, self.grains, self.np_, self.ns_
        )
        self.A_pp, self.A_ps, self.A_sp, self.A_ss = self.A_blocks
        self.layout = _multi_group_layout(self.grains, self.np_, self.ns_)

        # Build a random x and compute b = A @ x using the per-block matmul
        # with intmd_sum semantics — that's our reference for solve.
        self.x_p = torch.rand(self.nblk, self.sbat, self.grains, self.np_)
        self.x_s = torch.rand(self.nblk, self.sbat, self.ns_)
        self.x = NEML2BlockVector([self.x_p, self.x_s], self.layout, [1, 0])

    def _reference_matvec(self, x_p, x_s):
        """Reference 'per-grain' matvec interpretation: each grain's row gets only
        its own grain's contribution (no homogenization). This represents the
        mathematical structure of a per-grain Jacobian; Schur is built around the
        same convention.

        NEML2's native ``am * av`` does something different for BLOCK-col contractions
        (intmd_sum + broadcast back), which is the right semantic for the Schur
        complement formation (Steps 3-4) but not for a "regular" per-grain matvec.
        The round-trip test ``test_matvec_then_solve_recovers_x`` verifies whether
        Schur inverts NEML2 native matvec; if it doesn't, that's a real concern
        for chunked Thomas on multi-group models (chunk_size > 1).
        """
        # b_p = A_pp @ x_p (per-grain) + A_ps @ x_s (broadcast from global)
        bpp = torch.matmul(self.A_pp, x_p.unsqueeze(-1)).squeeze(-1)
        x_s_bc = x_s.unsqueeze(-2)
        bps = torch.matmul(self.A_ps, x_s_bc.unsqueeze(-1)).squeeze(-1)
        b_p = bpp + bps
        # b_s = Σ_grain(A_sp[grain] @ x_p[grain]) + A_ss @ x_s
        bsp_per_grain = torch.matmul(self.A_sp, x_p.unsqueeze(-1)).squeeze(-1)
        bsp = bsp_per_grain.sum(dim=-2)
        bss = torch.matmul(self.A_ss, x_s.unsqueeze(-1)).squeeze(-1)
        b_s = bsp + bss
        return b_p, b_s

    def test_matvec_then_solve_recovers_x(self):
        """matvec(x) = b ; solve(b) should recover x.

        This is the most useful self-consistency check for the operator —
        whatever NEML2's am*av and our Schur solve do, they must be inverse
        operations. (Avoids hardcoding a 'reference' matvec interpretation
        that may not match NEML2's intmd_sum + broadcast semantics for the
        Schur algorithm's intermediate matvecs.)"""
        b = self.op.matvec(self.x)
        x_recovered = self.op.solve(b)
        self.assertTrue(torch.allclose(x_recovered.raw_tensors[0], self.x_p, atol=1e-8))
        self.assertTrue(torch.allclose(x_recovered.raw_tensors[1], self.x_s, atol=1e-8))

    def test_solve_schur_recovers_x(self):
        """Build b = A @ x via reference matvec, solve via Schur, verify x recovered."""
        b_p, b_s = self._reference_matvec(self.x_p, self.x_s)
        b = NEML2BlockVector([b_p, b_s], self.layout, [1, 0])
        result = self.op.solve(b)
        self.assertTrue(torch.allclose(result.raw_tensors[0], self.x_p, atol=1e-8))
        self.assertTrue(torch.allclose(result.raw_tensors[1], self.x_s, atol=1e-8))

    def test_flat_solve_oracle_recovers_x(self):
        """The _flat_solve oracle should also recover x — independent verification."""
        b_p, b_s = self._reference_matvec(self.x_p, self.x_s)
        b = NEML2BlockVector([b_p, b_s], self.layout, [1, 0])
        result = _flat_solve_oracle(self.op, b)
        self.assertTrue(torch.allclose(result.raw_tensors[0], self.x_p, atol=1e-8))
        self.assertTrue(torch.allclose(result.raw_tensors[1], self.x_s, atol=1e-8))

    def test_schur_matches_flat_solve(self):
        """Cross-check: Schur and flat solve produce the same answer."""
        b_p, b_s = self._reference_matvec(self.x_p, self.x_s)
        b = NEML2BlockVector([b_p, b_s], self.layout, [1, 0])
        schur = self.op.solve(b)
        flat = _flat_solve_oracle(self.op, b)
        self.assertTrue(
            torch.allclose(schur.raw_tensors[0], flat.raw_tensors[0], atol=1e-9)
        )
        self.assertTrue(
            torch.allclose(schur.raw_tensors[1], flat.raw_tensors[1], atol=1e-9)
        )

    def test_t_matvec_via_inner_product_identity(self):
        """For any operator A and vectors x, y: <A@x, y> = <x, A^T@y>.

        This identity holds regardless of how A or its transpose are
        represented internally — so it cleanly verifies that t_matvec is
        consistent with matvec without depending on a hand-rolled reference
        for the BLOCK×BLOCK matmul semantics.
        """
        # y has same structure as the COL layout (input to matvec).
        # For matvec: A @ x -> result with ROW layout.
        # For t_matvec: A^T @ y where y has ROW layout -> result with COL layout.
        # Then <A @ x, y> = <x, A^T @ y>.
        # Here col_layout == row_layout, so x and y have the same structure.
        y_p = torch.rand(self.nblk, self.sbat, self.grains, self.np_)
        y_s = torch.rand(self.nblk, self.sbat, self.ns_)
        y = NEML2BlockVector([y_p, y_s], self.layout, [1, 0])

        Ax = self.op.matvec(self.x)
        ATy = self.op.t_matvec(y)

        # Inner products: flatten each group fully and dot.
        # <Ax, y> = sum(Ax_p * y_p) + sum(Ax_s * y_s)
        lhs = (Ax.raw_tensors[0] * y_p).sum() + (Ax.raw_tensors[1] * y_s).sum()
        # <x, A^T y> = sum(x_p * ATy_p) + sum(x_s * ATy_s)
        rhs = (self.x_p * ATy.raw_tensors[0]).sum() + (
            self.x_s * ATy.raw_tensors[1]
        ).sum()
        self.assertTrue(
            torch.allclose(lhs, rhs, atol=1e-8),
            f"Inner-product identity failed: <Ax,y>={lhs} vs <x,A^Ty>={rhs}",
        )

    def test_transpose_am_roundtrip(self):
        """_transpose_am twice should give back the original AssembledMatrix."""
        TT = _transpose_am(_transpose_am(self.op.am))
        for i in range(2):
            for j in range(2):
                orig = self.op.am.tensors[i][j].torch()
                back = TT.tensors[i][j].torch()
                self.assertTrue(torch.allclose(orig, back, atol=0))


if __name__ == "__main__":
    unittest.main()
