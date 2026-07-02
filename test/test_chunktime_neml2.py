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

"""Bidiagonal factorization correctness for the NEML2 backend.

Mirrors test_chunktime_dense.py: builds a bidiagonal block system on top of
NEML2SolvableBlockOperator and verifies Thomas / PCR / Hybrid produce identical
solutions, both for single-group and multi-group BLOCK+DENSE layouts.
"""

from __future__ import annotations

import unittest

import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)

from pyzag import chunktime
from pyzag.operators.neml2 import (
    AssembledMatrix,
    AxisLayout,
    NEML2BlockVector,
    NEML2SolvableBlockOperator,
    Tensor,
)


class TestNEML2ChunktimeSingleGroup(unittest.TestCase):
    """Single-group: should behave exactly like dense (which is well-tested)."""

    def setUp(self):
        self.sblk = 5
        self.sbat = 3

    def _build(self, nblk):
        layout = AxisLayout([["x"]], [[]], [[self.sblk]], [AxisLayout.IStructure.DENSE])
        I = torch.eye(self.sblk).reshape(1, 1, self.sblk, self.sblk)
        A_raw = torch.rand(nblk, self.sbat, self.sblk, self.sblk) + 2.0 * I
        B_raw = torch.rand(nblk - 1, self.sbat, self.sblk, self.sblk) * 0.1

        am_A = AssembledMatrix(layout, layout, [[Tensor(A_raw, 2, 0)]])
        am_B = AssembledMatrix(layout, layout, [[Tensor(B_raw, 2, 0)]])
        A_op = NEML2SolvableBlockOperator.factored(am_A)
        B_op = NEML2SolvableBlockOperator(am_B)
        rhs = NEML2BlockVector([torch.rand(nblk, self.sbat, self.sblk)], layout, [0])
        return A_op, B_op, rhs, A_raw, B_raw

    def _dense_reference(self, nblk, A_raw, B_raw, rhs):
        """Build the flat bidiagonal matrix and solve with torch.linalg."""
        n = nblk * self.sblk
        M = torch.zeros(self.sbat, n, n)
        for k in range(nblk):
            r = k * self.sblk
            M[:, r : r + self.sblk, r : r + self.sblk] = A_raw[k]
        for k in range(nblk - 1):
            r = (k + 1) * self.sblk
            M[:, r : r + self.sblk, r - self.sblk : r] = B_raw[k]
        rhs_flat = (
            rhs.raw_tensors[0].permute(1, 0, 2).reshape(self.sbat, -1).unsqueeze(-1)
        )
        out = torch.linalg.solve(M, rhs_flat).squeeze(-1)
        return out.reshape(self.sbat, nblk, self.sblk).permute(1, 0, 2)

    def test_thomas(self):
        nblk = 8
        A_op, B_op, rhs, A_raw, B_raw = self._build(nblk)
        expected = self._dense_reference(nblk, A_raw, B_raw, rhs)
        M = chunktime.BidiagonalThomasFactorization(A_op, B_op)
        result = M(rhs)
        self.assertTrue(torch.allclose(result.raw_tensors[0], expected, atol=1e-10))

    def test_pcr(self):
        nblk = 8  # power of 2
        A_op, B_op, rhs, A_raw, B_raw = self._build(nblk)
        expected = self._dense_reference(nblk, A_raw, B_raw, rhs)
        M = chunktime.BidiagonalPCRFactorization(A_op, B_op)
        result = M(rhs)
        self.assertTrue(torch.allclose(result.raw_tensors[0], expected, atol=1e-10))

    def test_thomas_vs_pcr_equivalence(self):
        """Thomas and PCR must produce identical results on the same input."""
        nblk = 8
        # Build the raw inputs once, then construct two pairs of operators on the
        # same underlying data (PCR/Thomas may mutate their cached factors).
        torch.manual_seed(42)
        A_op_t, B_op_t, rhs, A_raw, B_raw = self._build(nblk)

        # Reconstruct identical operators for the PCR run by re-wrapping the same
        # raw tensors (not re-rolling randoms).
        layout = AxisLayout([["x"]], [[]], [[self.sblk]], [AxisLayout.IStructure.DENSE])
        am_A = AssembledMatrix(layout, layout, [[Tensor(A_raw.clone(), 2, 0)]])
        am_B = AssembledMatrix(layout, layout, [[Tensor(B_raw.clone(), 2, 0)]])
        A_op_p = NEML2SolvableBlockOperator.factored(am_A)
        B_op_p = NEML2SolvableBlockOperator(am_B)

        thomas_result = chunktime.BidiagonalThomasFactorization(A_op_t, B_op_t)(rhs)
        pcr_result = chunktime.BidiagonalPCRFactorization(A_op_p, B_op_p)(rhs)

        self.assertTrue(
            torch.allclose(
                thomas_result.raw_tensors[0], pcr_result.raw_tensors[0], atol=1e-10
            )
        )


class TestNEML2ChunktimeMultiGroup(unittest.TestCase):
    """Multi-group BLOCK+DENSE: catches bugs that hide in single-group.

    Builds a 2-group bidiagonal system (BLOCK with intmd=3 + DENSE) with both
    the diagonal operator ``A`` and the sub-operator ``B`` block-diagonal in
    groups (per-group PCR's structural requirement). Verifies Thomas, PCR,
    and a flat reference all agree.

    Cross-block ``A`` (e.g. ``Aps``/``Asp`` for Taylor) breaks per-group PCR
    because each group's solve uses only ``Aii`` and ignores the
    cross-coupling. That case is exercised separately in
    ``test_neml2_pcr_structural.py`` (raises ``NotImplementedError``).
    """

    def setUp(self):
        # BLOCK group: 3 instances × base 4 (per-group flat = 12)
        self.intmd_p = 3
        self.base_p = 4
        # DENSE group: 7
        self.dim_s = 7
        self.sbat = 2
        self.layout = AxisLayout(
            [["a"], ["b"]],
            [[self.intmd_p], []],
            [[self.base_p], [self.dim_s]],
            [AxisLayout.IStructure.BLOCK, AxisLayout.IStructure.DENSE],
        )

    def _build(self, nblk, seed):
        """Build a multi-group bidiagonal system with block-diagonal A and B
        (so per-group PCR is structurally valid). Off-diagonal blocks (Aps,
        Asp, Bps, Bsp) are LEFT UNDEFINED so the test exercises the
        "undefined block → zero-fill at per-(i,j) shape" code path.
        """
        torch.manual_seed(seed)
        I_pp = torch.eye(self.base_p).reshape(1, 1, 1, self.base_p, self.base_p)
        A_pp = (
            torch.rand(nblk, self.sbat, self.intmd_p, self.base_p, self.base_p)
            + 2.0 * I_pp
        )
        I_ss = torch.eye(self.dim_s).reshape(1, 1, self.dim_s, self.dim_s)
        A_ss = torch.rand(nblk, self.sbat, self.dim_s, self.dim_s) + 2.0 * I_ss

        B_pp = (
            torch.rand(nblk - 1, self.sbat, self.intmd_p, self.base_p, self.base_p)
            * 0.1
        )
        B_ss = torch.rand(nblk - 1, self.sbat, self.dim_s, self.dim_s) * 0.1

        def _build_am(pp, ss):
            blocks = [
                [Tensor(pp, 2, 1), Tensor()],
                [Tensor(), Tensor(ss, 2, 0)],
            ]
            return AssembledMatrix(self.layout, self.layout, blocks)

        am_A = _build_am(A_pp, A_ss)
        am_B = _build_am(B_pp, B_ss)

        rhs = NEML2BlockVector(
            [
                torch.rand(nblk, self.sbat, self.intmd_p, self.base_p),
                torch.rand(nblk, self.sbat, self.dim_s),
            ],
            self.layout,
            [1, 0],
        )
        return am_A, am_B, rhs, (A_pp, A_ss), (B_pp, B_ss)

    def _flat_reference(self, nblk, A_blks, B_blks, rhs):
        """Build the equivalent flat bidiagonal matrix (BLOCK group expanded
        block-diagonally over its intmd dim) and solve with torch.linalg.
        Independent oracle for both Thomas and PCR results.
        """
        A_pp, A_ss = A_blks
        B_pp, B_ss = B_blks
        N = self.intmd_p
        np_ = self.base_p
        ns_ = self.dim_s
        n_flat = N * np_ + ns_

        def _embed(pp, ss):
            nb = pp.shape[0]
            M = torch.zeros(nb, self.sbat, n_flat, n_flat)
            for g in range(N):
                r = g * np_
                M[..., r : r + np_, r : r + np_] = pp[..., g, :, :]
            M[..., N * np_ :, N * np_ :] = ss
            return M

        M_A_blocks = _embed(A_pp, A_ss)
        M_B_blocks = _embed(B_pp, B_ss)

        n_total = nblk * n_flat
        big = torch.zeros(self.sbat, n_total, n_total)
        for k in range(nblk):
            r = k * n_flat
            big[:, r : r + n_flat, r : r + n_flat] = M_A_blocks[k]
        for k in range(nblk - 1):
            r = (k + 1) * n_flat
            big[:, r : r + n_flat, r - n_flat : r] = M_B_blocks[k]

        rhs_p = rhs.raw_tensors[0].reshape(nblk, self.sbat, N * np_)
        rhs_full = torch.cat([rhs_p, rhs.raw_tensors[1]], dim=-1)
        rhs_full = rhs_full.permute(1, 0, 2).reshape(self.sbat, n_total, 1)

        x_flat = torch.linalg.solve(big, rhs_full).squeeze(-1)
        x_flat = x_flat.reshape(self.sbat, nblk, n_flat).permute(1, 0, 2)
        x_p = x_flat[..., : N * np_].reshape(nblk, self.sbat, N, np_)
        x_s = x_flat[..., N * np_ :]
        return x_p, x_s

    def test_thomas_matches_flat_reference(self):
        nblk = 8
        am_A, am_B, rhs, A_blks, B_blks = self._build(nblk, seed=42)
        x_p_ref, x_s_ref = self._flat_reference(nblk, A_blks, B_blks, rhs)
        A_op = NEML2SolvableBlockOperator.factored(am_A)
        B_op = NEML2SolvableBlockOperator(am_B)
        out = chunktime.BidiagonalThomasFactorization(A_op, B_op)(rhs)
        self.assertTrue(torch.allclose(out.raw_tensors[0], x_p_ref, atol=1e-10))
        self.assertTrue(torch.allclose(out.raw_tensors[1], x_s_ref, atol=1e-10))

    def test_pcr_matches_flat_reference(self):
        nblk = 8  # power of 2
        am_A, am_B, rhs, A_blks, B_blks = self._build(nblk, seed=42)
        x_p_ref, x_s_ref = self._flat_reference(nblk, A_blks, B_blks, rhs)
        A_op = NEML2SolvableBlockOperator.factored(am_A)
        B_op = NEML2SolvableBlockOperator(am_B)
        out = chunktime.BidiagonalPCRFactorization(A_op, B_op)(rhs)
        self.assertTrue(torch.allclose(out.raw_tensors[0], x_p_ref, atol=1e-10))
        self.assertTrue(torch.allclose(out.raw_tensors[1], x_s_ref, atol=1e-10))

    def test_thomas_vs_pcr_equivalence(self):
        """Thomas and PCR are mathematically equivalent on systems where both
        algorithms are valid (block-diagonal A and B). This test catches:

        * ``pcr_init`` IndexError when ``B.am.col_layout`` had fewer groups
          than ``B.am.row_layout`` (caused by the old ``_extract_sublayout``
          collapsing to single-group DENSE)
        * Thomas's ``_mv_per_grain`` silently dropping the second input
          group when ``in_layout`` had fewer groups than the input vector

        Both bugs are invisible to single-group tests.
        """
        nblk = 8
        # Same seed → same raw tensors → same RHS.
        am_A_t, am_B_t, rhs_t, _, _ = self._build(nblk, seed=42)
        am_A_p, am_B_p, rhs_p, _, _ = self._build(nblk, seed=42)

        A_t = NEML2SolvableBlockOperator.factored(am_A_t)
        B_t = NEML2SolvableBlockOperator(am_B_t)
        thomas = chunktime.BidiagonalThomasFactorization(A_t, B_t)(rhs_t)

        A_p = NEML2SolvableBlockOperator.factored(am_A_p)
        B_p = NEML2SolvableBlockOperator(am_B_p)
        pcr = chunktime.BidiagonalPCRFactorization(A_p, B_p)(rhs_p)

        self.assertTrue(
            torch.allclose(thomas.raw_tensors[0], pcr.raw_tensors[0], atol=1e-10)
        )
        self.assertTrue(
            torch.allclose(thomas.raw_tensors[1], pcr.raw_tensors[1], atol=1e-10)
        )


class TestNEML2ChunktimeMultiGroupCrossBlock(unittest.TestCase):
    """Multi-group bidiagonal with CROSS-BLOCK A and BLOCK-DIAGONAL B.

    This is the Taylor mix-mode pattern: A has cross-group ``A_ps``/``A_sp``
    coupling (per-grain ↔ global from MixedControlSetup), but the sub-operator
    B is block-diagonal in groups (backward Euler gives ``B_pp = -I`` per
    grain, all other blocks zero).

    Success criterion: Multi-group PCR (Schur per-step + Schur-at-PCR-level)
    matches Multi-group Thomas, and both match Dense PCR run on the equivalent
    flat-materialized system. The Dense PCR result on the flat is the ground
    truth — it's the algorithm we trust because it operates on a single
    monolithic torch tensor.
    """

    def setUp(self):
        self.intmd_p = 3
        self.base_p = 4
        self.dim_s = 7
        self.sbat = 2
        self.layout = AxisLayout(
            [["a"], ["b"]],
            [[self.intmd_p], []],
            [[self.base_p], [self.dim_s]],
            [AxisLayout.IStructure.BLOCK, AxisLayout.IStructure.DENSE],
        )

    def _build(self, nblk, seed):
        """Build (am_A, am_B, rhs) with cross-block A and block-diag B.

        Matches Taylor mix-mode's structure: ``B_pp`` non-zero (per-grain
        backward-Euler ``-I``-style), ``B_ss = 0`` (algebraic vars not time-
        integrated). Only this specific shape of B preserves block-
        diagonality through PCR's ``-B A^{-1} B`` reduction:

            Z[g][h] = (A^{-1})[g][h] @ B[h][h]
            new B[g][h] = -B[g][g] @ Z[g][h]

        If only ``B_pp != 0`` then ``Z[*][s] = 0`` so all new B[*][s] = 0,
        and new B stays block-diagonal. With non-zero ``B_ss`` the invariant
        is broken and the multi-group PCR shortcut produces wrong answers.
        """
        torch.manual_seed(seed)
        I_pp = torch.eye(self.base_p).reshape(1, 1, 1, self.base_p, self.base_p)
        I_ss = torch.eye(self.dim_s).reshape(1, 1, self.dim_s, self.dim_s)
        # A with cross-blocks A_ps, A_sp (mix-mode coupling)
        A_pp = (
            torch.rand(nblk, self.sbat, self.intmd_p, self.base_p, self.base_p)
            + 3.0 * I_pp
        )
        A_ps = torch.rand(nblk, self.sbat, self.intmd_p, self.base_p, self.dim_s) * 0.1
        A_sp = torch.rand(nblk, self.sbat, self.intmd_p, self.dim_s, self.base_p) * 0.1
        A_ss = torch.rand(nblk, self.sbat, self.dim_s, self.dim_s) + 3.0 * I_ss
        # B with ONLY B_pp non-zero (Taylor's actual Jn pattern).
        B_pp = (
            torch.rand(nblk - 1, self.sbat, self.intmd_p, self.base_p, self.base_p)
            * 0.1
        )

        am_A = AssembledMatrix(
            self.layout,
            self.layout,
            [
                [Tensor(A_pp, 2, 1), Tensor(A_ps, 2, 1)],
                [Tensor(A_sp, 2, 1), Tensor(A_ss, 2, 0)],
            ],
        )
        am_B = AssembledMatrix(
            self.layout,
            self.layout,
            [
                [Tensor(B_pp, 2, 1), Tensor()],
                [Tensor(), Tensor()],  # B_ss undefined ≡ zero
            ],
        )
        rhs = NEML2BlockVector(
            [
                torch.rand(nblk, self.sbat, self.intmd_p, self.base_p),
                torch.rand(nblk, self.sbat, self.dim_s),
            ],
            self.layout,
            [1, 0],
        )
        return am_A, am_B, rhs, (A_pp, A_ps, A_sp, A_ss), (B_pp, None)

    def _flat_dense_pcr_reference(self, nblk, A_blks, B_blks, rhs):
        """Materialize the structured system to a flat (nblk, sbat, n, n)
        bidiagonal and solve via the existing Dense PCR. Ground truth."""
        from pyzag.operators.dense import DenseBlockOperator, DenseBlockVector

        A_pp, A_ps, A_sp, A_ss = A_blks
        B_pp, B_ss = B_blks
        N = self.intmd_p
        np_ = self.base_p
        ns_ = self.dim_s
        n_flat = N * np_ + ns_

        def _embed(pp, ps, sp, ss):
            nb = pp.shape[0]
            M = torch.zeros(nb, self.sbat, n_flat, n_flat)
            # BLOCK×BLOCK as block-diagonal over instances.
            for g in range(N):
                r = g * np_
                M[..., r : r + np_, r : r + np_] = pp[..., g, :, :]
            # BLOCK×DENSE (rows per-instance, cols global)
            if ps is not None:
                M[..., : N * np_, N * np_ :] = ps.reshape(nb, self.sbat, N * np_, ns_)
            # DENSE×BLOCK (rows global, cols per-instance)
            if sp is not None:
                permuted = sp.permute(0, 1, 3, 2, 4)
                M[..., N * np_ :, : N * np_] = permuted.reshape(
                    nb, self.sbat, ns_, N * np_
                )
            # DENSE×DENSE
            if ss is not None:
                M[..., N * np_ :, N * np_ :] = ss
            return M

        A_flat = _embed(A_pp, A_ps, A_sp, A_ss)  # (nblk, sbat, n, n)
        B_flat = _embed(B_pp, None, None, B_ss)  # (nblk-1, sbat, n, n)

        rhs_p = rhs.raw_tensors[0].reshape(nblk, self.sbat, N * np_)
        rhs_flat = torch.cat([rhs_p, rhs.raw_tensors[1]], dim=-1)  # (nblk, sbat, n)

        A_op = DenseBlockOperator.factored(A_flat)
        B_op = DenseBlockOperator(B_flat)
        rhs_bv = DenseBlockVector(rhs_flat)
        out = chunktime.BidiagonalPCRFactorization(A_op, B_op)(rhs_bv)

        # Unpack: (nblk, sbat, n_flat) → split into (..., N, np_) and (..., ns_)
        x_flat = out.data
        x_p = x_flat[..., : N * np_].reshape(nblk, self.sbat, N, np_)
        x_s = x_flat[..., N * np_ :]
        return x_p, x_s

    def test_mgpcr_matches_dense_pcr_on_flat(self):
        """SUCCESS CRITERION: multi-group PCR on the structured system matches
        Dense PCR on the flat-materialized equivalent."""
        nblk = 8
        am_A, am_B, rhs, A_blks, B_blks = self._build(nblk, seed=42)
        x_p_ref, x_s_ref = self._flat_dense_pcr_reference(nblk, A_blks, B_blks, rhs)

        A_op = NEML2SolvableBlockOperator.factored(am_A)
        B_op = NEML2SolvableBlockOperator(am_B)
        out = chunktime.BidiagonalPCRFactorization(A_op, B_op)(rhs)

        diff_p = (out.raw_tensors[0] - x_p_ref).abs().max().item()
        diff_s = (out.raw_tensors[1] - x_s_ref).abs().max().item()
        self.assertLess(diff_p, 1e-10, f"per-grain diff vs Dense-PCR ref: {diff_p}")
        self.assertLess(diff_s, 1e-10, f"global diff vs Dense-PCR ref: {diff_s}")

    def test_mgpcr_matches_thomas(self):
        """Multi-group PCR should also match multi-group Thomas (which uses
        the same per-step Schur solve)."""
        nblk = 8
        am_A_t, am_B_t, rhs_t, _, _ = self._build(nblk, seed=42)
        am_A_p, am_B_p, rhs_p, _, _ = self._build(nblk, seed=42)

        A_t = NEML2SolvableBlockOperator.factored(am_A_t)
        B_t = NEML2SolvableBlockOperator(am_B_t)
        thomas = chunktime.BidiagonalThomasFactorization(A_t, B_t)(rhs_t)

        A_p = NEML2SolvableBlockOperator.factored(am_A_p)
        B_p = NEML2SolvableBlockOperator(am_B_p)
        pcr = chunktime.BidiagonalPCRFactorization(A_p, B_p)(rhs_p)

        self.assertTrue(
            torch.allclose(thomas.raw_tensors[0], pcr.raw_tensors[0], atol=1e-10),
            f"Per-grain: max diff = "
            f"{(thomas.raw_tensors[0] - pcr.raw_tensors[0]).abs().max().item():.3e}",
        )
        self.assertTrue(
            torch.allclose(thomas.raw_tensors[1], pcr.raw_tensors[1], atol=1e-10),
            f"Global: max diff = "
            f"{(thomas.raw_tensors[1] - pcr.raw_tensors[1]).abs().max().item():.3e}",
        )


if __name__ == "__main__":
    unittest.main()
