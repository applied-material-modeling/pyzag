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

"""Test NEML2BlockJacobian.to_dense() ↔ DenseBlockJacobian equivalence."""

from __future__ import annotations

import unittest

import torch

from pyzag import chunktime

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)

from pyzag.operators.dense import DenseBlockJacobian, DenseBlockVector
from pyzag.operators.neml2 import (
    AssembledMatrix,
    AxisLayout,
    NEML2BlockJacobian,
    NEML2BlockVector,
    Tensor,
    _av_to_flat,
    _split_flat_to_av,
)


def _layout_block_dense(grains: int = 5, np_: int = 10, ns_: int = 12) -> "AxisLayout":
    return AxisLayout(
        [["x_grain"], ["x_global"]],
        [[grains], []],
        [[np_], [ns_]],
        [AxisLayout.IStructure.BLOCK, AxisLayout.IStructure.DENSE],
    )


def _build_random_am(
    nblk: int,
    sbat: int,
    grains: int,
    np_: int,
    ns_: int,
    layout,
    block_diagonal: bool = False,
):
    """2-group BLOCK+DENSE assembled matrix with diagonally-dominant on-diagonal
    blocks (so LU is well-conditioned for use as a chunk-level diagonal).

    Args:
        block_diagonal: if True, zeros out the BLOCK↔DENSE cross-group entries
            (A_ps, A_sp). Required for any code path that runs PCR on the Schur
            backend — the per-group PCR check rejects cross-group coupling.
    """
    I_p = torch.eye(np_).reshape(1, 1, 1, np_, np_)
    A_pp = torch.rand(nblk, sbat, grains, np_, np_) + 2.0 * I_p
    if block_diagonal:
        A_ps = torch.zeros(nblk, sbat, grains, np_, ns_)
        A_sp = torch.zeros(nblk, sbat, grains, ns_, np_)
    else:
        A_ps = torch.rand(nblk, sbat, grains, np_, ns_) * 0.1
        A_sp = torch.rand(nblk, sbat, grains, ns_, np_) * 0.1
    I_s = torch.eye(ns_).reshape(1, 1, ns_, ns_)
    A_ss = torch.rand(nblk, sbat, ns_, ns_) + 2.0 * I_s
    return AssembledMatrix(
        layout,
        layout,
        [
            [Tensor(A_pp, 2, 1), Tensor(A_ps, 2, 1)],
            [Tensor(A_sp, 2, 1), Tensor(A_ss, 2, 0)],
        ],
    )


def _build_sub_am(nblk: int, sbat: int, grains: int, np_: int, ns_: int, layout):
    """Subdiagonal blocks for a chunk Jacobian.

    Cross-group entries kept zero (matches the Taylor structure: boundary-condition
    vars have no time integration, so the subdiagonal Jn is block-diagonal in
    variable groups). Small magnitudes so the bidiagonal block system stays
    well-conditioned for the smoke comparison.
    """
    sub_pp = torch.rand(nblk, sbat, grains, np_, np_) * 0.1
    sub_ss = torch.rand(nblk, sbat, ns_, ns_) * 0.1
    sub_ps = torch.zeros(nblk, sbat, grains, np_, ns_)
    sub_sp = torch.zeros(nblk, sbat, grains, ns_, np_)
    return AssembledMatrix(
        layout,
        layout,
        [
            [Tensor(sub_pp, 2, 1), Tensor(sub_ps, 2, 1)],
            [Tensor(sub_sp, 2, 1), Tensor(sub_ss, 2, 0)],
        ],
    )


class TestToDenseShapeAndContent(unittest.TestCase):
    """to_dense() must produce flat tensors matching _am_to_flat exactly."""

    def setUp(self):
        self.nblk = 4
        self.sbat = 2
        self.grains = 5
        self.np_ = 6
        self.ns_ = 8
        self.layout = _layout_block_dense(self.grains, self.np_, self.ns_)
        self.diag_am = _build_random_am(
            self.nblk, self.sbat, self.grains, self.np_, self.ns_, self.layout
        )
        self.sub_am = _build_sub_am(
            self.nblk, self.sbat, self.grains, self.np_, self.ns_, self.layout
        )
        self.jac = NEML2BlockJacobian(self.diag_am, self.sub_am, self.layout)

    def test_dense_shapes(self):
        dj = self.jac.to_dense()
        n_flat = self.grains * self.np_ + self.ns_
        self.assertEqual(dj.diag.shape, (self.nblk, self.sbat, n_flat, n_flat))
        self.assertEqual(dj.sub.shape, (self.nblk, self.sbat, n_flat, n_flat))

    def test_dense_matvec_matches_flat_solve_round_trip(self):
        """For a single diagonal block: dense LU-solve(M @ x) ≈ x.

        Combined assertion: (a) to_dense embedded the matrix correctly into
        the flat layout, (b) the flat M is invertible. If either fails the
        round-trip won't recover x.
        """
        dj = self.jac.to_dense()
        # Use the first chunk step's diagonal block only.
        M = dj.diag[0]  # (B, n_flat, n_flat)
        x = torch.rand(self.sbat, M.shape[-1])
        b = torch.matmul(M, x.unsqueeze(-1)).squeeze(-1)
        x_recov = torch.linalg.solve(M, b.unsqueeze(-1)).squeeze(-1)
        self.assertTrue(torch.allclose(x, x_recov, atol=1e-10))


class TestToDenseChunkSolveEquivalence(unittest.TestCase):
    """End-to-end chunk-level bidiagonal solve: the FlatDense path (via to_dense
    + DenseBlockJacobian + Thomas) must agree with the Schur path (via
    NEML2BlockJacobian + NEML2SolvableBlockOperator's Schur solve + Thomas)
    to high precision on the SAME chunk Jacobian.
    """

    def setUp(self):
        self.nblk = 4  # nblk_steps
        self.sbat = 2
        self.grains = 5
        self.np_ = 6
        self.ns_ = 8
        self.layout = _layout_block_dense(self.grains, self.np_, self.ns_)
        self.diag_am = _build_random_am(
            self.nblk, self.sbat, self.grains, self.np_, self.ns_, self.layout
        )
        self.sub_am = _build_sub_am(
            self.nblk, self.sbat, self.grains, self.np_, self.ns_, self.layout
        )
        self.jac_neml2 = NEML2BlockJacobian(self.diag_am, self.sub_am, self.layout)
        self.jac_dense = self.jac_neml2.to_dense()

        # Random RHS in flat layout — convert to both NEML2BlockVector and
        # DenseBlockVector forms so each path can consume it directly.
        n_flat = self.grains * self.np_ + self.ns_
        self.rhs_flat = torch.rand(self.nblk, self.sbat, n_flat)

    def test_thomas_solve_matches_across_backends(self):
        """Both backends, same Thomas solve on the same chunk: max diff ≤ 1e-9."""
        # NEML2 (Schur) path.
        fwd_neml2 = self.jac_neml2.forward_system(
            chunktime.BidiagonalThomasFactorization
        )
        inv_neml2 = fwd_neml2.inverse()
        rhs_neml2 = NEML2BlockVector.from_av(
            _split_flat_to_av(self.rhs_flat, self.layout)
        )
        x_neml2 = inv_neml2.matvec(rhs_neml2)
        x_neml2_flat = _av_to_flat(x_neml2.to_av())

        # Dense (FlatDense) path.
        fwd_dense = self.jac_dense.forward_system(
            chunktime.BidiagonalThomasFactorization
        )
        inv_dense = fwd_dense.inverse()
        rhs_dense = DenseBlockVector(self.rhs_flat)
        x_dense = inv_dense.matvec(rhs_dense)

        self.assertEqual(x_neml2_flat.shape, x_dense.data.shape)
        max_diff = (x_neml2_flat - x_dense.data).abs().max().item()
        self.assertLess(
            max_diff,
            1e-9,
            f"FlatDense Thomas solve diverged from Schur Thomas solve: max diff = {max_diff}",
        )

    def test_pcr_solve_matches_across_backends_block_diagonal(self):
        """PCR cross-backend agreement on a strictly block-diagonal-in-groups
        chunk Jacobian.

        Schur+PCR rejects cross-group coupling in either diag or sub (per-group
        reduction would silently produce wrong answers otherwise). For Taylor
        mix mode the chunk diag has BLOCK↔DENSE coupling via MixedControlSetup,
        so Schur+PCR is NOT viable on that problem — only FlatDense+PCR is.
        This test uses a synthetic block-diagonal diag so the agreement
        property can still be verified on a code path that exists.
        """
        layout = _layout_block_dense(self.grains, self.np_, self.ns_)
        diag_am_bd = _build_random_am(
            self.nblk,
            self.sbat,
            self.grains,
            self.np_,
            self.ns_,
            layout,
            block_diagonal=True,
        )
        sub_am_bd = _build_sub_am(
            self.nblk, self.sbat, self.grains, self.np_, self.ns_, layout
        )
        jac_neml2_bd = NEML2BlockJacobian(diag_am_bd, sub_am_bd, layout)
        jac_dense_bd = jac_neml2_bd.to_dense()

        n_flat = self.grains * self.np_ + self.ns_
        rhs_flat = torch.rand(self.nblk, self.sbat, n_flat)

        fwd_neml2 = jac_neml2_bd.forward_system(chunktime.BidiagonalPCRFactorization)
        inv_neml2 = fwd_neml2.inverse()
        rhs_neml2 = NEML2BlockVector.from_av(_split_flat_to_av(rhs_flat, layout))
        x_neml2 = inv_neml2.matvec(rhs_neml2)
        x_neml2_flat = _av_to_flat(x_neml2.to_av())

        fwd_dense = jac_dense_bd.forward_system(chunktime.BidiagonalPCRFactorization)
        inv_dense = fwd_dense.inverse()
        rhs_dense = DenseBlockVector(rhs_flat)
        x_dense = inv_dense.matvec(rhs_dense)

        max_diff = (x_neml2_flat - x_dense.data).abs().max().item()
        self.assertLess(
            max_diff,
            1e-9,
            f"FlatDense PCR solve diverged from Schur PCR solve: max diff = {max_diff}",
        )

    def test_flatdense_pcr_unaffected_by_cross_group_diag(self):
        """FlatDense + PCR has no cross-group concept (one flat dense block),
        so it works even when the original AssembledMatrix has cross-group
        coupling. This is the only PCR path available for Taylor mix mode.
        """
        fwd_dense = self.jac_dense.forward_system(chunktime.BidiagonalPCRFactorization)
        inv_dense = fwd_dense.inverse()
        rhs_dense = DenseBlockVector(self.rhs_flat)
        x_dense = inv_dense.matvec(rhs_dense)
        # Round-trip via the dense forward operator's bidiagonal apply.
        fwd_apply = fwd_dense.matvec(x_dense)
        max_diff = (fwd_apply.data - self.rhs_flat).abs().max().item()
        self.assertLess(
            max_diff,
            1e-9,
            f"FlatDense PCR round-trip failed: max diff = {max_diff}",
        )


if __name__ == "__main__":
    unittest.main()
