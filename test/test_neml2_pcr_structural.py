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

"""Tests for the PCR structural-check enforcement and PCR correctness on
multi-group NEML2 systems.

PCR is only valid when the subdiagonal Jn is block-diagonal in variable groups.
The runtime check in NEML2SolvableBlockOperator.pcr_init must fire when this
assumption is violated (otherwise PCR silently produces wrong answers).
"""

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
)


class TestPCRStructuralCheck(unittest.TestCase):
    """Verify pcr_init raises if Jn has cross-group entries."""

    def test_raises_on_cross_group_jn(self):
        # Build a 2-group system with a non-zero cross-group block in the subdiagonal.
        nblk = 4
        sbat = 2
        np_ = 3
        ns_ = 4

        layout = AxisLayout(
            [["x_p"], ["x_s"]],
            [[], []],
            [[np_], [ns_]],
            [AxisLayout.IStructure.DENSE, AxisLayout.IStructure.DENSE],
        )

        # Diagonal Jacobian: valid block-diagonal-in-groups structure.
        I_p = torch.eye(np_).reshape(1, 1, np_, np_)
        I_s = torch.eye(ns_).reshape(1, 1, ns_, ns_)
        A_pp = torch.rand(nblk, sbat, np_, np_) + 2.0 * I_p
        A_ss = torch.rand(nblk, sbat, ns_, ns_) + 2.0 * I_s
        # Off-diagonal blocks (for forward solve, but irrelevant to PCR structural check).
        A_ps = torch.zeros(nblk, sbat, np_, ns_)
        A_sp = torch.zeros(nblk, sbat, ns_, np_)
        am_diag = AssembledMatrix(
            layout,
            layout,
            [
                [Tensor(A_pp, 2, 0), Tensor(A_ps, 2, 0)],
                [Tensor(A_sp, 2, 0), Tensor(A_ss, 2, 0)],
            ],
        )

        # Subdiagonal Jn: introduce a NON-ZERO cross-group entry.
        Jn_pp = (
            -torch.eye(np_)
            .reshape(1, 1, np_, np_)
            .expand(nblk, sbat, np_, np_)
            .contiguous()
        )
        Jn_ss = (
            -torch.eye(ns_)
            .reshape(1, 1, ns_, ns_)
            .expand(nblk, sbat, ns_, ns_)
            .contiguous()
        )
        Jn_ps_bad = (
            torch.rand(nblk, sbat, np_, ns_) * 0.5
        )  # ← non-zero, should trigger raise
        Jn_sp_zero = torch.zeros(nblk, sbat, ns_, np_)
        am_sub = AssembledMatrix(
            layout,
            layout,
            [
                [Tensor(Jn_pp, 2, 0), Tensor(Jn_ps_bad, 2, 0)],
                [Tensor(Jn_sp_zero, 2, 0), Tensor(Jn_ss, 2, 0)],
            ],
        )

        op_diag = NEML2SolvableBlockOperator(am_diag)
        op_sub = NEML2SolvableBlockOperator(am_sub)

        v = NEML2BlockVector(
            [torch.rand(nblk, sbat, np_), torch.rand(nblk, sbat, ns_)],
            layout,
            [0, 0],
        )

        with self.assertRaisesRegex(
            NotImplementedError, "block-diagonal in variable groups"
        ):
            op_diag.pcr_init(op_sub, v)

    def test_cross_group_diag_dispatches_to_multigroup(self):
        """Cross-block A (diagonal operator) no longer raises — dispatches
        to multi-group PCR which flat-materializes A and B then delegates
        to Dense PCR. The structural check now applies only to the
        per-group fast path; multi-group PCR handles cross-block A
        correctly via flat-dense delegation."""
        nblk = 4
        sbat = 2
        np_ = 3
        ns_ = 4

        layout = AxisLayout(
            [["x_p"], ["x_s"]],
            [[], []],
            [[np_], [ns_]],
            [AxisLayout.IStructure.DENSE, AxisLayout.IStructure.DENSE],
        )

        I_p = torch.eye(np_).reshape(1, 1, np_, np_)
        I_s = torch.eye(ns_).reshape(1, 1, ns_, ns_)
        A_pp = torch.rand(nblk, sbat, np_, np_) + 2.0 * I_p
        A_ss = torch.rand(nblk, sbat, ns_, ns_) + 2.0 * I_s
        # A has NON-ZERO cross-group entries (mix-mode-like coupling).
        A_ps = torch.rand(nblk, sbat, np_, ns_) * 0.1
        A_sp = torch.rand(nblk, sbat, ns_, np_) * 0.1
        am_diag = AssembledMatrix(
            layout,
            layout,
            [
                [Tensor(A_pp, 2, 0), Tensor(A_ps, 2, 0)],
                [Tensor(A_sp, 2, 0), Tensor(A_ss, 2, 0)],
            ],
        )

        # B is block-diagonal in groups (still required structure).
        Jn_pp = (
            -torch.eye(np_)
            .reshape(1, 1, np_, np_)
            .expand(nblk, sbat, np_, np_)
            .contiguous()
        )
        am_sub = AssembledMatrix(
            layout,
            layout,
            [
                [Tensor(Jn_pp, 2, 0), Tensor()],
                [Tensor(), Tensor()],
            ],
        )

        op_diag = NEML2SolvableBlockOperator(am_diag)
        op_sub = NEML2SolvableBlockOperator(am_sub)
        v = NEML2BlockVector(
            [torch.rand(nblk, sbat, np_), torch.rand(nblk, sbat, ns_)],
            layout,
            [0, 0],
        )

        # Should NOT raise — dispatches to multi-group PCR (flat-dense).
        state = op_diag.pcr_init(op_sub, v)
        self.assertIsNotNone(state)

    def test_passes_with_block_diagonal_jn(self):
        """Sanity: with proper block-diagonal Jn, pcr_init does NOT raise."""
        nblk = 8  # power of 2 for clean PCR (nblk_level = log2(nblk))
        sbat = 2
        np_ = 3
        ns_ = 4

        layout = AxisLayout(
            [["x_p"], ["x_s"]],
            [[], []],
            [[np_], [ns_]],
            [AxisLayout.IStructure.DENSE, AxisLayout.IStructure.DENSE],
        )

        I_p = torch.eye(np_).reshape(1, 1, np_, np_)
        I_s = torch.eye(ns_).reshape(1, 1, ns_, ns_)
        A_pp = torch.rand(nblk, sbat, np_, np_) + 2.0 * I_p
        A_ss = torch.rand(nblk, sbat, ns_, ns_) + 2.0 * I_s
        # Provide non-trivial diag blocks but zero off-diag (PCR will only touch diag).
        A_ps = torch.zeros(nblk, sbat, np_, ns_)
        A_sp = torch.zeros(nblk, sbat, ns_, np_)
        am_diag = AssembledMatrix(
            layout,
            layout,
            [
                [Tensor(A_pp, 2, 0), Tensor(A_ps, 2, 0)],
                [Tensor(A_sp, 2, 0), Tensor(A_ss, 2, 0)],
            ],
        )
        Jn_pp = (
            -torch.eye(np_)
            .reshape(1, 1, np_, np_)
            .expand(nblk, sbat, np_, np_)
            .contiguous()
        )
        Jn_ss = (
            -torch.eye(ns_)
            .reshape(1, 1, ns_, ns_)
            .expand(nblk, sbat, ns_, ns_)
            .contiguous()
        )
        # Cross-group blocks are zero — structural check should pass.
        Jn_ps = torch.zeros(nblk, sbat, np_, ns_)
        Jn_sp = torch.zeros(nblk, sbat, ns_, np_)
        am_sub = AssembledMatrix(
            layout,
            layout,
            [
                [Tensor(Jn_pp, 2, 0), Tensor(Jn_ps, 2, 0)],
                [Tensor(Jn_sp, 2, 0), Tensor(Jn_ss, 2, 0)],
            ],
        )

        op_diag = NEML2SolvableBlockOperator(am_diag)
        op_sub = NEML2SolvableBlockOperator(am_sub)

        v = NEML2BlockVector(
            [torch.rand(nblk, sbat, np_), torch.rand(nblk, sbat, ns_)],
            layout,
            [0, 0],
        )

        # Should not raise.
        state = op_diag.pcr_init(op_sub, v)
        self.assertIsNotNone(state)


class TestStructuredSchurPCR(unittest.TestCase):
    """Validate the structure-preserving O(N) Schur-PCR (the default for 2-group
    BLOCK+DENSE cross-block systems) against Dense Thomas and Dense PCR on
    synthetic 2-group BLOCK(p)+DENSE(s) systems with grains (the Taylor
    mix-mode structure: cross-block A via A_ps/A_sp, block-diagonal Jn)."""

    def _build(self, N, np_, ns, nblk, B=1):
        layout = AxisLayout(
            [["x_p"], ["x_s"]],
            [[N], []],
            [[np_], [ns]],
            [AxisLayout.IStructure.BLOCK, AxisLayout.IStructure.DENSE],
        )
        Ip = torch.eye(np_).reshape(1, 1, 1, np_, np_)
        App = torch.rand(nblk, B, N, np_, np_) + 3.0 * Ip
        Ass = torch.rand(nblk, B, ns, ns) + 3.0 * torch.eye(ns)
        Aps = torch.rand(nblk, B, N, np_, ns) * 0.3
        Asp = torch.rand(nblk, B, N, ns, np_) * 0.3
        diag = AssembledMatrix(
            layout,
            layout,
            [
                [Tensor(App, 2, 1), Tensor(Aps, 2, 1)],
                [Tensor(Asp, 2, 1), Tensor(Ass, 2, 0)],
            ],
        )
        Jn_pp = -Ip.expand(nblk, B, N, np_, np_).contiguous()
        sub = AssembledMatrix(
            layout,
            layout,
            [[Tensor(Jn_pp, 2, 1), Tensor()], [Tensor(), Tensor()]],
        )
        v = NEML2BlockVector(
            [torch.rand(nblk, B, N, np_), torch.rand(nblk, B, ns)], layout, [1, 0]
        )
        return layout, diag, sub, v

    def test_structured_pcr_matches_thomas_and_dense_pcr(self):
        """Structured Schur-PCR is now the DEFAULT for 2-group BLOCK+DENSE
        cross-block systems (no env var). Validate it against two independent
        Dense-backend oracles on the flat-materialized system: Dense Thomas and
        Dense PCR (the latter is exactly what the flat-Dense fallback delegates
        to). Power-of-two ``nblk`` keeps each chunk a single PCR window so the
        structured path is used end to end."""
        from pyzag import chunktime
        from pyzag.operators.dense import DenseBlockOperator, DenseBlockVector
        from pyzag.operators.neml2 import _am_to_flat, _av_to_flat

        for nblk in [2, 4, 8, 16]:
            for N in [3, 6, 16]:
                torch.manual_seed(1)
                layout, diag, sub, v = self._build(N, 10, 12, nblk)
                A_flat = _am_to_flat(diag)
                B_flat = _am_to_flat(sub)
                v_flat = _av_to_flat(v.to_av())
                # Dense Thomas oracle on the flat materialized system.
                x_thomas = (
                    chunktime.BidiagonalThomasFactorization(
                        DenseBlockOperator.factored(A_flat.clone()),
                        DenseBlockOperator(B_flat[1:].clone()),
                    )
                    .matvec(DenseBlockVector(v_flat.clone()))
                    .data
                )
                # Dense PCR oracle (equivalent to the flat-Dense fallback path).
                x_dense_pcr = (
                    chunktime.BidiagonalPCRFactorization(
                        DenseBlockOperator.factored(A_flat.clone()),
                        DenseBlockOperator(B_flat[1:].clone()),
                    )
                    .matvec(DenseBlockVector(v_flat.clone()))
                    .data
                )
                # Structured Schur-PCR via the NEML2 operator (default path).
                x_st = chunktime.BidiagonalPCRFactorization(
                    NEML2SolvableBlockOperator(diag),
                    NEML2SolvableBlockOperator(sub)[1:],
                ).matvec(v.clone())

                d_th = (x_thomas - _av_to_flat(x_st.to_av())).abs().max().item()
                d_dp = (x_dense_pcr - _av_to_flat(x_st.to_av())).abs().max().item()
                self.assertLess(
                    d_th, 1e-9, f"structured vs Thomas nblk={nblk} N={N}: {d_th}"
                )
                self.assertLess(
                    d_dp, 1e-9, f"structured vs Dense PCR nblk={nblk} N={N}: {d_dp}"
                )


class TestNonPowerOfTwoPCR(unittest.TestCase):
    """Regression tests for the multi-window PCR path (non-power-of-two chunk
    sizes), where the reduced subdiagonal ``B_red`` is written back into the
    operator and consumed by the next reduction window.

    Both NEML2 PCR finalizers must emit a *correct* ``B_red``:

    - the structured Schur-PCR path (2-group BLOCK+DENSE), and
    - the flat-dense multi-group path (2-group DENSE+DENSE cross-block).

    If a finalizer truncates or zeros ``B_red`` (only valid for a single PCR
    window), the second window reduces against a wrong subdiagonal and the solve
    is silently incorrect. Power-of-two ``nblk`` cannot catch this (single
    window), so these tests use nblk = 3, 5, 6, 7.
    """

    @staticmethod
    def _oracles(A_flat, B_flat, v_flat):
        """Two independent Dense-backend oracles on the flat-materialized system."""
        from pyzag import chunktime
        from pyzag.operators.dense import DenseBlockOperator, DenseBlockVector

        x_thomas = (
            chunktime.BidiagonalThomasFactorization(
                DenseBlockOperator.factored(A_flat.clone()),
                DenseBlockOperator(B_flat[1:].clone()),
            )
            .matvec(DenseBlockVector(v_flat.clone()))
            .data
        )
        x_dense_pcr = (
            chunktime.BidiagonalPCRFactorization(
                DenseBlockOperator.factored(A_flat.clone()),
                DenseBlockOperator(B_flat[1:].clone()),
            )
            .matvec(DenseBlockVector(v_flat.clone()))
            .data
        )
        return x_thomas, x_dense_pcr

    @staticmethod
    def _build_dense_dense(np_, ns, nblk, B=1):
        """2-group DENSE+DENSE system with cross-block A (routes to multi-group
        PCR) and a block-diagonal subdiagonal Jn."""
        layout = AxisLayout(
            [["x_p"], ["x_s"]],
            [[], []],
            [[np_], [ns]],
            [AxisLayout.IStructure.DENSE, AxisLayout.IStructure.DENSE],
        )
        Ip = torch.eye(np_).reshape(1, 1, np_, np_)
        Is = torch.eye(ns).reshape(1, 1, ns, ns)
        App = torch.rand(nblk, B, np_, np_) + 3.0 * Ip
        Ass = torch.rand(nblk, B, ns, ns) + 3.0 * Is
        Aps = torch.rand(nblk, B, np_, ns) * 0.3
        Asp = torch.rand(nblk, B, ns, np_) * 0.3
        diag = AssembledMatrix(
            layout,
            layout,
            [
                [Tensor(App, 2, 0), Tensor(Aps, 2, 0)],
                [Tensor(Asp, 2, 0), Tensor(Ass, 2, 0)],
            ],
        )
        Jn_pp = -Ip.expand(nblk, B, np_, np_).contiguous()
        Jn_ss = -Is.expand(nblk, B, ns, ns).contiguous()
        sub = AssembledMatrix(
            layout,
            layout,
            [[Tensor(Jn_pp, 2, 0), Tensor()], [Tensor(), Tensor(Jn_ss, 2, 0)]],
        )
        v = NEML2BlockVector(
            [torch.rand(nblk, B, np_), torch.rand(nblk, B, ns)], layout, [0, 0]
        )
        return layout, diag, sub, v

    def test_structured_schur_pcr_non_power_of_two(self):
        from pyzag import chunktime
        from pyzag.operators.neml2 import _am_to_flat, _av_to_flat

        builder = TestStructuredSchurPCR()
        for nblk in [3, 5, 6, 7]:
            for N in [3, 6]:
                torch.manual_seed(1)
                _, diag, sub, v = builder._build(N, 10, 12, nblk)
                A_flat = _am_to_flat(diag)
                B_flat = _am_to_flat(sub)
                v_flat = _av_to_flat(v.to_av())
                x_thomas, x_dense_pcr = self._oracles(A_flat, B_flat, v_flat)
                x_st = chunktime.BidiagonalPCRFactorization(
                    NEML2SolvableBlockOperator(diag),
                    NEML2SolvableBlockOperator(sub)[1:],
                ).matvec(v.clone())
                got = _av_to_flat(x_st.to_av())
                d_th = (x_thomas - got).abs().max().item()
                d_dp = (x_dense_pcr - got).abs().max().item()
                self.assertLess(
                    d_th, 1e-9, f"schur vs Thomas nblk={nblk} N={N}: {d_th}"
                )
                self.assertLess(
                    d_dp, 1e-9, f"schur vs Dense PCR nblk={nblk} N={N}: {d_dp}"
                )

    def test_multigroup_pcr_non_power_of_two(self):
        from pyzag import chunktime
        from pyzag.operators.neml2 import _am_to_flat, _av_to_flat

        for nblk in [3, 5, 6, 7]:
            torch.manual_seed(2)
            _, diag, sub, v = self._build_dense_dense(10, 12, nblk)
            A_flat = _am_to_flat(diag)
            B_flat = _am_to_flat(sub)
            v_flat = _av_to_flat(v.to_av())
            x_thomas, x_dense_pcr = self._oracles(A_flat, B_flat, v_flat)
            x_mg = chunktime.BidiagonalPCRFactorization(
                NEML2SolvableBlockOperator(diag),
                NEML2SolvableBlockOperator(sub)[1:],
            ).matvec(v.clone())
            got = _av_to_flat(x_mg.to_av())
            d_th = (x_thomas - got).abs().max().item()
            d_dp = (x_dense_pcr - got).abs().max().item()
            self.assertLess(d_th, 1e-9, f"multigroup vs Thomas nblk={nblk}: {d_th}")
            self.assertLess(d_dp, 1e-9, f"multigroup vs Dense PCR nblk={nblk}: {d_dp}")


if __name__ == "__main__":
    unittest.main()
