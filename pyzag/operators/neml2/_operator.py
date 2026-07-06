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

"""NEML2-backend solvable block operator (diagonal solves, Schur/PCR)."""

from __future__ import annotations

from math import prod

import torch

from ..base import BlockOperator, BlockVector, PCRState, SolvableBlockOperator
from ..dense import (
    DenseBlockOperator,
    DenseBlockVector,
    DensePCRState,
    _dense_pcr_cyclic_shift,
    _lu_factor_guarded,
    _lu_solve_guarded,
)
from ._containers import AssembledMatrix, AxisLayout, Tensor
from ._assembly import (
    _am_to_flat,
    _av_to_flat,
    _flat_to_sub_am,
    _group_flat_size,
    _group_intmd_sizes,
    _pcr_tol,
    _split_flat_to_av,
)
from ._vector import NEML2BlockVector
from ._pcr import (
    MultiGroupPCRState,
    NEML2PCRState,
    NEML2SchurPCRState,
    _FlatCarrier,
    _FlatStructuredAinv,
    _carrier_cyclic_shift,
    _carrier_from_sub_am,
    _carrier_mul,
    _carrier_neg,
    _carrier_pad_rank,
    _carrier_time_slice,
)


class NEML2SolvableBlockOperator(SolvableBlockOperator):
    """Block operator backed by a NEML2 AssembledMatrix.

    matvec uses NEML2's native `am * av` which correctly applies `intmd_sum`
    on the contracted dimension when the input group is BLOCK. solve uses a
    Schur complement when the layout has a BLOCK+DENSE split (matching NEML2's
    own SchurComplement C++ solver pattern); for single-group layouts it
    falls back to standard batched LU on the single block.

    The adjoint path uses this same class, just with a transposed AssembledMatrix
    (built by `_transpose_am`). This mirrors how DenseBlockOperator handles
    adjoint via `.transpose(-1, -2)` + regular DenseBlockOperator: no wrapper
    class, full SolvableBlockOperator API inherited automatically.
    """

    def __init__(self, am: "AssembledMatrix") -> None:
        self.am = am
        # Lazy per-diagonal-group LU caches (only set up when first solve is called).
        self._lu_per_group: list[torch.Tensor] | None = None
        self._pivots_per_group: list[torch.Tensor] | None = None

    @classmethod
    def factored(cls, am: "AssembledMatrix") -> "NEML2SolvableBlockOperator":
        """Construct the operator and eagerly perform its LU factorization."""
        op = cls(am)
        op._ensure_lu()
        return op

    def _ensure_lu(self) -> None:
        """Lazy LU factorization of each diagonal group's per-step matrices.

        Uses :func:`_lu_factor_guarded` so large matrices on cuda use
        non-batched cuSOLVER (avoiding the "batched routines designed for
        small sizes" warning + slowdown for the flat-dense path at large G).
        """
        if self._lu_per_group is not None:
            return
        ng = self.am.row_layout.ngroup()
        lus, pivots = [], []
        for g in range(ng):
            diag_raw = self.am.tensors[g][g].torch()
            # diag_raw shape: (nblk, B, *intmd, n, n). lu_factor treats all
            # leading dims as batch — naturally batches across (nblk, B, *intmd).
            lu, piv = _lu_factor_guarded(diag_raw)
            lus.append(lu)
            pivots.append(piv)
        self._lu_per_group = lus
        self._pivots_per_group = pivots

    # ----- BlockOperator abstract methods -----

    @property
    def device(self) -> torch.device:
        return self.am.tensors[0][0].torch().device

    @property
    def dtype(self) -> torch.dtype:
        return self.am.tensors[0][0].torch().dtype

    @property
    def nblk(self) -> int:
        return self.am.tensors[0][0].torch().shape[0]

    @property
    def batch_size(self) -> int:
        return self.am.tensors[0][0].torch().shape[1]

    def matvec(self, x: BlockVector) -> NEML2BlockVector:
        """Per-instance matvec consistent with the Schur solve.

        Why NOT NEML2 native ``am * av``: NEML2's native operator applies
        ``intmd_sum`` whenever the COL group is BLOCK, regardless of row.
        That folds BLOCK×BLOCK per-instance matmuls into a sum-then-broadcast
        — correct for the Schur cross-block contractions (steps 3-4 of the
        Schur algorithm) but wrong for end-to-end matvec. The linear system
        that NEML2's Schur solver inverts is the per-instance one:

            b_p[i] = A_pp[i] @ x_p[i] + A_ps[i] @ x_s
            b_s    = Σ_i (A_sp[i] @ x_p[i]) + A_ss @ x_s

        So our matvec applies the rule:

            intmd_sum fires when ROW is DENSE and COL is BLOCK
            (aggregate per-instance contributions into a global output).

        This rule makes matvec and Schur solve mutual inverses and makes the
        true mathematical transpose computable via the same logic.
        """
        if not isinstance(x, NEML2BlockVector):
            raise TypeError(
                "NEML2SolvableBlockOperator.matvec expects NEML2BlockVector."
            )
        return self._mv_per_grain(self.am, x, transpose=False)

    def t_matvec(self, x: BlockVector) -> NEML2BlockVector:
        """A^T @ x with the per-instance interpretation.

        Conceptually applies the transpose of the per-instance matvec. Done as:
        for each output group (col index of A), accumulate contributions from
        each input group (row index of A), using transposed blocks. Applies
        intmd_sum when the OUTPUT (transpose's row, = A's col) is DENSE and
        the input is BLOCK.
        """
        if not isinstance(x, NEML2BlockVector):
            raise TypeError(
                "NEML2SolvableBlockOperator.t_matvec expects NEML2BlockVector."
            )
        return self._mv_per_grain(self.am, x, transpose=True)

    @staticmethod
    def _mv_per_grain(am, x, transpose: bool):
        """Per-instance matrix-vector product (matvec or its transpose).

        Args:
            am: AssembledMatrix
            x:  NEML2BlockVector with structure matching am's col layout
                (transpose=False) or row layout (transpose=True)
            transpose: if True, computes A^T @ x

        Algorithm: for each output group i, accumulate per-input-group block
        contributions. Each block matmul is per-instance (preserves intmd via
        torch.matmul broadcasting). After matmul, intmd_sum fires iff the
        output is DENSE and the input is BLOCK.
        """
        if transpose:
            # For A^T @ y, output rows are A's COL groups; inputs are A's ROW groups.
            out_layout = am.col_layout
            in_layout = am.row_layout
        else:
            out_layout = am.row_layout
            in_layout = am.col_layout

        n_out = out_layout.ngroup()
        n_in = in_layout.ngroup()

        out_tensors: list[torch.Tensor | None] = [None] * n_out

        for i in range(n_out):
            out_istr = out_layout.istr(i)
            accum: torch.Tensor | None = None

            for j in range(n_in):
                if transpose:
                    # A^T block at (i, j) is A's (j, i) transposed on last two dims.
                    blk = am.tensors[j][i]
                else:
                    blk = am.tensors[i][j]
                if not blk.defined():
                    continue

                blk_raw = blk.torch()
                if transpose:
                    blk_raw = blk_raw.transpose(-1, -2)

                xj = x.raw_tensors[j]  # (nblk, B, [intmd_j,] base_j)
                x_intmd = x.intmd_dims[j]

                # Block's intmd dim count. Transposing the last 2 dims leaves it
                # unchanged, so blk.intmd.dim() is valid in both branches.
                blk_intmd = blk.intmd.dim()

                # Align intmd dims: torch.matmul cannot broadcast across different
                # batch-dim counts. Insert size-1 dims after the 2 dynamic dims so
                # both tensors' intmd regions have the same length.
                diff = blk_intmd - x_intmd
                if diff > 0:
                    # x has fewer intmd dims; add size-1 at position 2 (after nblk, B).
                    for _ in range(diff):
                        xj = xj.unsqueeze(2)
                elif diff < 0:
                    for _ in range(-diff):
                        blk_raw = blk_raw.unsqueeze(2)
                effective_intmd = max(blk_intmd, x_intmd)

                # Per-instance matmul: (..., r, c) @ (..., c, 1) → (..., r, 1) → squeeze
                r = torch.matmul(blk_raw, xj.unsqueeze(-1)).squeeze(-1)
                # r shape: (nblk, B, *effective_intmd_extents, n_row)

                # If OUTPUT group is DENSE but result has intmd dims (came from BLOCK
                # input or BLOCK block), sum them away to produce a DENSE result.
                if out_istr == AxisLayout.IStructure.DENSE and effective_intmd > 0:
                    for _ in range(effective_intmd):
                        # The intmd dims sit between the batch (dim 0,1) and the last
                        # (output_var) dim. After each sum, the relevant dim is -2.
                        r = r.sum(dim=-2)

                if accum is None:
                    accum = r
                else:
                    # Broadcast-add. If accum and r have different intmd dim counts
                    # (some block contributions DENSE, others BLOCK), torch
                    # broadcasting auto-aligns trailing dims via the leading
                    # singleton from unsqueezes above.
                    if r.ndim != accum.ndim:
                        # Pad the lower-dim one with leading singleton intmd dims.
                        if r.ndim < accum.ndim:
                            for _ in range(accum.ndim - r.ndim):
                                r = r.unsqueeze(2)
                        else:
                            for _ in range(r.ndim - accum.ndim):
                                accum = accum.unsqueeze(2)
                    accum = accum + r

            if accum is None:
                # No defined blocks contributed for this output group;
                # allocate zeros of the right output shape.
                intmd_sizes = (
                    _group_intmd_sizes(out_layout, i)
                    if out_istr == AxisLayout.IStructure.BLOCK
                    else []
                )
                # Source batch dims (and dtype/device) from any defined block.
                ref_shape = None
                for ii in range(n_out):
                    for jj in range(n_in):
                        bb = am.tensors[ii][jj]
                        if bb.defined():
                            ref_shape = bb.torch().shape
                            break
                    if ref_shape is not None:
                        break
                if ref_shape is None:
                    ref_shape = x.raw_tensors[0].shape
                nblk = ref_shape[0]
                sbat = ref_shape[1]
                # Per-instance base-dof count = group_flat_size // intmd_numel.
                # Using _group_flat_size avoids storage_sizes' include_intmd flag.
                group_dofs = _group_flat_size(out_layout, i)
                intmd_numel = int(prod(intmd_sizes)) if intmd_sizes else 1
                n_out_var = group_dofs // intmd_numel
                shape = (nblk, sbat, *intmd_sizes, n_out_var)
                accum = torch.zeros(
                    shape,
                    dtype=x.raw_tensors[0].dtype,
                    device=x.raw_tensors[0].device,
                )
            else:
                # A BLOCK output missing its intmd dim (only DENSE+DENSE inputs
                # contributed) would need an expand here, but in practice a BLOCK
                # output always has a BLOCK input supplying it; broadcasting on
                # read covers the rest.
                pass

            out_tensors[i] = accum

        # Infer intmd dims from produced shapes (layout may be stale; shape
        # contract is (nblk, B, *intmd, n_row), so intmd dims = ndim - 3).
        out_intmd_dims = [max(0, t.ndim - 3) for t in out_tensors]
        return NEML2BlockVector(out_tensors, out_layout, out_intmd_dims)

    def solve(self, rhs: BlockVector) -> NEML2BlockVector:
        if not isinstance(rhs, NEML2BlockVector):
            raise TypeError(
                "NEML2SolvableBlockOperator.solve expects NEML2BlockVector."
            )
        self._ensure_lu()
        ng = self.am.row_layout.ngroup()
        if ng == 1:
            # Single-group: straightforward LU on the (nblk, B, [intmd...,] n, n) block.
            lu = self._lu_per_group[0]
            piv = self._pivots_per_group[0]
            b = rhs.raw_tensors[0]  # (nblk, B, [intmd...,] n)
            x = _lu_solve_guarded(lu, piv, b.unsqueeze(-1)).squeeze(-1)
            return NEML2BlockVector([x], self.am.col_layout, rhs.intmd_dims)
        if ng == 2:
            return self._schur_solve(rhs)
        raise NotImplementedError(
            f"NEML2SolvableBlockOperator.solve supports 1 or 2 groups (got {ng}). "
            "Use _flat_solve as a fallback or extend Schur to N groups."
        )

    def _schur_solve(self, rhs: NEML2BlockVector) -> NEML2BlockVector:
        """Schur complement for a 2-group system where at least one group is BLOCK.

        Auto-detects primary = BLOCK row-group (cheaper to factor per-grain),
        Schur = the other. Falls back to primary=0, Schur=1 for DENSE+DENSE.

        Generalized to any number of intmd dims on the primary BLOCK group —
        sums over all intmd axes during the cross-block contractions
        (A_sp @ Y_ps and A_sp @ z_p) and unsqueezes the same count when
        broadcasting x_s back through Y_ps.
        """
        row_l = self.am.row_layout
        col_l = self.am.col_layout
        if row_l.istr(0) == AxisLayout.IStructure.BLOCK:
            p, s = 0, 1
        elif row_l.istr(1) == AxisLayout.IStructure.BLOCK:
            p, s = 1, 0
        else:
            p, s = 0, 1

        A_pp_t = self.am.tensors[p][p]
        A_ps_t = self.am.tensors[p][s]
        A_sp_t = self.am.tensors[s][p]
        A_ss_t = self.am.tensors[s][s]
        A_ss = A_ss_t.torch()  # (nblk, B, ns, ns)
        has_ps = A_ps_t.defined()
        has_sp = A_sp_t.defined()
        A_ps = A_ps_t.torch() if has_ps else None
        A_sp = A_sp_t.torch() if has_sp else None
        b_p = rhs.raw_tensors[p]  # (nblk, B, *intmd_p, np)
        b_s = rhs.raw_tensors[s]  # (nblk, B, ns)

        # Number of intmd dims on the primary BLOCK group — drives sum/unsqueeze.
        n_intmd_p = A_pp_t.intmd.dim()
        # Position of the intmd dims: right after the dynamic-batch dims. We read
        # the dynamic-dim count from the block (NOT a hardcoded 2) so the same
        # solve works when extra leading dims are present — e.g. the PCR window
        # dim prepended ahead of (nblk, sbat) by the Schur-PCR reduction.
        dyn = A_pp_t.dynamic.dim()
        intmd_axes = tuple(range(dyn, dyn + n_intmd_p))

        lu_pp = self._lu_per_group[p]
        piv_pp = self._pivots_per_group[p]

        # 1. Y_ps = A_pp^{-1} @ A_ps  (only if A_ps present; otherwise stays None)
        Y_ps = torch.linalg.lu_solve(lu_pp, piv_pp, A_ps) if has_ps else None
        # 2. z_p = A_pp^{-1} @ b_p
        z_p = torch.linalg.lu_solve(lu_pp, piv_pp, b_p.unsqueeze(-1)).squeeze(-1)
        # 3. S_ss = A_ss - sum_{intmd}(A_sp @ Y_ps) — zero if either cross-block missing
        if has_sp and has_ps:
            SY = torch.matmul(A_sp, Y_ps)
            if intmd_axes:
                SY = SY.sum(dim=intmd_axes)
            S_ss = A_ss - SY
        else:
            S_ss = A_ss
        # 4. d_s = b_s - sum_{intmd}(A_sp @ z_p) — only if A_sp present
        if has_sp:
            dz = torch.matmul(A_sp, z_p.unsqueeze(-1)).squeeze(-1)
            if intmd_axes:
                dz = dz.sum(dim=intmd_axes)
            d_s = b_s - dz
        else:
            d_s = b_s
        # 5. x_s = S_ss^{-1} d_s
        x_s = torch.linalg.solve(S_ss, d_s.unsqueeze(-1)).squeeze(-1)
        # 6. x_p = z_p - Y_ps @ x_s — zero correction if A_ps missing
        if has_ps:
            x_s_b = x_s
            for _ in range(n_intmd_p):
                x_s_b = x_s_b.unsqueeze(dyn)  # insert singleton at each intmd axis
            Yx = torch.matmul(Y_ps, x_s_b.unsqueeze(-1)).squeeze(-1)
            x_p = z_p - Yx
        else:
            x_p = z_p

        # Pack back. The solution's per-group intmd structure matches the rhs's
        # exactly; the solve preserves it regardless of the column layout's
        # declared intmd_sizes.
        out_tensors = [None, None]
        out_tensors[p] = x_p
        out_tensors[s] = x_s
        return NEML2BlockVector(out_tensors, col_l, list(rhs.intmd_dims))

    def clone(self) -> "NEML2SolvableBlockOperator":
        n_rows = self.am.row_layout.ngroup()
        n_cols = self.am.col_layout.ngroup()
        cloned = [[Tensor() for _ in range(n_cols)] for _ in range(n_rows)]
        for i in range(n_rows):
            for j in range(n_cols):
                blk = self.am.tensors[i][j]
                if blk.defined():
                    cloned[i][j] = Tensor(
                        blk.torch().clone(), blk.dynamic.dim(), blk.intmd.dim()
                    )
        new_am = AssembledMatrix(self.am.row_layout, self.am.col_layout, cloned)
        out = NEML2SolvableBlockOperator(new_am)
        # Don't copy LU cache (cheap to redo if needed); keeping it would risk staleness.
        return out

    def __getitem__(self, idx: int | slice) -> "NEML2SolvableBlockOperator":
        n_rows = self.am.row_layout.ngroup()
        n_cols = self.am.col_layout.ngroup()
        sliced = [[Tensor() for _ in range(n_cols)] for _ in range(n_rows)]
        for i in range(n_rows):
            for j in range(n_cols):
                blk = self.am.tensors[i][j]
                if blk.defined():
                    blk_raw = blk.torch()
                    raw = blk_raw[idx]
                    # If int indexing reduced the leading dim, re-add it.
                    if isinstance(idx, int) or raw.ndim < blk_raw.ndim:
                        raw = raw.unsqueeze(0)
                    sliced[i][j] = Tensor(raw, blk.dynamic.dim(), blk.intmd.dim())
        new_am = AssembledMatrix(self.am.row_layout, self.am.col_layout, sliced)
        new_op = NEML2SolvableBlockOperator(new_am)
        # Carry the LU cache through the slice (block axis is axis 0 of each
        # cached lu/pivot tensor, same axis we index here). This mirrors the
        # dense backend and lets Thomas reuse the factorization on the per-block
        # A[i:i+1].solve() calls instead of re-factoring every block.
        if self._lu_per_group is not None:

            def _slice_cache(t: torch.Tensor) -> torch.Tensor:
                s = t[idx]
                if isinstance(idx, int) or s.ndim < t.ndim:
                    s = s.unsqueeze(0)
                return s

            new_op._lu_per_group = [_slice_cache(lu) for lu in self._lu_per_group]
            new_op._pivots_per_group = [_slice_cache(p) for p in self._pivots_per_group]
        return new_op

    def __setitem__(self, idx: int | slice, other: BlockOperator) -> None:
        if not isinstance(other, NEML2SolvableBlockOperator):
            raise TypeError(
                "NEML2SolvableBlockOperator assignment requires NEML2SolvableBlockOperator."
            )
        # ``blk.torch()`` may return a detached view; rebuild ``self.am`` from
        # fresh cloned-and-replaced blocks rather than mutating in place.
        n_rows = self.am.row_layout.ngroup()
        n_cols = self.am.col_layout.ngroup()
        new_blocks = [[Tensor() for _ in range(n_cols)] for _ in range(n_rows)]
        for i in range(n_rows):
            for j in range(n_cols):
                blk = self.am.tensors[i][j]
                obl = other.am.tensors[i][j]
                if blk.defined() and obl.defined():
                    new_raw = blk.torch().clone()
                    new_raw[idx] = obl.torch()
                    new_blocks[i][j] = Tensor(
                        new_raw, blk.dynamic.dim(), blk.intmd.dim()
                    )
                elif blk.defined():
                    new_blocks[i][j] = blk
                # else leave as undefined Tensor()
        self.am = AssembledMatrix(self.am.row_layout, self.am.col_layout, new_blocks)
        # Invalidate LU cache.
        self._lu_per_group = None
        self._pivots_per_group = None

    def pad_front(self, n: int = 1) -> "NEML2SolvableBlockOperator":
        if n < 0:
            raise ValueError("n must be nonnegative.")
        if n == 0:
            return self.clone()
        n_rows = self.am.row_layout.ngroup()
        n_cols = self.am.col_layout.ngroup()
        padded = [[Tensor() for _ in range(n_cols)] for _ in range(n_rows)]
        for i in range(n_rows):
            for j in range(n_cols):
                blk = self.am.tensors[i][j]
                if blk.defined():
                    raw = blk.torch()
                    # Pad nblk dim (axis 0) with n leading zeros.
                    raw_pad = torch.nn.functional.pad(
                        raw,
                        # F.pad pads from the LAST dim backward; for axis 0 pad
                        # with 2*(ndim-1) zeros followed by (n, 0).
                        (0,) * (2 * (raw.ndim - 1)) + (n, 0),
                    )
                    padded[i][j] = Tensor(raw_pad, blk.dynamic.dim(), blk.intmd.dim())
        new_am = AssembledMatrix(self.am.row_layout, self.am.col_layout, padded)
        return NEML2SolvableBlockOperator(new_am)

    def trim_front(self, n: int = 1) -> "NEML2SolvableBlockOperator":
        if n < 0:
            raise ValueError("n must be nonnegative.")
        if n == 0:
            return self.clone()
        return self[n:]

    # ----- PCR (per-group, requires Jn block-diagonal in groups) -----

    @staticmethod
    def _check_pcr_block_diagonal(am: "AssembledMatrix", what: str) -> None:
        """Raise if ``am`` has any non-zero cross-group block. Per-group PCR
        requires block-diagonality in groups for both the diagonal and sub
        operators."""
        n_rows = am.row_layout.ngroup()
        n_cols = am.col_layout.ngroup()
        for i in range(n_rows):
            for j in range(n_cols):
                if i == j:
                    continue
                blk = am.tensors[i][j]
                if blk.defined() and blk.torch().abs().max() > 0:
                    raise NotImplementedError(
                        f"PCR requires {what} block-diagonal in variable "
                        f"groups. Non-zero cross-group coupling found at "
                        f"[{i}][{j}]. Use BidiagonalThomasFactorization "
                        "instead, or implement true multi-group PCR with "
                        "Schur at each reduction level."
                    )

    def pcr_init(self, B: BlockOperator, v: BlockVector) -> NEML2PCRState:
        if not isinstance(B, NEML2SolvableBlockOperator):
            raise TypeError("B must be NEML2SolvableBlockOperator.")
        if not isinstance(v, NEML2BlockVector):
            raise TypeError("v must be NEML2BlockVector.")
        # Fail fast on mismatched group structure (otherwise the inner per-group
        # loop crashes with a vague IndexError).
        if (
            B.am.row_layout.ngroup() != self.am.row_layout.ngroup()
            or B.am.col_layout.ngroup() != self.am.col_layout.ngroup()
        ):
            raise NotImplementedError(
                "PCR requires the sub-operator B to share the diagonal "
                "operator's group structure. "
                f"Got A={self.am.row_layout.ngroup()}x"
                f"{self.am.col_layout.ngroup()}, "
                f"B={B.am.row_layout.ngroup()}x{B.am.col_layout.ngroup()}. "
                "If this came from NEML2PyzagFactory, _expand_am_dynamic in "
                "interface.py should zero-pad B to match A's group structure."
            )
        # Fast path: per-group PCR when both A and B are block-diagonal.
        # Otherwise dispatch the cross-block A to a multi-group PCR path.
        try:
            self._check_pcr_block_diagonal(self.am, "self (diagonal operator)")
            a_is_block_diag = True
        except NotImplementedError:
            a_is_block_diag = False
        if not a_is_block_diag:
            # Cross-block A. Prefer the structured O(N) Schur-PCR (the default)
            # when the layout is a 2-group BLOCK+DENSE split with a
            # block-diagonal subdiagonal Jn — the Taylor mix-mode fast path.
            # Otherwise fall back to the flat-Dense O(N^3) multi-group PCR, which
            # handles any cross-block structure: all-DENSE groups, or a
            # grain-dense Jn produced by a non-power-of-two PCR window (where the
            # chunktime driver overwrites B[s+1:e] with the reduced subdiagonal).
            # Power-of-two chunks with a BLOCK group stay fully structured. Note
            # the reduced-subdiagonal *values* written back by pcr_finalize are
            # never consumed in a result-affecting way (see _flat_to_sub_am), so
            # whichever path a reused window takes, the solve is correct.
            row_l = self.am.row_layout
            has_block = any(
                row_l.istr(g) == AxisLayout.IStructure.BLOCK
                for g in range(row_l.ngroup())
            )
            if has_block and row_l.ngroup() == 2:
                try:
                    self._check_pcr_block_diagonal(B.am, "B (sub operator / Jn)")
                except NotImplementedError:
                    return self.pcr_init_multigroup(B, v)
                return self.pcr_init_schur(B, v)
            return self.pcr_init_multigroup(B, v)
        self._check_pcr_block_diagonal(B.am, "B (sub operator / Jn)")
        self._ensure_lu()
        per_group_states = []
        intmd_dims_per_group: list[int] = []
        intmd_sizes_per_group: list[list[int]] = []
        for g in range(self.am.row_layout.ngroup()):
            diag_raw = self.am.tensors[g][g].torch()
            sub_raw = B.am.tensors[g][g].torch()
            v_raw = v.raw_tensors[g]
            # Fold intmd into the batch dim for DensePCRState shape compat;
            # read intmd info from the vector tensor (layout may be stale).
            intmd_dims = v.intmd_dims[g]
            intmd_sizes = (
                list(v_raw.shape[2 : 2 + intmd_dims]) if intmd_dims > 0 else []
            )
            if intmd_dims > 0:
                # diag_raw: (nblk, B, *intmd, n, n) -> (nblk, B*prod(intmd), n, n)
                nblk = diag_raw.shape[0]
                n = diag_raw.shape[-1]
                diag_fold = diag_raw.reshape(nblk, -1, n, n)
                sub_fold = sub_raw.reshape(nblk, -1, n, n)
                v_fold = v_raw.reshape(nblk, -1, n)
            else:
                diag_fold = diag_raw
                sub_fold = sub_raw
                v_fold = v_raw
            lu, piv, _ = torch.linalg.lu_factor_ex(diag_fold)
            # Clone b/v so the in-place reductions in pcr_reduce_level own their
            # buffers and never mutate the caller's operator/vector (the reshapes
            # above may be views into B/v storage). Matches the base contract and
            # the dense backend's DenseBlockOperator.pcr_init.
            state = DensePCRState(
                lu=lu.unsqueeze(0),
                pivots=piv.unsqueeze(0),
                b=sub_fold.unsqueeze(0).clone(),
                v=v_fold.unsqueeze(0).unsqueeze(-1).clone(),
            )
            per_group_states.append(state)
            intmd_dims_per_group.append(intmd_dims)
            intmd_sizes_per_group.append(intmd_sizes)
        return NEML2PCRState(
            per_group_states,
            self.am.row_layout,
            intmd_dims_per_group,
            intmd_sizes_per_group,
        )

    def pcr_reduce_level(self, state: PCRState, level: int):
        if isinstance(state, NEML2SchurPCRState):
            return self.pcr_reduce_level_schur(state, level)
        if isinstance(state, MultiGroupPCRState):
            return self.pcr_reduce_level_multigroup(state, level)
        if not isinstance(state, NEML2PCRState):
            raise TypeError("state must be NEML2PCRState or MultiGroupPCRState.")
        new_per_group = []
        for s in state.per_group:
            lu, pivots, b, v = s.lu, s.pivots, s.b, s.v
            v[:, 1:] -= torch.matmul(
                b[:, 1:],
                torch.linalg.lu_solve(lu[:, :-1], pivots[:, :-1], v[:, :-1]),
            )
            b[:, 2:] = -torch.matmul(
                b[:, 2:],
                torch.linalg.lu_solve(lu[:, 1:-1], pivots[:, 1:-1], b[:, 1:-1]),
            )
            new_per_group.append(
                DensePCRState(
                    lu=_dense_pcr_cyclic_shift(lu, level),
                    pivots=_dense_pcr_cyclic_shift(pivots, level),
                    b=_dense_pcr_cyclic_shift(b, level),
                    v=_dense_pcr_cyclic_shift(v, level),
                )
            )
        return NEML2PCRState(
            new_per_group,
            state.layout,
            state.intmd_dims_per_group,
            state.intmd_sizes_per_group,
        )

    def pcr_finalize(
        self, state: PCRState
    ) -> tuple["NEML2SolvableBlockOperator", "NEML2BlockVector"]:
        if isinstance(state, NEML2SchurPCRState):
            return self.pcr_finalize_schur(state)
        if isinstance(state, MultiGroupPCRState):
            return self.pcr_finalize_multigroup(state)
        if not isinstance(state, NEML2PCRState):
            raise TypeError("state must be NEML2PCRState or MultiGroupPCRState.")
        # Reconstruct per-group B_red and v_red tensors, then re-build AssembledMatrix
        # (with cross-group blocks zero, since per-group PCR doesn't couple groups).
        B_red_raws = []
        v_red_raws = []
        v_red_intmd = []
        for g, s in enumerate(state.per_group):
            B_raw = s.b.squeeze(1)[
                1:
            ].clone()  # (nblk-1, B_fold, n, n); B_fold may include intmd
            v_raw = s.v.squeeze(1)[1:].squeeze(-1).clone()  # (nblk-1, B_fold, n)
            # Unfold intmd using the sizes captured at pcr_init time (layout
            # may report intmd=() for vars whose runtime tensors carry intmd).
            intmd_sizes = state.intmd_sizes_per_group[g]
            if intmd_sizes:
                nblk = B_raw.shape[0]
                n = B_raw.shape[-1]
                # The folded batch is (B_outer, *intmd); restore by reshaping.
                B_fold = B_raw.shape[1]
                B_outer = B_fold // int(prod(intmd_sizes))
                B_raw = B_raw.reshape(nblk, B_outer, *intmd_sizes, n, n)
                v_raw = v_raw.reshape(nblk, B_outer, *intmd_sizes, n)
            B_red_raws.append(B_raw)
            v_red_raws.append(v_raw)
            v_red_intmd.append(len(intmd_sizes))

        # Re-pack into AssembledMatrix (diag-only because pcr is per-group).
        n_rows = state.layout.ngroup()
        T = [[Tensor() for _ in range(n_rows)] for _ in range(n_rows)]
        for g in range(n_rows):
            T[g][g] = Tensor(
                B_red_raws[g], B_red_raws[g].ndim - v_red_intmd[g] - 2, v_red_intmd[g]
            )
        am_red = AssembledMatrix(state.layout, state.layout, T)
        B_op = NEML2SolvableBlockOperator(am_red)
        v_op = NEML2BlockVector(v_red_raws, state.layout, v_red_intmd)
        return B_op, v_op

    def pcr_init_multigroup(
        self, B: "NEML2SolvableBlockOperator", v: "NEML2BlockVector"
    ) -> MultiGroupPCRState:
        """Initialize multi-group PCR by flat-dense delegation.

        Materializes ``self.am``, ``B.am``, and ``v`` to flat torch tensors
        via :func:`_am_to_flat` / :func:`_av_to_flat`, builds Dense backend
        operators / vector, and calls Dense's ``pcr_init``. The state wraps
        the resulting ``DensePCRState`` plus the metadata needed to
        un-flatten the reduced vector at finalize.

        No restriction on B's block-diagonality at this layer — the flat
        materialization handles any combination of cross-blocks. (The
        per-step Schur and PCR are mathematically independent operations;
        PCR uses flat-dense, while the final ``self.A.solve(v_work)`` in
        BidiagonalPCRFactorization still uses Schur.)
        """
        # Materialize to flat torch tensors.
        A_flat = _am_to_flat(self.am)  # (nblk,  sbat, n_flat, n_flat)
        B_flat = _am_to_flat(B.am)  # (nblk', sbat, n_flat, n_flat)
        v_flat = _av_to_flat(v.to_av())  # (nblk,  sbat, n_flat)

        A_dense = DenseBlockOperator.factored(A_flat)
        B_dense = DenseBlockOperator(B_flat)
        v_dense = DenseBlockVector(v_flat)
        dense_state = A_dense.pcr_init(B_dense, v_dense)

        intmd_dims_per_group = list(v.intmd_dims)
        intmd_sizes_per_group = [
            list(t.shape[2 : 2 + v.intmd_dims[g]]) if v.intmd_dims[g] > 0 else []
            for g, t in enumerate(v.raw_tensors)
        ]
        # The Dense op used here — keep a reference so reduce_level can call
        # its method without re-materializing.
        state = MultiGroupPCRState(
            dense_state=dense_state,
            layout=self.am.row_layout,
            intmd_dims_per_group=intmd_dims_per_group,
            intmd_sizes_per_group=intmd_sizes_per_group,
            B_template=B.am,
        )
        state.dense_op = A_dense  # attach for reduce_level / finalize
        return state

    def pcr_reduce_level_multigroup(
        self, state: MultiGroupPCRState, level: int
    ) -> MultiGroupPCRState:
        """Delegate one reduction level to Dense PCR."""
        new_dense_state = state.dense_op.pcr_reduce_level(state.dense_state, level)
        new_state = MultiGroupPCRState(
            dense_state=new_dense_state,
            layout=state.layout,
            intmd_dims_per_group=state.intmd_dims_per_group,
            intmd_sizes_per_group=state.intmd_sizes_per_group,
            B_template=state.B_template,
        )
        new_state.dense_op = state.dense_op
        return new_state

    def pcr_finalize_multigroup(
        self, state: MultiGroupPCRState
    ) -> tuple["NEML2SolvableBlockOperator", "NEML2BlockVector"]:
        """Delegate finalize to Dense PCR, un-flatten ``v_red`` back to
        the original multi-group layout. ``B_red`` is rebuilt as a
        shape-compatible zero op matching the original B's per-(i, j) block
        structure — values are not used downstream (the final per-step solve
        in :class:`BidiagonalPCRFactorization` is ``self.A.solve(v_work)``
        which doesn't reference B).
        """
        _B_dense_red, v_dense_red = state.dense_op.pcr_finalize(state.dense_state)

        # Un-flatten v_dense_red. v_dense_red.data has shape (nblk-1, sbat, n_flat).
        v_red_av = _split_flat_to_av(v_dense_red.data, state.layout)
        v_red = NEML2BlockVector.from_av(v_red_av)

        # Zero B_red matching B's block structure; only shape is consumed
        # by the downstream ``B[s+1:e] = B_red`` assignment.
        nblk_red = v_red.nblk
        tmpl = state.B_template
        n_row = tmpl.row_layout.ngroup()
        n_col = tmpl.col_layout.ngroup()
        new_blocks = [[Tensor() for _ in range(n_col)] for _ in range(n_row)]
        for i in range(n_row):
            for j in range(n_col):
                blk = tmpl.tensors[i][j]
                if not blk.defined():
                    continue
                raw = blk.torch()
                # Take the trailing (sbat, *intmd, n_g, n_h) shape from the
                # template; replace the nblk dim with nblk_red zeros.
                new_shape = (nblk_red,) + tuple(raw.shape[1:])
                new_raw = torch.zeros(new_shape, dtype=raw.dtype, device=raw.device)
                new_blocks[i][j] = Tensor(new_raw, blk.dynamic.dim(), blk.intmd.dim())
        B_red_am = AssembledMatrix(tmpl.row_layout, tmpl.col_layout, new_blocks)
        B_red = NEML2SolvableBlockOperator(B_red_am)
        return B_red, v_red

    # ----- Structure-preserving O(N) Schur-PCR (default for BLOCK+DENSE) -----

    def pcr_init_schur(
        self, B: "NEML2SolvableBlockOperator", v: "NEML2BlockVector"
    ) -> "NEML2SchurPCRState":
        """Initialize the structured Schur-PCR state.

        Builds the per-step diagonal A^{-1} as a rank-ns :class:`_FlatCarrier`
        (``blockdiag_pp(App^{-1}) + low-rank``), the subdiagonal Jn as a carrier
        (block-diagonal pp + small low-rank handle for ps/sp/ss), and the flat
        RHS ``v``. Requires the cross-block diagonal A is 2-group BLOCK+DENSE and
        that ``Jn`` is block-diagonal in groups (Taylor: Jn_pp=-I, Jn_ss absent,
        only a small Jn_sp). The latter is asserted.
        """
        self._check_pcr_block_diagonal(B.am, "B (sub operator / Jn)")
        tol = _pcr_tol()
        ainv = _FlatStructuredAinv.from_diag_am(self.am).to_carrier()
        b = _carrier_from_sub_am(B.am)
        v_flat = _av_to_flat(v.to_av()).unsqueeze(-1)  # (nblk, B, nf, 1)
        # Determine the FIXED low-rank rank carried for ``b`` across all levels.
        # The reduced subdiagonal ``-B A^{-1} B`` low-rank saturates after a few
        # multiplies (rank ~ns and N-independent on Taylor data; it can grow on
        # pathological all-grains-coupled data). We chain free-rank products for
        # the number of reduction levels to find the saturated rank, then carry
        # ``b`` zero-padded to it so the per-level update can be done in place on
        # the strided view (which needs a constant trailing shape, mirroring
        # Dense PCR's in-place aliased reduction — that aliasing is load-bearing
        # for PCR correctness, so an out-of-place rebuild is not an option). The
        # rank is capped at ``nf`` (the structured path degenerates gracefully to
        # dense-equivalent rank in the worst case rather than truncating).
        nblk_probe = v_flat.shape[0]
        niter = max(nblk_probe.bit_length() - 1, 1)
        fixed_rank = max(b.U.shape[-1], ainv.U.shape[-1])
        # Conservative STRUCTURAL fallback (a pure function of ns / nf — no value
        # inspection): the reduced ``-B A^{-1} B`` low-rank is bounded by a small
        # multiple of ns in practice (saturates at ~ns on Taylor data); cap at nf
        # so the worst case degrades to dense-rank rather than truncating.
        fallback = min(b.nf, max(4 * b.ns, fixed_rank))
        # The numerical probe below inspects tensor VALUES (rank via SVD,
        # finite-ness checks, possible LinAlg errors) so it is inherently
        # data-dependent and would graph-break under ``torch.compile``. When
        # tracing we therefore skip it and adopt the structural ``fallback``
        # directly: this keeps ``pcr_init_schur`` a single break-free graph and
        # only ever carries a SAFE (possibly larger) constant rank — correctness
        # is unaffected because the reduce-level recompression truncates/pads to
        # this fixed rank either way. In eager we keep the tighter probed rank
        # for efficiency.
        if torch.compiler.is_compiling():
            fixed_rank = fallback
        elif torch.isfinite(ainv.U).all() and torch.isfinite(ainv.V).all():
            probe = b
            try:
                for _ in range(niter):
                    probe = _carrier_neg(
                        _carrier_mul(_carrier_mul(probe, ainv, tol=tol), probe, tol=tol)
                    )
                    if not torch.isfinite(probe.U).all():
                        fixed_rank = fallback
                        break
                    fixed_rank = max(fixed_rank, probe.U.shape[-1])
            except torch.linalg.LinAlgError:
                fixed_rank = fallback
        else:
            fixed_rank = fallback
        fixed_rank = min(fixed_rank, b.nf)
        b = _carrier_pad_rank(b, fixed_rank)
        # Prepend the PCR tree-window dim W=1 (axis 0) ahead of (nblk, B, ...).
        ainv = _FlatCarrier(
            ainv.Dg.unsqueeze(0).contiguous(),
            ainv.U.unsqueeze(0).contiguous(),
            ainv.V.unsqueeze(0).contiguous(),
            ainv.np_,
            ainv.ns,
            ainv.ngrain,
        )
        b = _FlatCarrier(
            b.Dg.unsqueeze(0).contiguous(),
            b.U.unsqueeze(0).contiguous(),
            b.V.unsqueeze(0).contiguous(),
            b.np_,
            b.ns,
            b.ngrain,
        )
        v_flat = v_flat.unsqueeze(0).contiguous()  # (1, nblk, B, nf, 1)
        return NEML2SchurPCRState(ainv, b, v_flat, self.am.row_layout, tol, fixed_rank)

    def pcr_reduce_level_schur(
        self, state: "NEML2SchurPCRState", level: int
    ) -> "NEML2SchurPCRState":
        """One structured PCR reduction level.

        Implements (in carrier form, batched over the time axis 1):
            v[1:]  -= B[1:]  @ (A[:-1]^{-1}  @ v[:-1])
            b[2:]   = -B[2:] @ (A[1:-1]^{-1} @ B[1:-1])
        then a block cyclic shift of ainv / b / v.

        The updates are performed IN PLACE on the strided ``(W, nblk, ...)``
        views — exactly mirroring the Dense PCR reduction. This is load-bearing:
        the cyclic-shift strided views overlap across the window axis, and the
        in-place writes propagate the reduced blocks through that overlap. An
        out-of-place rebuild (computing all updates from the pre-level values)
        gives the WRONG answer — verified against the Dense reference. Carrying
        ``b`` at a constant low rank (set in ``pcr_init_schur``) is what makes
        the in-place write well-defined despite the structured representation.
        """
        ainv, b, v = state.ainv, state.b, state.v
        tol = state.tol
        R = state.fixed_rank

        # --- v update: v[:, 1:] -= B[:, 1:] @ (A[:, :-1]^{-1} @ v[:, :-1]) ---
        ainv_lo = _carrier_time_slice(ainv, slice(None, -1))
        Av = ainv_lo.matmul_flat(v[:, :-1])  # (W, nblk-1, B, nf, 1)
        b_hi = _carrier_time_slice(b, slice(1, None))
        corr = b_hi.matmul_flat(Av)
        v[:, 1:] -= corr  # in-place on the strided view (matches Dense)

        # --- b update: b[:, 2:] = -B[:, 2:] @ A[:, 1:-1]^{-1} @ B[:, 1:-1] ---
        # Skip when the time axis is too short (nblk <= 2 -> b[:, 2:] empty),
        # mirroring Dense's no-op on the empty slice.
        if b.Dg.shape[1] > 2:
            ainv_mid = _carrier_time_slice(ainv, slice(1, -1))
            b_mid = _carrier_time_slice(b, slice(1, -1))
            b_top = _carrier_time_slice(b, slice(2, None))
            chained = _carrier_mul(
                _carrier_mul(b_top, ainv_mid, tol=tol, fixed_rank=R),
                b_mid,
                tol=tol,
                fixed_rank=R,
            )
            new_b_top = _carrier_neg(chained)
            # In-place write into the strided b views (fixed rank R guarantees
            # shape compatibility).
            b.Dg[:, 2:] = new_b_top.Dg
            b.U[:, 2:] = new_b_top.U
            b.V[:, 2:] = new_b_top.V

        # --- cyclic shift (axis 0/1 doubling/halving), exactly like Dense ---
        return NEML2SchurPCRState(
            _carrier_cyclic_shift(ainv, level),
            _carrier_cyclic_shift(b, level),
            _dense_pcr_cyclic_shift(v, level),
            state.layout,
            tol,
            R,
        )

    def pcr_finalize_schur(
        self, state: "NEML2SchurPCRState"
    ) -> tuple["NEML2SolvableBlockOperator", "NEML2BlockVector"]:
        """Extract the reduced subdiagonal operator (nblk-1 blocks) and RHS.

        ``v_red`` is the flat tail un-split back to the multi-group layout.
        ``B_red`` carries the REAL reduced subdiagonal values (no zero
        template): each reduced step's carrier is materialized to its flat
        ``(nf, nf)`` block and re-embedded as a 2-group AssembledMatrix so the
        downstream ``B[s+1:e] = B_red`` assignment (and the PCR matvec) carries
        correct values.
        """
        layout = state.layout
        # After all reductions the time axis (axis 1) is length 1 and the window
        # axis (axis 0) holds the reduced blocks — mirror Dense ``squeeze(1)[1:]``.
        # v_red: squeeze nblk axis, take window tail [1:], squeeze the rhs col.
        v_tail = state.v.squeeze(1)[1:].squeeze(-1).contiguous()  # (W-1, B, nf)
        v_red = NEML2BlockVector.from_av(_split_flat_to_av(v_tail, layout))

        # B_red: squeeze nblk axis, take the carrier window tail [1:], densify.
        b = state.b
        b_tail = _FlatCarrier(
            b.Dg.squeeze(1)[1:],
            b.U.squeeze(1)[1:],
            b.V.squeeze(1)[1:],
            b.np_,
            b.ns,
            b.ngrain,
        )
        B_red_flat = b_tail.dense().contiguous()  # (nblk-1, B, nf, nf)
        B_red_am = _flat_to_sub_am(B_red_flat, layout)
        B_red = NEML2SolvableBlockOperator(B_red_am)
        return B_red, v_red

    # NOTE: the shape-only ``identity``/``from_diagonal`` constructors are no
    # longer part of the BlockOperator interface (they cannot express the NEML2
    # multi-group layout). Use :meth:`identity_with_layout`, or wrap an
    # ``AssembledMatrix`` directly via ``NEML2SolvableBlockOperator(am)``.

    @classmethod
    def identity_with_layout(
        cls,
        nblk: int,
        batch_size: int,
        layout: "AxisLayout",
        dtype: torch.dtype,
        device: torch.device,
    ) -> "NEML2SolvableBlockOperator":
        """Build an identity operator matching the given NEML2 axis layout."""
        ng = layout.ngroup()
        T = [[Tensor() for _ in range(ng)] for _ in range(ng)]
        for g in range(ng):
            intmd_sizes = _group_intmd_sizes(layout, g)
            # Per-group flat DOF count (intmd × Σbase); NOT the variable-index
            # offsets that ``layout.group_offsets`` returns.
            group_dofs = _group_flat_size(layout, g)
            if intmd_sizes:
                base_flat = group_dofs // int(prod(intmd_sizes))
                eye = torch.eye(base_flat, dtype=dtype, device=device)
                shape = (nblk, batch_size, *intmd_sizes, base_flat, base_flat)
                I_raw = eye.expand(shape).contiguous()
            else:
                eye = torch.eye(group_dofs, dtype=dtype, device=device)
                I_raw = eye.expand(
                    nblk, batch_size, group_dofs, group_dofs
                ).contiguous()
            T[g][g] = Tensor(I_raw, I_raw.ndim - len(intmd_sizes) - 2, len(intmd_sizes))
        am = AssembledMatrix(layout, layout, T)
        return cls(am)
