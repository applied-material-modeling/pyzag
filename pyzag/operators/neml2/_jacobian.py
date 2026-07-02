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

"""NEML2-backend bidiagonal block Jacobian."""

from __future__ import annotations

import torch

from ..base import BlockJacobian, BlockVector
from ._containers import AssembledMatrix, Tensor
from ._assembly import _am_to_flat, _layout_flat_size, _split_flat_to_av, _transpose_am
from ._vector import NEML2BlockVector
from ._operator import NEML2SolvableBlockOperator


class NEML2BlockJacobian(BlockJacobian):
    """Per-chunk Jacobian wrapping NEML2 AssembledMatrix for diag and sub.

    The adjoint path uses regular `NEML2SolvableBlockOperator` instances
    constructed from `_transpose_am(diag)` and `_transpose_am(sub)` — mirroring
    how DenseBlockJacobian uses `.transpose(-1, -2)` + regular DenseBlockOperator.
    This gives full SolvableBlockOperator API (matvec, t_matvec, solve, PCR)
    on both forward and adjoint paths automatically.
    """

    def __init__(
        self,
        diag_am: "AssembledMatrix",
        sub_am: "AssembledMatrix",
        layout: "AxisLayout",
        _reversed: bool = False,
    ) -> None:
        self.diag_am = diag_am
        self.sub_am = sub_am
        self._layout = layout
        self._reversed = _reversed

    def _first_defined_diag_raw(self) -> torch.Tensor:
        """Return the underlying torch tensor of any defined diagonal block.

        All four shape / device / dtype properties share this lookup so they
        raise a clear error rather than a cryptic NEML2 one when no diagonal
        block is defined.
        """
        for i in range(self.diag_am.row_layout.ngroup()):
            for j in range(self.diag_am.col_layout.ngroup()):
                blk = self.diag_am.tensors[i][j]
                if blk.defined():
                    return blk.torch()
        raise RuntimeError(
            "NEML2BlockJacobian: no defined block in diag_am. "
            "_expand_am_dynamic should have zero-filled undefined blocks."
        )

    @property
    def device(self) -> torch.device:
        return self._first_defined_diag_raw().device

    @property
    def dtype(self) -> torch.dtype:
        return self._first_defined_diag_raw().dtype

    @property
    def nblk_steps(self) -> int:
        return self._first_defined_diag_raw().shape[0]

    @property
    def batch_size(self) -> int:
        return self._first_defined_diag_raw().shape[1]

    @property
    def block_size(self) -> int:
        return _layout_flat_size(self._layout)

    # ---- helpers ----

    def _walk_diag(self) -> "AssembledMatrix":
        """Diagonal in walk order (flipped along nblk if reversed)."""
        if not self._reversed:
            return self.diag_am
        return self._flip_am(self.diag_am)

    def _walk_sub(self) -> "AssembledMatrix":
        if not self._reversed:
            return self.sub_am
        return self._flip_am(self.sub_am)

    @staticmethod
    def _flip_am(am: "AssembledMatrix") -> "AssembledMatrix":
        n_rows = am.row_layout.ngroup()
        n_cols = am.col_layout.ngroup()
        T = [[Tensor() for _ in range(n_cols)] for _ in range(n_rows)]
        for i in range(n_rows):
            for j in range(n_cols):
                blk = am.tensors[i][j]
                if blk.defined():
                    T[i][j] = Tensor(
                        blk.torch().flip(0), blk.dynamic.dim(), blk.intmd.dim()
                    )
        return AssembledMatrix(am.row_layout, am.col_layout, T)

    @staticmethod
    def _slice_am(am: "AssembledMatrix", idx) -> "AssembledMatrix":
        """Slice the nblk dim of every block."""
        n_rows = am.row_layout.ngroup()
        n_cols = am.col_layout.ngroup()
        T = [[Tensor() for _ in range(n_cols)] for _ in range(n_rows)]
        for i in range(n_rows):
            for j in range(n_cols):
                blk = am.tensors[i][j]
                if blk.defined():
                    blk_raw = blk.torch()
                    raw = blk_raw[idx]
                    if isinstance(idx, int) or raw.ndim < blk_raw.ndim:
                        raw = raw.unsqueeze(0)
                    T[i][j] = Tensor(raw, blk.dynamic.dim(), blk.intmd.dim())
        return AssembledMatrix(am.row_layout, am.col_layout, T)

    # ---- BlockJacobian abstract methods ----

    def forward_system(self, inverse_operator):
        from pyzag.chunktime import BidiagonalForwardOperator

        if self._reversed:
            raise RuntimeError(
                "forward_system() must be called on a forward-walk BlockJacobian, "
                "not one returned by as_adjoint_walk()."
            )
        A_ops = NEML2SolvableBlockOperator.factored(self.diag_am)
        B_ops = NEML2SolvableBlockOperator(self._slice_am(self.sub_am, slice(1, None)))
        return BidiagonalForwardOperator(
            A_ops, B_ops, inverse_operator=inverse_operator
        )

    def adjoint_system(self, inverse_operator):
        if not self._reversed:
            raise RuntimeError(
                "adjoint_system() must be called on the BlockJacobian returned by "
                "as_adjoint_walk(), not the forward one."
            )
        diag_walk = self._walk_diag()
        sub_walk = self._walk_sub()
        A_T = _transpose_am(self._slice_am(diag_walk, slice(1, None)))
        B_T = _transpose_am(self._slice_am(sub_walk, slice(1, -1)))
        A_ops = NEML2SolvableBlockOperator.factored(A_T)
        B_ops = NEML2SolvableBlockOperator(B_T)
        return inverse_operator(A_ops, B_ops)

    def solve_terminal_adjoint(self, g_terminal: torch.Tensor) -> NEML2BlockVector:
        # The terminal block is the very last forward-time diagonal block.
        terminal = self._slice_am(self.diag_am, slice(-1, None))
        term_T = _transpose_am(terminal)
        op = NEML2SolvableBlockOperator.factored(term_T)
        # Wrap g_terminal as NEML2BlockVector (single-block).
        g_bv = NEML2BlockVector.from_av(
            _split_flat_to_av(g_terminal.unsqueeze(0), self._layout)
        )
        sol = op.solve(g_bv)
        # The terminal adjoint is -A_T^{-1} g.
        return NEML2BlockVector(
            [-t for t in sol.raw_tensors], sol.layout, sol.intmd_dims
        )

    def couple_prev_chunk(self, a_first: BlockVector) -> NEML2BlockVector:
        if not isinstance(a_first, NEML2BlockVector):
            raise TypeError(
                "NEML2BlockJacobian.couple_prev_chunk expects NEML2BlockVector."
            )
        sub_walk = self._walk_sub()
        # The boundary block is sub_walk[0] (the first walked subdiagonal).
        boundary = self._slice_am(sub_walk, slice(0, 1))
        op = NEML2SolvableBlockOperator(boundary)
        return op.t_matvec(a_first)

    def as_adjoint_walk(self) -> "NEML2BlockJacobian":
        return NEML2BlockJacobian(
            self.diag_am, self.sub_am, self._layout, _reversed=not self._reversed
        )

    def to_dense(self):
        """Materialize as a :class:`DenseBlockJacobian` over the flat per-step layout.

        Folds the BLOCK group's intmd dim into the flat row/col axes via
        :func:`_am_to_flat`, producing ``(nblk_steps, batch, n_flat, n_flat)``
        where ``n_flat`` matches the per-step state size of
        :func:`_av_to_flat` / :func:`_split_flat_to_av` — so the downstream
        :class:`DenseBlockJacobian` operates on the flat state the pyzag wrapper
        boundary already exposes.

        Drives the "FlatDense" variant of the Taylor mix-mode comparison (see
        ``examples/taylor_comparison/``), replacing the BLOCK+DENSE Schur path
        with a single LU on the flat per-step Jacobian. Inherits
        :func:`_am_to_flat`'s single-intmd restriction.
        """
        from pyzag.operators.dense import DenseBlockJacobian

        diag_flat = _am_to_flat(self.diag_am)
        sub_flat = _am_to_flat(self.sub_am)
        return DenseBlockJacobian(
            diag=diag_flat, sub=sub_flat, _reversed=self._reversed
        )
