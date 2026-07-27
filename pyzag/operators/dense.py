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

"""
Packed block operators and vectors with dense tensor storage.
"""

from __future__ import annotations

import os
from math import prod
from typing import Sequence

import torch
from torch.nn.functional import pad

from pyzag.chunktime import BidiagonalForwardOperator
from .base import (
    BlockJacobian,
    BlockOperator,
    BlockVector,
    PCRState,
    SolvableBlockOperator,
)

_CUDA_BATCHED_LU_MAX_N_DEFAULT = 256
"""cuSOLVER's batched LU is tuned for very small matrices. Above this
per-matrix size, batched ``getrf``/``getrs`` print a "batched routines are
designed for small sizes" warning and run slower than the non-batched
("Native") path. We loop the leading dims and call non-batched LU when the
factorized matrix is larger than this threshold on cuda.

torch does not expose this crossover. Override it with the
``PYZAG_CUDA_BATCHED_LU_MAX_N`` environment variable to
retune for a different GPU."""


def _read_cuda_batched_lu_max_n() -> int:
    """Read the cuda batched-LU size threshold from the environment, falling
    back to :data:`_CUDA_BATCHED_LU_MAX_N_DEFAULT`. See that constant for the
    meaning and provenance of the value."""
    raw = os.environ.get("PYZAG_CUDA_BATCHED_LU_MAX_N")
    if raw is None:
        return _CUDA_BATCHED_LU_MAX_N_DEFAULT
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError(
            f"PYZAG_CUDA_BATCHED_LU_MAX_N must be an integer, got {raw!r}."
        ) from e
    if value < 0:
        raise ValueError(
            f"PYZAG_CUDA_BATCHED_LU_MAX_N must be non-negative, got {value}."
        )
    return value


_CUDA_BATCHED_LU_MAX_N = _read_cuda_batched_lu_max_n()


def _lu_factor_guarded(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """LU factorization that avoids cuSOLVER's "batched routines" warning for
    large matrices on cuda. See :data:`_CUDA_BATCHED_LU_MAX_N`."""
    n = A.shape[-1]
    if A.device.type != "cuda" or n <= _CUDA_BATCHED_LU_MAX_N:
        lu, piv, _ = torch.linalg.lu_factor_ex(A)
        return lu, piv
    lead_shape = A.shape[:-2]
    flat = A.reshape(-1, n, n)
    lus_flat = torch.empty_like(flat)
    pivs_flat = torch.empty(flat.shape[0], n, dtype=torch.int32, device=A.device)
    for i in range(flat.shape[0]):
        lu_i, piv_i, _ = torch.linalg.lu_factor_ex(flat[i])
        lus_flat[i] = lu_i
        pivs_flat[i] = piv_i
    return lus_flat.reshape(*lead_shape, n, n), pivs_flat.reshape(*lead_shape, n)


def _lu_solve_guarded(
    lu: torch.Tensor, piv: torch.Tensor, B: torch.Tensor
) -> torch.Tensor:
    """Companion of :func:`_lu_factor_guarded`: loops over leading dims for
    large matrices to avoid the cuSOLVER batched-getrs warning."""
    n = lu.shape[-1]
    if lu.device.type != "cuda" or n <= _CUDA_BATCHED_LU_MAX_N:
        return torch.linalg.lu_solve(lu, piv, B)
    rhs_cols = B.shape[-1]
    lead_shape = lu.shape[:-2]
    lu_flat = lu.reshape(-1, n, n)
    piv_flat = piv.reshape(-1, n)
    B_flat = B.reshape(-1, n, rhs_cols)
    out = torch.empty_like(B_flat)
    for i in range(B_flat.shape[0]):
        out[i] = torch.linalg.lu_solve(lu_flat[i], piv_flat[i], B_flat[i])
    return out.reshape(*lead_shape, n, rhs_cols)


def batch_lu_solve(
    lu: torch.Tensor, pivots: torch.Tensor, rhs: torch.Tensor
) -> torch.Tensor:
    """Batched version of torch.linalg.lu_solve that accepts separate LU and pivot tensors."""
    return _lu_solve_guarded(lu, pivots, rhs)


def _dense_pcr_cyclic_shift(A: torch.Tensor, level: int) -> torch.Tensor:
    """Perform a cyclic shift on a dense tensor for PCR reduction."""
    return A.as_strided(
        (A.shape[0] * 2, A.shape[1] // 2) + A.shape[2:],
        (prod(A.shape[2:]), 2 ** (level + 1) * prod(A.shape[2:])) + A.stride()[2:],
    )


class DenseBlockVector(BlockVector):
    """Dense tensor-backed packed block vector.

    Args:
        data (torch.Tensor): shape (nblk, sbat, sblk)
    """

    def __init__(self, data: torch.Tensor) -> None:
        self.data = data

    @property
    def device(self) -> torch.device:
        """Execution device of the backing tensor."""
        return self.data.device

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the backing tensor."""
        return self.data.dtype

    @property
    def nblk(self) -> int:
        """Number of logical blocks (axis 0 of ``data``)."""
        return self.data.shape[0]

    @property
    def batch_size(self) -> int:
        """Logical batch size (axis 1 of ``data``)."""
        return self.data.shape[1]

    @property
    def block_size(self) -> int:
        """Logical size of one block (last axis of ``data``)."""
        return self.data.shape[-1]

    def clone(self) -> DenseBlockVector:
        """Return a safe copy backed by a cloned tensor."""
        return DenseBlockVector(self.data.clone())

    def norm(self, dim: int = -1) -> torch.Tensor:
        """Compute the norm along ``dim``. Returns a raw tensor (used for
        scalar convergence checks; not wrapped as a block vector)."""
        return torch.norm(self.data, dim=dim)

    def flatten(self) -> torch.Tensor:
        """Flatten to a raw ``(batch_size, nblk * block_size)`` tensor:
        transpose the block axis behind the batch axis and flatten the rest,
        so each batch element's entries are concatenated along the last axis."""
        return self.data.transpose(0, 1).flatten(1)

    def where(self, mask: torch.Tensor, other: BlockVector) -> DenseBlockVector:
        """Batch-axis conditional combination (``torch.where`` convention):
        return ``self`` where ``mask`` is True, else ``other``. ``mask`` is a
        ``(batch_size,)`` tensor broadcast over the block and state axes."""
        if not isinstance(other, DenseBlockVector):
            raise TypeError("DenseBlockVector.where expects DenseBlockVector.")
        # Broadcast mask (batch,) over (nblk, batch, *state): leading
        # singleton for the block axis, trailing singletons for state dims.
        broadcast_shape = (1, -1) + (1,) * (self.data.ndim - 2)
        return DenseBlockVector(
            torch.where(mask.reshape(broadcast_shape), self.data, other.data)
        )

    def scale_batches(self, factor: torch.Tensor) -> DenseBlockVector:
        """Multiply each batch element by its own scalar from ``factor``
        (shape ``(batch_size,)``), reshaped to ``(1, batch_size, 1, ...)`` so
        it broadcasts over the block and state axes rather than the trailing
        (state) axis. Used by line-search backtracking."""
        broadcast_shape = (1, -1) + (1,) * (self.data.ndim - 2)
        return DenseBlockVector(self.data * factor.reshape(broadcast_shape))

    def flip(self, dim: int) -> DenseBlockVector:
        """Return a new vector with axis ``dim`` reversed."""
        return DenseBlockVector(self.data.flip(dim))

    def __neg__(self) -> DenseBlockVector:
        """Return the negated vector."""
        return DenseBlockVector(-self.data)

    def __add__(self, other: BlockVector) -> DenseBlockVector:
        """Element-wise addition with another ``DenseBlockVector``."""
        if not isinstance(other, DenseBlockVector):
            raise TypeError("DenseBlockVector can only add to DenseBlockVector.")
        return DenseBlockVector(self.data + other.data)

    def __sub__(self, other: BlockVector) -> DenseBlockVector:
        """Element-wise subtraction with another ``DenseBlockVector``."""
        if not isinstance(other, DenseBlockVector):
            raise TypeError("DenseBlockVector can only subtract from DenseBlockVector.")
        return DenseBlockVector(self.data - other.data)

    def __mul__(self, other: torch.Tensor | float | int) -> DenseBlockVector:
        """Scalar/broadcast multiplication."""
        return DenseBlockVector(self.data * other)

    def __getitem__(self, idx: int | slice) -> DenseBlockVector:
        """Slice along the block dimension, preserving that axis (a bare int
        result is re-expanded to a single-block vector)."""
        sliced = self.data[idx]
        if sliced.ndim == 2:
            sliced = sliced.unsqueeze(0)
        return DenseBlockVector(sliced)

    def __setitem__(self, idx: int | slice, value: BlockVector) -> None:
        """Assign into a block-dim slice from another ``DenseBlockVector``."""
        if not isinstance(value, DenseBlockVector):
            raise TypeError("DenseBlockVector can only assign from DenseBlockVector.")
        self.data[idx] = value.data

    @classmethod
    def cat(cls, vectors: Sequence[BlockVector], dim: int = 0) -> DenseBlockVector:
        """Concatenate a sequence of ``DenseBlockVector`` along ``dim``."""
        for v in vectors:
            if not isinstance(v, DenseBlockVector):
                raise TypeError("All vectors must be DenseBlockVector.")
        return DenseBlockVector(torch.cat([v.data for v in vectors], dim=dim))

    @classmethod
    def zeros(
        cls,
        nblk: int,
        batch_size: int,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> DenseBlockVector:
        """Return a zero-filled DenseBlockVector of the given shape."""
        return DenseBlockVector(
            torch.zeros(nblk, batch_size, block_size, dtype=dtype, device=device)
        )

    @classmethod
    def zeros_like(cls, other: BlockVector) -> DenseBlockVector:
        """Return a zero-filled ``DenseBlockVector`` with the same shape as
        ``other``."""
        if not isinstance(other, DenseBlockVector):
            raise TypeError("DenseBlockVector.zeros_like requires DenseBlockVector.")
        return DenseBlockVector(torch.zeros_like(other.data))

    @classmethod
    def empty(
        cls,
        nblk: int,
        batch_size: int,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> DenseBlockVector:
        """Return an uninitialized DenseBlockVector of the given shape."""
        return DenseBlockVector(
            torch.empty(nblk, batch_size, block_size, dtype=dtype, device=device)
        )


class DensePCRState(PCRState):
    """Dense backend PCR working state.

    Holds the four working tensors used by the Dense cyclic-shift PCR kernel.
    Each tensor has an extra leading dimension prepended by :meth:`pcr_init`
    so that :func:`_dense_pcr_cyclic_shift` can double it at each level.

    Attributes:
        lu (torch.Tensor): shape ``(1, nblk, sbat, sblk, sblk)`` initially.
        pivots (torch.Tensor): shape ``(1, nblk, sbat, sblk)`` initially.
        b (torch.Tensor): shape ``(1, nblk, sbat, sblk, sblk)`` initially.
        v (torch.Tensor): shape ``(1, nblk, sbat, sblk, 1)`` initially.
    """

    def __init__(
        self,
        lu: torch.Tensor,
        pivots: torch.Tensor,
        b: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        self.lu = lu
        self.pivots = pivots
        self.b = b
        self.v = v


class DenseBlockOperator(SolvableBlockOperator):
    """Dense tensor-backed packed block operator.

    Implements :class:`SolvableBlockOperator` (LU-based ``solve`` and
    PCR primitives ``pcr_init`` / ``pcr_reduce_level`` / ``pcr_finalize``).

    The cached LU factorization (``self.lu`` / ``self.pivots``) is
    materialized lazily on the first ``solve`` or ``pcr_init`` call.
    Use :meth:`factored` to construct with eager factorization for cases
    where you know LU will be needed (e.g., the diagonal of a Newton
    system). Slicing, cloning, and in-place assignment preserve the
    cached LU when present, so chained Thomas-style ``A[i:i+1].solve(...)``
    calls don't re-factor every block.

    Args:
        data (torch.Tensor): shape ``(nblk, sbat, sblk, sblk)``.
        lu (torch.Tensor, optional): pre-computed LU factor with the
            same shape as ``data``.
        pivots (torch.Tensor, optional): pre-computed pivots, shape
            ``(nblk, sbat, sblk)``. Must be paired with ``lu``.
    """

    def __init__(
        self,
        data: torch.Tensor,
        lu: torch.Tensor | None = None,
        pivots: torch.Tensor | None = None,
    ) -> None:
        if data.ndim != 4:
            raise ValueError(
                "DenseBlockOperator expects shape (nblk, sbat, sblk, sblk)."
            )
        if data.shape[-1] != data.shape[-2]:
            raise ValueError("DenseBlockOperator requires square blocks.")
        if (lu is None) != (pivots is None):
            raise ValueError("lu and pivots must both be provided or both None.")
        self.data = data
        self.lu = lu
        self.pivots = pivots

    @classmethod
    def factored(cls, data: torch.Tensor) -> DenseBlockOperator:
        """Construct with eager LU factorization. Use when the caller
        knows ``solve`` or ``pcr_init`` will be invoked."""
        op = cls(data)
        op._ensure_lu()
        return op

    def _ensure_lu(self) -> None:
        if self.lu is None:
            self.lu, self.pivots = _lu_factor_guarded(self.data)

    @property
    def device(self) -> torch.device:
        """Execution device of the backing tensor."""
        return self.data.device

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the backing tensor."""
        return self.data.dtype

    @property
    def nblk(self) -> int:
        """Number of logical blocks (axis 0 of ``data``)."""
        return self.data.shape[0]

    @property
    def batch_size(self) -> int:
        """Logical batch size expected in block-major vector inputs."""
        return self.data.shape[1]

    def matvec(self, x: BlockVector) -> DenseBlockVector:
        """Apply the (block-diagonal) operator to ``x``: per block, the
        matrix-vector product ``A_k x_k``. Requires ``x.nblk == self.nblk``."""
        if not isinstance(x, DenseBlockVector):
            raise TypeError("DenseBlockOperator.matvec expects DenseBlockVector.")
        nblk, sbat, sblk = x.data.shape
        return DenseBlockVector(
            torch.bmm(
                self.data.view(nblk * sbat, sblk, sblk),
                x.data.view(nblk * sbat, sblk, 1),
            ).view(nblk, sbat, sblk)
        )

    def t_matvec(self, x: BlockVector) -> DenseBlockVector:
        """Apply the transpose of the operator to ``x`` (per block,
        ``A_k^T x_k``). Requires ``x.nblk == self.nblk``."""
        if not isinstance(x, DenseBlockVector):
            raise TypeError("DenseBlockOperator.t_matvec expects DenseBlockVector.")
        nblk, sbat, sblk = x.data.shape
        return DenseBlockVector(
            torch.bmm(
                self.data.view(nblk * sbat, sblk, sblk).transpose(-1, -2),
                x.data.view(nblk * sbat, sblk, 1),
            ).view(nblk, sbat, sblk)
        )

    def solve(self, rhs: BlockVector) -> DenseBlockVector:
        """Solve the block system ``A x = rhs`` per block via the cached LU
        factorization (materialized lazily on first call). Requires
        ``rhs.nblk == self.nblk``."""
        if not isinstance(rhs, DenseBlockVector):
            raise TypeError("DenseBlockOperator.solve expects DenseBlockVector.")
        self._ensure_lu()
        return DenseBlockVector(
            batch_lu_solve(self.lu, self.pivots, rhs.data.unsqueeze(-1)).squeeze(-1)
        )

    def clone(self) -> DenseBlockOperator:
        """Return a safe copy, cloning the cached LU factors if present."""
        return DenseBlockOperator(
            self.data.clone(),
            None if self.lu is None else self.lu.clone(),
            None if self.pivots is None else self.pivots.clone(),
        )

    def __getitem__(self, idx: int | slice) -> DenseBlockOperator:
        """Return a logical block window, carrying the cached LU/pivots for the
        same window so chained ``A[i:i+1].solve(...)`` calls avoid re-factoring.
        A bare-int result is re-expanded to a single-block operator."""
        sliced_data = self.data[idx]
        sliced_lu = None if self.lu is None else self.lu[idx]
        sliced_pivots = None if self.pivots is None else self.pivots[idx]
        if sliced_data.ndim == 3:
            sliced_data = sliced_data.unsqueeze(0)
            if sliced_lu is not None:
                sliced_lu = sliced_lu.unsqueeze(0)
                sliced_pivots = sliced_pivots.unsqueeze(0)
        return DenseBlockOperator(sliced_data, sliced_lu, sliced_pivots)

    def __setitem__(self, idx: int | slice, other: BlockOperator) -> None:
        """Overwrite a logical block window with ``other``. The cached LU is
        kept consistent when both sides have it, otherwise invalidated so the
        next ``solve`` re-factors."""
        if not isinstance(other, DenseBlockOperator):
            raise TypeError(
                "DenseBlockOperator assignment requires DenseBlockOperator."
            )
        self.data[idx].copy_(other.data)
        # Keep cached LU consistent if both sides have it; otherwise
        # invalidate so the next solve re-factors.
        if self.lu is not None and other.lu is not None:
            self.lu[idx].copy_(other.lu)
            self.pivots[idx].copy_(other.pivots)
        else:
            self.lu = None
            self.pivots = None

    def pad_front(self, n: int = 1) -> DenseBlockOperator:
        """Return an operator with ``n`` leading zero (dummy) blocks prepended.
        Creates new data rather than a view; the result is un-factored since
        padding changes the per-block layout."""
        if n < 0:
            raise ValueError("n must be nonnegative.")
        if n == 0:
            return self.clone()
        # Padding changes the per-block identity, so any cached LU is
        # invalid for the new layout; build a fresh (un-factored) op.
        data = pad(self.data, (0, 0, 0, 0, 0, 0, n, 0))
        return DenseBlockOperator(data)

    def pcr_init(self, B: BlockOperator, v: BlockVector) -> DensePCRState:
        """Initialise Dense PCR working state for a power-of-two window.

        The state owns its ``b``/``v`` working buffers (cloned here), so the
        in-place reductions in :meth:`pcr_reduce_level` never mutate the caller's
        operator/vector -- honouring the base-class contract that the backend
        allocates internal working tensors.
        """
        if not isinstance(B, DenseBlockOperator):
            raise TypeError("B must be DenseBlockOperator.")
        if not isinstance(v, DenseBlockVector):
            raise TypeError("v must be DenseBlockVector.")
        self._ensure_lu()
        return DensePCRState(
            lu=self.lu.unsqueeze(0),
            pivots=self.pivots.unsqueeze(0),
            b=B.data.unsqueeze(0).clone(),
            v=v.data.unsqueeze(0).unsqueeze(-1).clone(),
        )

    def pcr_reduce_level(self, state: PCRState, level: int) -> DensePCRState:
        """Apply one Dense PCR level: update v and B, then cyclic-shift all four tensors."""
        if not isinstance(state, DensePCRState):
            raise TypeError("state must be DensePCRState.")
        lu, pivots, b, v = state.lu, state.pivots, state.b, state.v

        v[:, 1:] -= torch.matmul(
            b[:, 1:],
            _lu_solve_guarded(lu[:, :-1], pivots[:, :-1], v[:, :-1]),
        )
        b[:, 2:] = -torch.matmul(
            b[:, 2:],
            _lu_solve_guarded(lu[:, 1:-1], pivots[:, 1:-1], b[:, 1:-1]),
        )
        return DensePCRState(
            lu=_dense_pcr_cyclic_shift(lu, level),
            pivots=_dense_pcr_cyclic_shift(pivots, level),
            b=_dense_pcr_cyclic_shift(b, level),
            v=_dense_pcr_cyclic_shift(v, level),
        )

    def pcr_finalize(
        self, state: PCRState
    ) -> tuple[DenseBlockOperator, DenseBlockVector]:
        """Extract (B_red, v_red) with nblk = window_size - 1 from the final PCR state."""
        if not isinstance(state, DensePCRState):
            raise TypeError("state must be DensePCRState.")
        return (
            DenseBlockOperator(state.b.squeeze(1)[1:].clone()),
            DenseBlockVector(state.v.squeeze(1)[1:].squeeze(-1).clone()),
        )

    @classmethod
    def identity(
        cls,
        nblk: int,
        batch_size: int,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> DenseBlockOperator:
        """Return an identity DenseBlockOperator of the given shape."""
        eye = torch.eye(block_size, dtype=dtype, device=device)
        data = eye.expand(nblk, batch_size, block_size, block_size).contiguous()
        return cls(data)

    @classmethod
    def from_diagonal(cls, data: torch.Tensor) -> DenseBlockOperator:
        """Return a DenseBlockOperator built from per-block diagonal data."""
        return cls(data)


class DenseBlockJacobian(BlockJacobian):
    """Dense tensor-backed per-chunk Jacobian.

    Storage is canonical (forward-time order). The :meth:`as_adjoint_walk`
    method returns a sibling instance that shares the same underlying
    tensors and only flips a private flag; the four adjoint methods read
    the data with appropriate slicing/indexing under that flag, so a
    physical ``flip`` never materializes.

    Args:
        diag (torch.Tensor): per-step diagonal blocks ``dR[k]/dx[k]``,
            shape ``(nblk_steps, batch, n, n)``.
        sub (torch.Tensor): per-step subdiagonal blocks
            ``dR[k]/dx[k-1]``, shape ``(nblk_steps, batch, n, n)``. The
            first block ``sub[0]`` is the inter-chunk boundary coupling
            to the lookback / previous chunk's last state.
    """

    def __init__(
        self,
        diag: torch.Tensor,
        sub: torch.Tensor,
        _reversed: bool = False,
    ) -> None:
        if diag.ndim != 4 or sub.ndim != 4:
            raise ValueError(
                "DenseBlockJacobian expects diag and sub of shape "
                "(nblk_steps, batch, n, n)."
            )
        if diag.shape != sub.shape:
            raise ValueError("diag and sub must have the same shape.")
        if diag.shape[-1] != diag.shape[-2]:
            raise ValueError("DenseBlockJacobian requires square blocks.")
        self.diag = diag
        self.sub = sub
        self._reversed = _reversed

    @classmethod
    def from_stacked(cls, J: torch.Tensor) -> DenseBlockJacobian:
        """Construct from a stacked tensor of shape
        ``(2, nblk_steps, batch, n, n)`` where ``J[1]`` is the diagonal
        and ``J[0]`` is the subdiagonal (with ``J[0, 0]`` being the
        boundary block).
        """
        if J.ndim != 5 or J.shape[0] != 2:
            raise ValueError(
                "DenseBlockJacobian.from_stacked expects shape "
                "(2, nblk_steps, batch, n, n)."
            )
        return cls(diag=J[1], sub=J[0])

    @property
    def device(self) -> torch.device:
        """Execution device of the backing tensors."""
        return self.diag.device

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the backing tensors."""
        return self.diag.dtype

    @property
    def nblk_steps(self) -> int:
        """Number of residual rows in the chunk (axis 0 of ``diag``)."""
        return self.diag.shape[0]

    @property
    def batch_size(self) -> int:
        """Logical batch size (axis 1 of ``diag``)."""
        return self.diag.shape[1]

    @property
    def block_size(self) -> int:
        """Per-step state size (last axis of ``diag``)."""
        return self.diag.shape[-1]

    def _walk_diag(self) -> torch.Tensor:
        """Diagonal in walk order (forward by default; reversed if
        :meth:`as_adjoint_walk` was called)."""
        return self.diag.flip(0) if self._reversed else self.diag

    def _walk_sub(self) -> torch.Tensor:
        """Subdiagonal in walk order (forward by default; reversed if
        :meth:`as_adjoint_walk` was called). Reversing flips the block axis, so
        the forward boundary block ``sub[0]`` moves to the end and the original
        last forward block ``sub[-1]`` becomes index 0 -- the block
        :meth:`couple_prev_chunk` uses for inter-chunk adjoint coupling.
        """
        return self.sub.flip(0) if self._reversed else self.sub

    def forward_system(self, inverse_operator):
        """Build the chunk's forward bidiagonal system for Newton: diagonal
        ``A`` = the (factored) per-step ``diag`` blocks and subdiagonal ``B`` =
        ``sub[1:]``, wrapped in a :class:`BidiagonalForwardOperator`. Must be
        called on a forward-walk Jacobian."""
        if self._reversed:
            raise RuntimeError(
                "forward_system() must be called on a forward-walk "
                "BlockJacobian, not one returned by as_adjoint_walk()."
            )
        A_ops = DenseBlockOperator.factored(self.diag)
        B_ops = DenseBlockOperator(self.sub[1:])
        return BidiagonalForwardOperator(
            A_ops, B_ops, inverse_operator=inverse_operator
        )

    def adjoint_system(self, inverse_operator):
        """Build the chunk's adjoint bidiagonal *solve* operator, in
        adjoint-walk order with transposes baked in: drop the first walked
        diagonal block (handled by the terminal seed) and the first/last walked
        subdiagonal blocks. Must be called on the walk-order Jacobian from
        :meth:`as_adjoint_walk`; applying ``.matvec(rhs)`` yields the adjoint
        solution."""
        if not self._reversed:
            raise RuntimeError(
                "adjoint_system() must be called on the BlockJacobian "
                "returned by as_adjoint_walk(), not the forward one."
            )
        # In walk order: drop the first block of A (it's handled by the
        # terminal seed); drop first and last of B.
        diag_walk = self._walk_diag()
        sub_walk = self._walk_sub()
        A_ops = DenseBlockOperator.factored(diag_walk[1:].transpose(-1, -2))
        B_ops = DenseBlockOperator(sub_walk[1:-1].transpose(-1, -2))
        return inverse_operator(A_ops, B_ops)

    def solve_terminal_adjoint(self, g_terminal: torch.Tensor) -> DenseBlockVector:
        """Compute ``-A_terminal^{-T} @ g_terminal`` for the very last
        forward-time step, returned as a single-block ``DenseBlockVector`` that
        does not alias ``g_terminal``."""
        # The "terminal" block is the very last forward-time diagonal
        # block; this method may be called on either a forward or
        # walk-order Jacobian since both refer to the same trajectory.
        terminal = self.diag[-1]
        adjoint_data = -torch.linalg.solve(
            terminal.transpose(-1, -2), g_terminal
        ).unsqueeze(0)
        return DenseBlockVector(adjoint_data)

    def couple_prev_chunk(self, a_first: BlockVector) -> DenseBlockVector:
        """Compute the inter-chunk adjoint coupling ``B_boundary^T @ a_first``,
        where ``a_first`` is the previous chunk's adjoint tail (single-block, in
        walk order). Returns a single-block ``DenseBlockVector`` to subtract into
        the current chunk's first-row RHS."""
        if not isinstance(a_first, DenseBlockVector):
            raise TypeError(
                "DenseBlockJacobian.couple_prev_chunk expects DenseBlockVector."
            )
        # In walk order, the boundary block (originally sub[0]) sits at
        # the end of the walked subdiagonal, but the inter-chunk coupling
        # uses the *first* walked block, which is the original sub[-1].
        sub_walk = self._walk_sub()
        coupling = torch.matmul(
            sub_walk[0].transpose(-1, -2), a_first.data[0].unsqueeze(-1)
        ).squeeze(-1)
        return DenseBlockVector(coupling.unsqueeze(0))

    def as_adjoint_walk(self) -> DenseBlockJacobian:
        """Return a sibling Jacobian whose forward time order is reversed. This
        shares the same underlying tensors and only toggles a private flag, so
        no physical ``flip`` materializes."""
        return DenseBlockJacobian(
            diag=self.diag, sub=self.sub, _reversed=not self._reversed
        )
