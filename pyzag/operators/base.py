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
Abstract block operator and block vector interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

import torch

if TYPE_CHECKING:
    from pyzag.chunktime import BidiagonalForwardOperator


class BlockVector(ABC):
    """
    Abstract interface for a logical packed block vector.

    Logical conventions
    -------------------
    A block vector consists of `nblk` logical blocks. Solver code treats
    block vectors as block-major: axis 0 indexes logical blocks.

    For the current bidiagonal solvers, the logical shape is

        (nblk, batch_size, block_size)

    where:
        - axis 0 is the logical block / time index
        - axis 1 is the batch index
        - axis 2 is the local vector entries for one block

    Backend freedom
    ---------------
    A backend may store data in any representation it wants
    (dense, structured, sparse, factored, etc.) as long as:
        - `nblk`, `batch_size`, `block_size` describe the logical shape
        - all methods preserve the same logical block ordering
    """

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Execution device for torch-backed implementations."""

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Data type for torch-backed implementations."""

    @property
    @abstractmethod
    def nblk(self) -> int:
        """Number of logical blocks in this vector."""

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Logical batch size."""

    @property
    @abstractmethod
    def block_size(self) -> int:
        """Logical size of one block (last axis)."""

    @abstractmethod
    def clone(self) -> BlockVector:
        """Return a safe copy of the vector."""

    @abstractmethod
    def norm(self, dim: int = -1) -> torch.Tensor:
        """Compute the norm along `dim`. Returns a raw tensor (used for
        scalar convergence checks; the result has a different shape than
        a block vector and should not be wrapped)."""

    @abstractmethod
    def flatten(self) -> torch.Tensor:
        """Flatten to a raw tensor of shape ``(batch_size, ndof)``:
        batch-major, with every block and state entry for a given batch
        element concatenated along the last axis."""

    @abstractmethod
    def where(self, mask: torch.Tensor, other: BlockVector) -> BlockVector:
        """Batch-axis conditional combination, ``torch.where`` convention:
        returns ``self`` where ``mask`` is True, else ``other``.

        ``mask`` is a raw tensor of shape ``(batch_size,)``; backends
        broadcast it over the block and state axes. Used by the Newton
        step for partial updates."""

    @abstractmethod
    def scale_batches(self, factor: torch.Tensor) -> BlockVector:
        """Per-batch scalar broadcast: multiply each batch element by the
        corresponding entry of ``factor`` (raw tensor of shape
        ``(batch_size,)``, broadcast over block and state axes). Used by
        line-search backtracking."""

    @abstractmethod
    def flip(self, dim: int) -> BlockVector:
        """Return a new vector with axis `dim` reversed."""

    @abstractmethod
    def __neg__(self) -> BlockVector:
        """Return the negated vector."""

    def neg(self) -> BlockVector:
        """Return the negated vector (alias for ``-self``)."""
        return self.__neg__()  # pylint: disable=unnecessary-dunder-call

    @abstractmethod
    def __add__(self, other: BlockVector) -> BlockVector:
        """Element-wise addition with another block vector."""

    @abstractmethod
    def __sub__(self, other: BlockVector) -> BlockVector:
        """Element-wise subtraction with another block vector."""

    @abstractmethod
    def __mul__(self, other: torch.Tensor | float | int) -> BlockVector:
        """Scalar/broadcast multiplication."""

    def __rmul__(self, other: torch.Tensor | float | int) -> BlockVector:
        return self.__mul__(other)

    @abstractmethod
    def __getitem__(self, idx: int | slice) -> BlockVector:
        """Slice along the block dimension. Returns a `BlockVector`.

        Use slice syntax (`v[i:j]`, `v[-1:]`) to preserve the block
        dimension. For scalar indexing on other axes (batch/state), use
        the concrete backend's underlying storage directly.
        """

    @abstractmethod
    def __setitem__(self, idx: int | slice, value: BlockVector) -> None:
        """Assign into a block-dim slice."""

    def __len__(self) -> int:
        return self.nblk

    @classmethod
    @abstractmethod
    def cat(cls, vectors: Sequence[BlockVector], dim: int = 0) -> BlockVector:
        """Concatenate a sequence of compatible block vectors along `dim`."""

    @classmethod
    @abstractmethod
    def zeros_like(cls, other: BlockVector) -> BlockVector:
        """Construct a zero-filled block vector with the same shape as `other`."""

    # NOTE: shape-only constructors (``zeros(nblk, batch, block, ...)`` and
    # ``empty(...)``) are intentionally NOT part of this interface: the solver
    # never constructs a block vector from scalar shape metadata (it always has
    # a reference vector and uses ``zeros_like``), and a backend whose block
    # layout is richer than a single ``block_size`` (e.g. a multi-group
    # layout) cannot honour them. Backends may still offer them as extras.


class BlockOperator(ABC):
    """
    Abstract interface for a logical packed block operator.

    Logical conventions
    -------------------
    The operator consists of `nblk` logical blocks. The solver treats
    block vectors passed to this operator as block-major.

    Backend freedom
    ---------------
    A backend may store its blocks in any representation it wants as long as:
        - `nblk` reports the correct logical number of blocks
        - `batch_size` describes the logical block action
        - all methods preserve the same logical block ordering
    """

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Execution device for torch-backed implementations."""

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Data type for torch-backed implementations."""

    @property
    @abstractmethod
    def nblk(self) -> int:
        """Number of logical blocks in this operator."""

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Logical batch size expected in block-major vector inputs."""

    @abstractmethod
    def matvec(self, x: BlockVector) -> BlockVector:
        """Apply the operator to a block vector `x`.

        Required:
            x.nblk == self.nblk
        """

    @abstractmethod
    def t_matvec(self, x: BlockVector) -> BlockVector:
        """Apply the transpose of the operator to a block vector `x`.

        Required:
            x.nblk == self.nblk
        """

    @abstractmethod
    def clone(self) -> BlockOperator:
        """Return a safe copy of the operator."""

    @abstractmethod
    def __getitem__(self, idx: int | slice) -> BlockOperator:
        """Return a logical block window. `A[i:j]` returns blocks `[i, j)`;
        `A[i:i+1]` returns a single-block operator.
        """

    @abstractmethod
    def __setitem__(self, idx: int | slice, other: BlockOperator) -> None:
        """Overwrite a logical block window with `other`. Requires
        `other.nblk == end - start` for slice assignment."""

    @abstractmethod
    def pad_front(self, n: int = 1) -> BlockOperator:
        """Return an operator with `n` leading dummy logical blocks
        (creates new data, not a view)."""

    def __len__(self) -> int:
        return self.nblk


class PCRState(ABC):
    """Opaque PCR working state managed by the backend across levels.

    Created by :meth:`SolvableBlockOperator.pcr_init`, updated by
    :meth:`SolvableBlockOperator.pcr_reduce_level`, and consumed by
    :meth:`SolvableBlockOperator.pcr_finalize`. Callers in ``chunktime.py``
    treat this as an opaque handle and never inspect its contents.
    """


class SolvableBlockOperator(BlockOperator):
    """Block operator supporting direct block solves and PCR-based reduction."""

    @abstractmethod
    def solve(self, rhs: BlockVector) -> BlockVector:
        """Solve the block system `A x = rhs`.

        Required:
            rhs.nblk == self.nblk
        """

    @abstractmethod
    def pcr_init(self, B: BlockOperator, v: BlockVector) -> PCRState:
        """Initialise the backend-native PCR working state for a power-of-two
        window.

        ``self`` is the diagonal operator (A), ``B`` is the subdiagonal
        (already padded with one dummy leading block so ``B.nblk == self.nblk``),
        and ``v`` is the RHS slice for this window.

        The backend allocates internal working tensors (e.g. adds the extra
        leading dimension used by the Dense cyclic-shift trick) and returns an
        opaque :class:`PCRState`.
        """

    @abstractmethod
    def pcr_reduce_level(self, state: PCRState, level: int) -> PCRState:
        """Apply one PCR level to the working state.

        Updates the RHS vector and subdiagonal via A-inverse products, then
        applies the backend-native cyclic interleaving (``as_strided`` for
        Dense). Returns the updated :class:`PCRState` ready for the next level
        or for :meth:`pcr_finalize`.

        ``level`` (0-based) lets the backend compute the correct stride pattern
        without the caller knowing about the internal working shape.
        """

    @abstractmethod
    def pcr_finalize(self, state: PCRState) -> tuple[BlockOperator, BlockVector]:
        """Extract the reduced ``(B_red, v_red)`` from the final PCR state.

        Returns a pair with ``nblk == window_size - 1``, suitable for writing
        back into the full B and v_work arrays in
        :class:`pyzag.chunktime.BidiagonalPCRFactorization`.
        """


class BlockJacobian(ABC):
    """
    Abstract per-chunk Jacobian for a recursive nonlinear system.

    Logical model
    -------------
    For lookback = 1 (the only case the solver currently supports), the
    chunk Jacobian represents the linearization of
    ``R[k] = f(x[k-1], x[k])`` over ``k = 1..nblk_steps``. The two
    structural pieces are:

        - the diagonal:    ``dR[k]/dx[k]``    for ``k = 1..nblk_steps``
        - the subdiagonal: ``dR[k]/dx[k-1]``  for ``k = 1..nblk_steps``

    The boundary subdiagonal blocks couple a chunk to its neighbours. In
    forward time order, ``sub[0]`` (``k = 1``, ``dR[1]/dx[0]``) couples the
    chunk's first residual to the lookback / previous chunk's last state; the
    remaining subdiagonal blocks are internal to the chunk's bidiagonal system.
    The adjoint pass walks time in reverse (see :meth:`as_adjoint_walk`) and
    couples to the previously-processed (adjoint-order) chunk through index 0 of
    the *walk-order* subdiagonal -- which is the original last forward block, not
    ``sub[0]``.

    Time order is forward (low index = early time). The adjoint walk
    methods internalize all reversal -- callers use :meth:`couple_prev_chunk`
    and must NOT reach into storage to flip, reorder, or pick a boundary block
    themselves.

    Backend freedom
    ---------------
    Backends may store the diagonal/subdiagonal in any layout (dense per
    block, structured/arrowhead, sparse, factored). The contract is purely
    behavioral.
    """

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Execution device for torch-backed implementations."""

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Data type for torch-backed implementations."""

    @property
    @abstractmethod
    def nblk_steps(self) -> int:
        """Number of residual rows in the chunk."""

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Logical batch size."""

    @property
    @abstractmethod
    def block_size(self) -> int:
        """Per-step state size (the user-facing ``n``)."""

    @abstractmethod
    def forward_system(self, inverse_operator) -> "BidiagonalForwardOperator":
        """Build the chunk's forward bidiagonal system, ready for Newton.

        Returns a :class:`pyzag.chunktime.BidiagonalForwardOperator` whose
        diagonal ``A`` has ``nblk == nblk_steps`` and whose subdiagonal
        ``B`` has ``nblk == nblk_steps - 1``.

        Args:
            inverse_operator: factory used to build the inverse
                (e.g. :class:`pyzag.chunktime.BidiagonalThomasFactorization`).
        """

    @abstractmethod
    def adjoint_system(self, inverse_operator):
        """Build the chunk's adjoint bidiagonal **solve** operator, in
        adjoint-walk order.

        Transposes are baked in. The first-row / first+last-col slicing
        (``J[1, 1:].T`` and ``J[0, 1:-1].T`` after ``flip(1)``) is also
        baked in. Unlike :meth:`forward_system`,
        which returns a :class:`BidiagonalForwardOperator` (so Newton can
        call ``.inverse()`` between iterations), this returns the
        **inverse / solve** operator directly: applying ``.matvec(rhs)``
        on the returned object yields the adjoint solution. This matches
        how ``block_update_adjoint`` consumes the result with a single
        linear solve per chunk.

        This method should be called on a ``BlockJacobian`` returned by
        :meth:`as_adjoint_walk`.
        """

    @abstractmethod
    def solve_terminal_adjoint(self, g_terminal: torch.Tensor) -> BlockVector:
        """Compute ``-A_terminal^{-T} @ g_terminal`` for the very last
        forward-time step of the trajectory.

        Returns a single-block :class:`BlockVector` (``nblk == 1`` for
        ``lookback == 1``; for higher lookback this would be a
        ``lookback``-block vector). The returned vector must not alias
        ``g_terminal``.
        """

    @abstractmethod
    def couple_prev_chunk(self, a_first: BlockVector) -> BlockVector:
        """Compute the inter-chunk adjoint coupling
        ``B_boundary^T @ a_first``.

        ``a_first`` is a single-block :class:`BlockVector` holding the
        previous chunk's adjoint tail (in adjoint-walk order). Returns a
        single-block :class:`BlockVector` to be subtracted into the
        current chunk's RHS first row (``nblk == 1`` for ``lookback ==
        1``; ``lookback``-block for higher lookback).
        """

    @abstractmethod
    def as_adjoint_walk(self) -> BlockJacobian:
        """Return a :class:`BlockJacobian` whose forward time order is
        the reverse of this one. Backends are free to implement lazily
        (e.g. via a flag) so that no copy happens unless storage requires
        it. Replaces direct ``flip`` calls in solver code.
        """
