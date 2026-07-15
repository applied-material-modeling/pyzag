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

# pylint: disable=abstract-method

r"""
Functions and objects to help with blocked/chunked time integration.

Within a chunk the Newton system is block lower-bidiagonal (lookback 1): with
diagonal blocks :math:`A_k = \partial R_k/\partial x_k` and subdiagonal blocks
:math:`B_k = \partial R_k/\partial x_{k-1}`,

.. math::

    (J\,x)_k = A_k\,x_k + B_{k-1}\,x_{k-1},

and each Newton iteration solves :math:`J\,\Delta = R`. Two solvers are provided:
**Thomas** (sequential forward substitution, :func:`thomas_solve`) and **parallel
cyclic reduction** (PCR), which eliminates odd blocks over power-of-two windows
using :math:`A^{-1}`-products; a hybrid switches between them by window size.
"""

from __future__ import annotations

import warnings
from math import floor, log2
from typing import Callable, Sequence

import numpy as np
import torch

from pyzag.operators.base import (
    BlockOperator,
    BlockVector,
    SolvableBlockOperator,
)


class ChunkNewtonRaphson:
    """Solve a nonlinear system with Newton's method where the residual and Jacobian are presented as chunked operators

    Keyword Args:
        rtol (float): nonlinear relative tolerance
        atol (float): nonlinear absolute tolerance
        miter (int): maximum number of iterations
        throw_on_fail (bool): if True, throw an exception on a failed solve.  If False just issue a warning.
        record_failed (bool): if True, store the indices of the bad batches
        ignore_batches (list of indices): if provided, don't check these batches in evaluating the stopping criteria
    """

    def __init__(
        self,
        rtol: float = 1e-6,
        atol: float = 1e-10,
        miter: int = 200,
        throw_on_fail: bool = False,
        record_failed: bool = False,
        ignore_batches: Sequence[int] | None = None,
    ) -> None:
        self.rtol = rtol
        self.atol = atol
        self.miter = miter
        self.throw_on_fail = throw_on_fail

        self.record_failed = record_failed
        self.failed: torch.Tensor | None = None

        self.ignore_batches = ignore_batches

    def setup(self, x: BlockVector) -> None:
        """Do any initialization required before solving"""

    def solve(
        self,
        fn: Callable[[BlockVector], tuple[BlockVector, "BidiagonalForwardOperator"]],
        x0: BlockVector,
    ) -> BlockVector:
        """Solve the nonlinear system.

        Args:
            fn: callable that returns ``(R, J)`` where ``R`` is a
                :class:`BlockVector` and ``J`` is a
                :class:`BidiagonalForwardOperator`.
            x0: initial guess as a :class:`BlockVector`.

        Returns:
            BlockVector: solution
        """
        self.setup(x0)
        x = x0
        R, J = fn(x)

        nR = R.norm(dim=-1)
        nR0 = nR.clone()
        i = 0

        while i < self.miter:
            not_converged = torch.logical_and(
                self.not_converged(nR, nR0), torch.logical_not(torch.isnan(nR))
            )
            if self.ignore_batches is not None:
                not_converged[:, self.ignore_batches] = False

            if torch.all(torch.logical_not(not_converged)):
                break

            x, R, J, nR = self.step(x, J, fn, R, not_converged)

            i += 1

        if i == self.miter:
            if self.throw_on_fail:
                raise RuntimeError("Implicit solve did not succeed.")
            warnings.warn(
                "Implicit solve did not succeed.  Results may be inaccurate..."
            )

        if self.record_failed:
            self._store_failed(
                torch.logical_or(self.not_converged(nR, nR0), torch.isnan(nR))
            )

        return x

    def not_converged(self, nR: torch.Tensor, nR0: torch.Tensor) -> torch.Tensor:
        """The logic to determine if we've converged in a particular time/batch."""
        return torch.logical_and(nR > self.atol, nR / nR0 > self.rtol)

    def _store_failed(self, not_converged: torch.Tensor) -> None:
        """Store which batches did not converge."""
        failed_this_time = torch.any(not_converged, dim=0)
        if self.failed is None:
            self.failed = failed_this_time
        else:
            self.failed = torch.logical_or(failed_this_time, self.failed)

    def step(
        self,
        x: BlockVector,
        J: "BidiagonalForwardOperator",
        fn: Callable[[BlockVector], tuple[BlockVector, "BidiagonalForwardOperator"]],
        R0: BlockVector,
        take_step: torch.Tensor,
    ) -> tuple[BlockVector, BlockVector, "BidiagonalForwardOperator", torch.Tensor]:
        """Take a simple Newton step.

        Partial step application uses the abstract
        :meth:`BlockVector.where` primitive: the candidate ``x - dx`` is
        committed only for batches whose entries in ``final_steps`` are
        True; converged batches keep their current value.
        """
        final_steps = torch.any(take_step, dim=0)

        dx = J.inverse().matvec(R0)

        candidate = x - dx
        x = candidate.where(final_steps, x)
        R, J = fn(x)
        nR = R.norm(dim=-1)

        return x, R, J, nR


class ChunkNewtonRaphsonLineSearch(ChunkNewtonRaphson):
    """Newton Raphson with backtracking line search

    Keyword Args:
        rtol (float): nonlinear relative tolerance
        atol (float): nonlinear absolute tolerance
        miter (int): maximum number of iterations
        throw_on_fail (bool): if True, throw an exception on a failed solve.  If False just issue a warning.
        record_failed (bool): if True, store the indices of the bad batches
        ignore_batches (list of indices): if provided, don't check these batches in evaluating the stopping criteria
        alpha (float): line search cutback
        linesearch_iter (int): maximum number of line search iterations
    """

    def __init__(
        self,
        *args,
        alpha: float = 0.5,
        linesearch_iter: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.linesearch_iter = linesearch_iter

    def step(
        self,
        x: BlockVector,
        J: "BidiagonalForwardOperator",
        fn: Callable[[BlockVector], tuple[BlockVector, "BidiagonalForwardOperator"]],
        R0: BlockVector,
        take_step: torch.Tensor,
    ) -> tuple[BlockVector, BlockVector, "BidiagonalForwardOperator", torch.Tensor]:
        """Take a Newton step with backtracking line search.

        Uses the abstract :meth:`BlockVector.flatten`,
        :meth:`BlockVector.scale_batches`, and :meth:`BlockVector.where`
        primitives so no backend storage is touched directly. The per-batch
        convergence scalar is ``flatten().norm(-1)`` (cross-block L2 norm
        per batch element).
        """
        final_steps = torch.any(take_step, dim=0)

        nR0 = R0.flatten().norm(dim=-1)
        dx = J.inverse().matvec(R0)
        x0 = x.clone()

        f = torch.ones(x.batch_size, dtype=R0.dtype, device=R0.device)

        for _ in range(self.linesearch_iter):
            candidate = x0 - dx.scale_batches(f)
            x = candidate.where(final_steps, x)

            R, J = fn(x)
            nR = R.norm(dim=-1)
            nRR = R.flatten().norm(dim=-1)

            # Inactive batches count as decreasing to avoid spurious backtrack.
            decreasing = (nRR < nR0) | torch.logical_not(final_steps)

            if torch.all(decreasing):
                break

            f = torch.where(decreasing, f, f * self.alpha)

        return x, R, J, nR


class BidiagonalOperator(torch.nn.Module):
    """Base class for block bidiagonal operators."""

    def __init__(self, A: BlockOperator, B: BlockOperator, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if not isinstance(A, BlockOperator):
            raise TypeError("A must be a BlockOperator.")
        if not isinstance(B, BlockOperator):
            raise TypeError("B must be a BlockOperator.")
        if B.nblk != A.nblk - 1:
            raise ValueError("B must have nblk = A.nblk - 1.")

        self.A = A
        self.B = B
        self.nblk = A.nblk

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the underlying operator blocks."""
        return self.A.dtype

    @property
    def device(self) -> torch.device:
        """The device of the underlying operator blocks."""
        return self.A.device

    @property
    def batch_size(self) -> int:
        """The batch size of the underlying operator blocks."""
        return self.A.batch_size


class BidiagonalInverseOperator(BidiagonalOperator):
    """Base for bidiagonal *inverse* operators (Thomas / PCR / hybrid).

    Applying the operator solves the bidiagonal system, i.e. it acts as the
    inverse. The actual factorization now lives in the ``SolvableBlockOperator``
    backend (e.g. ``DenseBlockOperator``); this class only aliases ``forward``
    to :meth:`matvec`.
    """

    def forward(self, v: BlockVector) -> BlockVector:
        """Apply the inverse operator (solve the system) for a vector v"""
        return self.matvec(v)


def thomas_solve(A: BlockOperator, B: BlockOperator, v: BlockVector) -> BlockVector:
    """Generic Thomas solve over block views.

    All vector and operator arguments use the abstract block interfaces.
    """
    if A.nblk != v.nblk:
        raise ValueError("A.nblk must match v.nblk.")
    if B.nblk != A.nblk - 1:
        raise ValueError("B.nblk must equal A.nblk - 1.")

    v_work = v.clone()

    v_work[0:1] = A[0:1].solve(v_work[0:1])

    for i in range(1, A.nblk):
        Ai = A[i : i + 1]
        Bi = B[i - 1 : i]

        # Clone block i-1 only when building an autograd graph: matvec saves it
        # for backward, and the in-place ``v_work[i] = ...`` below would bump the
        # shared storage's version and invalidate that saved view. Under
        # torch.no_grad() (the adjoint solve) no graph is saved, so skip the copy.
        prev = v_work[i - 1 : i]
        if torch.is_grad_enabled():
            prev = prev.clone()
        ri = v_work[i : i + 1] - Bi.matvec(prev)
        v_work[i : i + 1] = Ai.solve(ri)

    return v_work


class BidiagonalThomasFactorization(BidiagonalInverseOperator):
    """Manages the data needed to solve our bidiagonal system via Thomas factorization."""

    def __init__(self, A: BlockOperator, B: BlockOperator, *args, **kwargs) -> None:
        super().__init__(A, B, *args, **kwargs)

        if not isinstance(self.A, SolvableBlockOperator):
            raise TypeError("A must implement SolvableBlockOperator.")

    def matvec(self, v: BlockVector) -> BlockVector:
        """Apply the Thomas factorization."""
        return thomas_solve(self.A, self.B, v)


class BidiagonalPCRFactorization(BidiagonalInverseOperator):
    """PCR factorization — algorithm lives here, backend provides pcr_init/reduce_level/finalize."""

    def __init__(self, A: BlockOperator, B: BlockOperator, *args, **kwargs) -> None:
        super().__init__(A, B, *args, **kwargs)

        if not isinstance(self.A, SolvableBlockOperator):
            raise TypeError("A must implement SolvableBlockOperator.")

    def matvec(self, v: BlockVector) -> BlockVector:
        """Apply the PCR factorization."""
        B = self.B.pad_front(1)
        v_work = v.clone()

        for s, e in zip(*self._pow2(self.nblk)):
            A_blk = self.A[s:e]
            B_blk = B[s:e]

            niter = (e - s).bit_length() - 1
            state = A_blk.pcr_init(B_blk, v_work[s:e])
            for level in range(niter):
                state = A_blk.pcr_reduce_level(state, level)
            B_red, v_red = A_blk.pcr_finalize(state)

            self._check_pcr_reduce_result(B_red, v_red, s, e)
            B[s + 1 : e] = B_red
            v_work[s + 1 : e] = v_red

        return self.A.solve(v_work)

    @staticmethod
    def _check_pcr_reduce_result(
        B_red: BlockOperator, v_red: BlockVector, s: int, e: int
    ) -> None:
        expected = e - s - 1
        if B_red.nblk != expected:
            raise RuntimeError(
                f"PCR backend returned wrong B_red size: got {B_red.nblk}, expected {expected}"
            )
        if v_red.nblk != expected:
            raise RuntimeError(
                f"PCR backend returned wrong rhs size: got {v_red.nblk}, expected {expected}"
            )

    @staticmethod
    def _pow2(n: int) -> tuple[list[int], list[int]]:
        """Return lists of start and end indices for power-of-two windows covering n blocks."""

        def sz(nv: int) -> int:
            return 2 ** floor(log2(nv))

        start = [0]
        end = [sz(n)]
        n -= end[-1]

        while n > 0:
            cz = sz(n + 1)
            start.append(end[-1] - 1)
            end.append(start[-1] + cz)
            n -= cz - 1

        return start, end


class BidiagonalHybridFactorizationImpl(BidiagonalPCRFactorization):
    """Hybrid PCR/Thomas factorization."""

    def __init__(self, *args, min_size: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.min_size = min_size + 1

    def matvec(self, v: BlockVector) -> BlockVector:
        """Apply the hybrid PCR/Thomas factorization."""
        B = self.B.pad_front(1)
        v_work = v.clone()

        start, end, last = self._pcr_blocks()
        self._apply_pcr_windows(B, v_work, start, end)

        B = B[1:]
        self._solve_hybrid_tail(B, v_work, last)

        return v_work

    def _apply_pcr_windows(
        self,
        B: BlockOperator,
        v_work: BlockVector,
        start: Sequence[int],
        end: Sequence[int],
    ) -> None:
        for s, e in zip(start, end):
            A_blk = self.A[s:e]
            B_blk = B[s:e]

            niter = (e - s).bit_length() - 1
            state = A_blk.pcr_init(B_blk, v_work[s:e])
            for level in range(niter):
                state = A_blk.pcr_reduce_level(state, level)
            B_red, v_red = A_blk.pcr_finalize(state)

            self._check_pcr_reduce_result(B_red, v_red, s, e)
            B[s + 1 : e] = B_red
            v_work[s + 1 : e] = v_red

    def _solve_hybrid_tail(
        self, B: BlockOperator, v_work: BlockVector, last: int
    ) -> None:
        if last > 0:
            v_work[:last] = self.A[0:last].solve(v_work[:last])

        for i in range(last, self.nblk):
            Ai = self.A[i : i + 1]
            Bi = B[i - 1 : i]
            ri = v_work[i : i + 1] - Bi.matvec(v_work[i - 1 : i])
            v_work[i : i + 1] = Ai.solve(ri)

    def _pcr_blocks(self) -> tuple[list[int], list[int], int]:
        """Return the start and end indices for PCR blocks, plus the size of the first reduced prefix."""
        start, end = self._pow2(self.nblk)
        blk_size = [e - s for e, s in zip(end, start)]

        if blk_size[0] < self.min_size:
            return [], [], 1

        ilast = [i for i, j in enumerate(blk_size) if j < self.min_size]
        if len(ilast) == 0:
            ilast = len(start)
        else:
            ilast = ilast[0]

        start = start[:ilast]
        end = end[:ilast]

        return start, end, end[-1]


def BidiagonalHybridFactorization(min_size: int = 1):
    """Factory wrapper for the hybrid factorization with a given min_size."""
    return lambda A, B, min_size=min_size: BidiagonalHybridFactorizationImpl(
        A, B, min_size=min_size
    )


class BidiagonalForwardOperator(BidiagonalOperator):
    """Forward bidiagonal operator that wraps an inverse-operator factory."""

    def __init__(
        self,
        *args,
        inverse_operator=BidiagonalThomasFactorization,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.inverse_operator = inverse_operator

    def forward(self, v: BlockVector) -> BlockVector:
        """Apply the forward bidiagonal operator to a vector v."""
        return self.matvec(v)

    def matvec(self, v: BlockVector) -> BlockVector:
        """Return the matrix-vector product of the bidiagonal operator with v."""
        out = self.A.matvec(v)
        if self.nblk > 1:
            tail = out[1:] + self.B.matvec(v[:-1])
            out = type(out).cat([out[:1], tail], dim=0)
        return out

    def vecmat(self, v: BlockVector) -> BlockVector:
        """Return the transpose matrix-vector product of the operator with v."""
        out = self.A.t_matvec(v)
        if self.nblk > 1:
            head = out[:-1] + self.B.t_matvec(v[1:])
            out = type(out).cat([head, out[-1:]], dim=0)
        return out

    def inverse(self) -> "BidiagonalInverseOperator":
        """Return the inverse operator built via the configured factory."""
        return self.inverse_operator(self.A, self.B)


class SquareBatchedBlockDiagonalMatrix:
    """Utility for converting block-diagonal data into dense / sparse representations."""

    def __init__(self, data, diags) -> None:
        iargs = np.argsort(diags)

        self.data = [data[i] for i in iargs]
        self.diags = [diags[i] for i in iargs]

        self.nblk = self.data[0].shape[0] + abs(self.diags[0])
        self.sbat = self.data[0].shape[1]
        self.sblk = self.data[0].shape[-1]

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the block-diagonal data."""
        return self.data[0].dtype

    @property
    def device(self) -> torch.device:
        """The device of the block-diagonal data."""
        return self.data[0].device

    @property
    def n(self) -> int:
        """Size of the unbatched square matrix."""
        return self.nblk * self.sblk

    @property
    def shape(self) -> tuple[int, int, int]:
        """Logical shape of the dense array."""
        return (self.sbat, self.n, self.n)

    @property
    def nnz(self) -> int:
        """Number of logical non-zeros (not counting the batch dimension)."""
        return sum(
            self.data[i].shape[0] * self.sblk * self.sblk
            for i in range(len(self.diags))
        )

    def to_dense(self) -> torch.Tensor:
        """Convert the representation to a dense tensor."""
        A = torch.zeros(*self.shape, dtype=self.dtype, device=self.device)

        for d, data in zip(self.diags, self.data):
            for k in range(self.nblk - abs(d)):
                if d <= 0:
                    i = k - d
                    j = k
                else:
                    i = k
                    j = k + d
                A[
                    :,
                    i * self.sblk : (i + 1) * self.sblk,
                    j * self.sblk : (j + 1) * self.sblk,
                ] = data[k]

        return A

    def to_batched_coo(self) -> torch.Tensor:
        """Convert to a torch sparse batched COO tensor."""
        inds = torch.zeros(2, self.nnz)
        data = torch.zeros(self.nnz, self.sbat, dtype=self.dtype, device=self.device)

        c = 0
        chunk = self.sblk * self.sblk
        for d, bdata in zip(self.diags, self.data):
            for i in range(bdata.shape[0]):
                data[c : c + chunk] = bdata[i].flatten(start_dim=1).t()

                offset = (i + abs(d)) * self.sblk

                if d < 0:
                    roffset = offset
                    coffset = i * self.sblk
                else:
                    roffset = i * self.sblk
                    coffset = offset

                inds[0, c : c + chunk] = (
                    torch.repeat_interleave(
                        torch.arange(
                            0, self.sblk, dtype=torch.int64, device=self.device
                        ).unsqueeze(-1),
                        self.sblk,
                        -1,
                    ).flatten()
                    + roffset
                )
                inds[1, c : c + chunk] = (
                    torch.repeat_interleave(
                        torch.arange(
                            0, self.sblk, dtype=torch.int64, device=self.device
                        ).unsqueeze(0),
                        self.sblk,
                        0,
                    ).flatten()
                    + coffset
                )

                c += chunk

        return torch.sparse_coo_tensor(
            inds,
            data,
            dtype=self.dtype,
            device=self.device,
            size=(self.n, self.n, self.sbat),
        ).coalesce()

    def to_unrolled_csr(self) -> list[torch.Tensor]:
        """Return a list of CSR tensors with length equal to the batch size."""
        coo = self.to_batched_coo()
        return [
            torch.sparse_coo_tensor(coo.indices(), coo.values()[:, i]).to_sparse_csr()
            for i in range(self.sbat)
        ]
