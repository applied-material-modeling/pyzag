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

"""
Functions and objects to help with blocked/chunked time integration.

These include:
    1. Sparse matrix classes for banded systems
    2. General sparse matrix classes
    3. Specialized solver routines working with banded systems
"""

import warnings
from math import log2, floor

import torch
import numpy as np

from pyzag.operators.base import (
    BlockOperator,
    SolvableBlockOperator,
    BlockViewOps,
    PCRFactorizedDiagonalOps,
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
        rtol=1e-6,
        atol=1e-10,
        miter=200,
        throw_on_fail=False,
        record_failed=False,
        ignore_batches=None,
    ):
        self.rtol = rtol
        self.atol = atol
        self.miter = miter
        self.throw_on_fail = throw_on_fail

        self.record_failed = record_failed
        self.failed = None

        self.ignore_batches = ignore_batches

    def setup(self, x):
        """Do any initialization required before solving"""

    def solve(self, fn, x0):
        """Actually solve the system

        Args:
            fn (function): function that returns the residual and Jacobian (as appropriate chunked operators)
            x0 (torch.tensor): initial guess, again properly chunked

        Returns:
            torch.tensor:   solution
        """
        self.setup(x0)
        x = x0
        R, J = fn(x)

        nR = torch.norm(R, dim=-1)
        nR0 = nR.clone()
        i = 0

        while i < self.miter:
            # There is no reason to thunk on nans
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
            # We took one more newton step since we calculated this
            self._store_failed(
                torch.logical_or(self.not_converged(nR, nR0), torch.isnan(nR))
            )

        return x

    def not_converged(self, nR, nR0):
        """The logical to determine if we've converged in a particular time/batch

        Args:
            nR (torch.tensor): current residual
            nR0 (torch.tensor): original residual
        """
        return torch.logical_and(nR > self.atol, nR / nR0 > self.rtol)

    def _store_failed(self, not_converged):
        """Store which batches did not converge

        Args:
            not_converged (torch.tensor of bool): which entries did not converge
        """
        failed_this_time = torch.any(not_converged, dim=0)
        if self.failed is None:
            self.failed = failed_this_time
        else:
            self.failed = torch.logical_or(failed_this_time, self.failed)

    def step(self, x, J, fn, R0, take_step):
        """Take a simple Newton step

        Args:
            x (torch.tensor): current solution
            dx (torch.tensor): newton increment
            fn (function): function
            R0 (torch.tensor): current residual
            take_step (torch.tensor): which entries to take a step with
        """
        final_steps = torch.any(take_step, dim=0)

        dx = J.inverse().matvec(R0)

        x[:, final_steps] = x[:, final_steps] - dx[:, final_steps]
        R, J = fn(x)
        nR = torch.norm(R, dim=-1)

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

    def __init__(self, *args, alpha=0.5, linesearch_iter=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.linesearch_iter = linesearch_iter

    def step(self, x, J, fn, R0, take_step):
        """Take a Newton step with backtracking line search

        Args:
            x (torch.tensor): current solution
            dx (torch.tensor): newton increment
            fn (function): function
            R0 (torch.tensor): current residual
            take_step (torch.tensor): which entries to take a step with
        """
        # Need to map into the full x
        final_steps = torch.any(take_step, dim=0)

        nR0 = torch.norm(R0.transpose(0, 1).flatten(1), dim=-1)[final_steps]
        dx = J.inverse().matvec(R0)[:, final_steps]
        x0 = x[:, final_steps].clone()

        f = torch.ones_like(nR0)

        for _ in range(self.linesearch_iter):
            x[:, final_steps] = x0 - f.unsqueeze(-1).unsqueeze(0) * dx

            R, J = fn(x)
            nR = torch.norm(R, dim=-1)
            nRR = torch.norm(R.transpose(0, 1).flatten(1), dim=-1)[final_steps]

            decreasing = nRR < nR0

            if torch.all(decreasing):
                break

            f = torch.where(decreasing, f, f * self.alpha)

        return x, R, J, nR


class BidiagonalOperator(torch.nn.Module):
    """
    This is now handled abstractions.
    """

    def __init__(self, A, B, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(A, BlockOperator):
            raise TypeError("A must be a packed BlockOperator.")

        if not isinstance(B, BlockOperator):
            raise TypeError("B must be a packed BlockOperator.")

        if B.nblk != A.nblk - 1:
            raise ValueError("B must have nblk = A.nblk - 1.")

        self.A = A
        self.B = B
        self.nblk = A.nblk

    @property
    def dtype(self):
        """dtype"""
        return self.A.dtype

    @property
    def device(self):
        """device"""
        return self.A.device

    @property
    def batch_size(self):
        """batch size"""
        return self.A.batch_size

    @property
    def block_shape(self):
        """block shape"""
        return self.A.block_shape

    @property
    def sblk(self):
        """size of each block"""
        return self.block_shape[-1]

    @property
    def sbat(self):
        """batch size"""
        return self.batch_size

    @property
    def n(self):
        """size of the full matrix"""
        return self.nblk * self.sblk

    @property
    def shape(self):
        """shape of the full matrix"""
        return (self.sbat, self.n, self.n)


class LUFactorization(BidiagonalOperator):
    """A factorization that uses the LU decomposition of A"""

    def forward(self, v):
        """Apply the factorization to a vector v"""
        return self.matvec(v)


def thomas_solve(A, B, v):
    """Generic solver-owned Thomas solve over block views."""
    if v.ndim != 3:
        raise ValueError("v must have shape (nblk, sbat, sblk).")
    if A.nblk != v.shape[0]:
        raise ValueError("A.nblk must match v.shape[0].")
    if B.nblk != A.nblk - 1:
        raise ValueError("B.nblk must equal A.nblk - 1.")

    v_work = v.clone()

    v_work[0:1] = A.block(0).solve(v_work[0:1])

    for i in range(1, A.nblk):
        Ai = A.block(i)
        Bi = B.block(i - 1)

        ri = v_work[i : i + 1] - Bi.matvec(v_work[i - 1 : i].clone())
        v_work[i : i + 1] = Ai.solve(ri)

    return v_work


class BidiagonalThomasFactorization(LUFactorization):
    """
    Manages the data needed to solve our bidiagonal system via Thomas
    factorization
    """

    def __init__(self, A, B, *args, **kwargs):
        super().__init__(A, B, *args, **kwargs)

        if not isinstance(self.A, SolvableBlockOperator):
            raise TypeError("A must implement SolvableBlockOperator.")
        if not isinstance(self.A, BlockViewOps):
            raise TypeError("A must implement BlockViewOps.")
        if not isinstance(self.B, BlockViewOps):
            raise TypeError("B must implement BlockViewOps.")

    def matvec(self, v):
        return thomas_solve(self.A, self.B, v)


########## NEW PCR #################
class BidiagonalPCRFactorization(LUFactorization):
    """
    Fast PCR factorization for PCR-capable packed backends only.
    """

    def __init__(self, A, B, *args, **kwargs):
        super().__init__(A, B, *args, **kwargs)

        if not isinstance(self.A, PCRFactorizedDiagonalOps):
            raise TypeError("A must implement PCRFactorizedDiagonalOps.")
        if not isinstance(self.B, BlockViewOps):
            raise TypeError("B must implement BlockViewOps.")

    def matvec(self, v):
        """
        Apply the factorization to a vector v.
        This mirrors the original dense PCR algorithm:
        - pad B once
        - reduce each power-of-two window independently
        - solve with the original A ordering
        """
        B = self.B.pad_front(1)
        v_work = v.clone()

        for s, e in zip(*self._pow2(self.nblk)):
            A_blk = self.A.window(s, e)
            B_blk = B.window(s, e)

            B_red, v_red = A_blk.reduce_block(B_blk, v_work[s:e])

            expected = e - s - 1
            if B_red.nblk != expected:
                raise RuntimeError(
                    f"PCR backend returned wrong B_red size: got {B_red.nblk}, expected {expected}"
                )
            if v_red.shape[0] != expected:
                raise RuntimeError(
                    f"PCR backend returned wrong rhs size: got {v_red.shape[0]}, expected {expected}"
                )

            B.update_window(s + 1, e, B_red)
            v_work[s + 1 : e] = v_red

        return self.A.solve(v_work)

    @staticmethod
    def _pow2(n):
        """
        Return lists of start and end indices for power-of-two windows covering n blocks.
        """

        def sz(nv):
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

    def __init__(self, *args, min_size=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_size = min_size + 1

    def matvec(self, v):
        """
        Original hybrid semantics:
        - pad B once
        - perform PCR only on selected pow2 windows
        - trim B back
        - solve the first reduced prefix directly
        - finish the remaining blocks with Thomas in original ordering
        """
        B = self.B.pad_front(1)
        v_work = v.clone()

        start, end, last = self._pcr_blocks()
        self._apply_pcr_windows(B, v_work, start, end)

        B = B.trim_front(1)
        self._solve_hybrid_tail(B, v_work, last)

        return v_work

    def _apply_pcr_windows(self, B, v_work, start, end):
        for s, e in zip(start, end):
            A_blk = self.A.window(s, e)
            B_blk = B.window(s, e)

            B_red, v_red = A_blk.reduce_block(B_blk, v_work[s:e])
            self._check_pcr_reduce_result(B_red, v_red, s, e)

            B.update_window(s + 1, e, B_red)
            v_work[s + 1 : e] = v_red

    @staticmethod
    def _check_pcr_reduce_result(B_red, v_red, s, e):
        expected = e - s - 1
        if B_red.nblk != expected:
            raise RuntimeError(
                f"PCR backend returned wrong B_red size: got {B_red.nblk}, expected {expected}"
            )
        if v_red.shape[0] != expected:
            raise RuntimeError(
                f"PCR backend returned wrong rhs size: got {v_red.shape[0]}, expected {expected}"
            )

    def _solve_hybrid_tail(self, B, v_work, last):
        if last > 0:
            v_work[:last] = self.A.window(0, last).solve(v_work[:last])

        for i in range(last, self.nblk):
            Ai = self.A.window(i, i + 1)
            Bi = B.window(i - 1, i)
            ri = v_work[i : i + 1] - Bi.matvec(v_work[i - 1 : i])
            v_work[i : i + 1] = Ai.solve(ri)

    def _pcr_blocks(self):
        """
        Return the start and end indices for PCR blocks, as well as the size of the first reduced prefix.
        """
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


######## END OF NEW PCR ############


# Cheater wrapper
def BidiagonalHybridFactorization(min_size=1):
    """Apply the hybrid factorization with a given min_size"""
    return lambda A, B, min_size=min_size: BidiagonalHybridFactorizationImpl(
        A, B, min_size=min_size
    )


class BidiagonalForwardOperator(BidiagonalOperator):
    """
    Now handles abstractions
    """

    def __init__(self, *args, inverse_operator=BidiagonalThomasFactorization, **kwargs):
        super().__init__(*args, **kwargs)
        self.inverse_operator = inverse_operator

    def forward(self, v):
        """Apply the operator to a vector v"""
        return self.matvec(v)

    def matvec(self, v):
        """Apply the operator to a vector v"""
        out = self.A.matvec(v)
        if self.nblk > 1:
            tail = out[1:] + self.B.matvec(v[:-1])
            out = torch.cat([out[:1], tail], dim=0)
        return out

    def vecmat(self, v):
        """Apply the transpose of the operator to a vector v"""
        out = self.A.t_matvec(v)
        if self.nblk > 1:
            head = out[:-1] + self.B.t_matvec(v[1:])
            out = torch.cat([head, out[-1:]], dim=0)
        return out

    def inverse(self):
        """
        Return an inverse operator
        """
        return self.inverse_operator(self.A, self.B)


class SquareBatchedBlockDiagonalMatrix:
    """
    now handles abstractions
    """

    def __init__(self, data, diags):
        # We will want this in order later
        iargs = np.argsort(diags)

        self.data = [data[i] for i in iargs]
        self.diags = [diags[i] for i in iargs]

        self.nblk = self.data[0].shape[0] + abs(self.diags[0])
        self.sbat = self.data[0].shape[1]
        self.sblk = self.data[0].shape[-1]

    @property
    def dtype(self):
        """
        dtype, as reported by the first entry in self.data
        """
        return self.data[0].dtype

    @property
    def device(self):
        """
        device, as reported by the first entry in self.device
        """
        return self.data[0].device

    @property
    def n(self):
        """
        Size of the unbatched square matrix
        """
        return self.nblk * self.sblk

    @property
    def shape(self):
        """
        Logical shape of the dense array
        """
        return (self.sbat, self.n, self.n)

    @property
    def nnz(self):
        """
        Number of logical non-zeros (not counting the batch dimension)
        """
        return sum(
            self.data[i].shape[0] * self.sblk * self.sblk
            for i in range(len(self.diags))
        )

    def to_dense(self):
        """
        Convert the representation to a dense tensor
        """
        A = torch.zeros(*self.shape, dtype=self.dtype, device=self.device)

        # There may be a more clever way than for loops, but for now
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

    def to_batched_coo(self):
        """
        Convert to a torch sparse batched COO tensor

        This is done in a weird way.  torch recognizes "batch" dimensions at
        the start of the tensor and "dense" dimensions at the end (with "sparse"
        dimensions in between).  batch dimensions can/do have difference indices,
        dense dimensions all share the same indices.  We have the latter situation
        so this is setup as a tensor with no "batch" dimensions, 2 "sparse" dimensions,
        and 1 "dense" dimension.  So it will be the transpose of the shape of the
        to_dense function.
        """
        inds = torch.zeros(2, self.nnz)
        data = torch.zeros(self.nnz, self.sbat, dtype=self.dtype, device=self.device)

        # Order doesn't matter, nice!
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

    def to_unrolled_csr(self):
        """
        Return a list of CSR tensors with length equal to the batch size
        """
        coo = self.to_batched_coo()
        return [
            torch.sparse_coo_tensor(coo.indices(), coo.values()[:, i]).to_sparse_csr()
            for i in range(self.sbat)
        ]
