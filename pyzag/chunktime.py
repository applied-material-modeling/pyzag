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
from math import prod, log2, floor

import torch
from torch.nn.functional import pad
import numpy as np

from pyzag.operators.base import BlockOperator, SolvableBlockOperator
# this is to ensure backward compatibility
from pyzag.operators.dense import DenseBlockOperator, DenseBlockLUFactorizedOperator


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
    An object working with a Batched block diagonal operator of the type

    .. math::

        \\begin{bmatrix}
        A_1 & 0 & 0 & 0 & \\cdots  & 0\\\\
        B_1 & A_2 & 0 & 0 & \\cdots & 0\\\\
        0 & B_2 & A_3 & 0 & \\cdots & 0\\\\
        \\vdots & \\vdots & \\ddots & \\ddots & \\ddots  & \\vdots \\\\
        0 & 0 & 0 & B_{n-2} & A_{n-1} & 0\\\\
        0 & 0 & 0 & 0 & B_{n-1} & A_n
        \\end{bmatrix}

    that is, a blocked banded system with the main
    diagonal and the first lower diagonal filled

    We use the following sizes:
        - nblk:   number of blocks in the square matrix
        - sblk:   size of each block
        - sbat:   batch size

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
        return self.A.dtype

    @property
    def device(self):
        return self.A.device

    @property
    def batch_size(self):
        return self.A.batch_size

    @property
    def block_shape(self):
        return self.A.block_shape

    @property
    def sblk(self):
        return self.block_shape[-1]

    @property
    def sbat(self):
        return self.batch_size

    @property
    def n(self):
        return self.nblk * self.sblk

    @property
    def shape(self):
        return (self.sbat, self.n, self.n)


class LUFactorization(BidiagonalOperator):
    """A factorization that uses the LU decomposition of A
    """

    def __init__(self, A, B, *args, **kwargs):
        super().__init__(A, B, *args, **kwargs)

    def forward(self, v):
        return self.matvec(v)


def thomas_solve(A, B, v):
    return A.solve_lower_bidiagonal(B, v)

class BidiagonalThomasFactorization(LUFactorization):
    """
    Manages the data needed to solve our bidiagonal system via Thomas
    factorization
    """
    def __init__(self, A, B, *args, **kwargs):
        if isinstance(A, DenseBlockOperator):
            A = DenseBlockLUFactorizedOperator(A.data)
        super().__init__(A, B, *args, **kwargs)

    def matvec(self, v):
        return thomas_solve(self.A, self.B, v)


########## NEW PCR #################
class BidiagonalPCRFactorization(LUFactorization):
    """
    Packed PCR factorization.
    """

    def __init__(self, A, B, *args, **kwargs):
        if isinstance(A, DenseBlockOperator):
            A = DenseBlockLUFactorizedOperator(A.data)
        super().__init__(A, B, *args, **kwargs)
        self._dense_pcr = isinstance(self.A, DenseBlockLUFactorizedOperator) and isinstance(
            self.B, DenseBlockOperator
        )

    def matvec(self, v):
        if self._dense_pcr:
            return self._matvec_dense(v)
        return self._matvec_generic(v)

    def _matvec_dense(self, v):
        B = pad(self.B.data, (0, 0, 0, 0, 0, 0, 1, 0))
        v_work = v.clone()

        for s, e in zip(*self._pow2(self.nblk)):
            B[s + 1 : e], v_work[s + 1 : e] = self._solve_block(
                self.A.lu[s:e], self.A.pivots[s:e], B[s:e], v_work[s:e]
            )

        return torch.linalg.lu_solve(
            self.A.lu, self.A.pivots, v_work.unsqueeze(-1)
        ).squeeze(-1)

    def _matvec_generic(self, v):
        rhs = v
        A = self.A
        B = self.B

        if A.nblk == 1:
            return A.solve(rhs)

        hist = []

        while A.nblk > 1:
            A_red, B_red, rhs_red = self._reduce_generic(A, B, rhs)
            hist.append((A, B, rhs))
            A, B, rhs = A_red, B_red, rhs_red

        x = A.solve(rhs)

        for A_prev, B_prev, rhs_prev in reversed(hist):
            x = self._expand_generic(A_prev, B_prev, rhs_prev, x)

        return x

    @staticmethod
    def _reduce_generic(A, B, rhs):
        n = A.nblk

        A_red = A.slice_blocks(0, n, 2)
        rhs_red = rhs[0:n:2].clone()

        m = A_red.nblk - 1
        if m == 0:
            B_red = B.slice_blocks(0, 0)
            return A_red, B_red, rhs_red

        # all three must have length m
        A1 = A.slice_blocks(1, 1 + 2 * m, 2)
        B1 = B.slice_blocks(1, 1 + 2 * m, 2)
        B0 = B.slice_blocks(0, 2 * m, 2)

        rhs_odd = rhs[1 : 1 + 2 * m : 2]

        B_red = B1.neg().compose(A1.inv_compose(B0))
        rhs_red[1:] = rhs_red[1:] - B1.matvec(A1.solve(rhs_odd))

        return A_red, B_red, rhs_red

    @staticmethod
    def _expand_generic(A, B, rhs, x_even):
        n = A.nblk
        x = torch.empty_like(rhs)
        x[0:n:2] = x_even

        if n > 1:
            A_odd = A.slice_blocks(1, n, 2)
            B_odd = B.slice_blocks(0, n - 1, 2)
            x[1:n:2] = A_odd.solve(
                rhs[1:n:2] - B_odd.matvec(x[0:n:2][: A_odd.nblk])
            )

        return x

    def _solve_block(self, lu, pivots, B, v):
        niter = lu.shape[0].bit_length() - 1

        lu = lu.unsqueeze(0)
        pivots = pivots.unsqueeze(0)
        B = B.unsqueeze(0)
        v = v.unsqueeze(0).unsqueeze(-1)

        for i in range(niter):
            v[:, 1:] -= torch.matmul(
                B[:, 1:],
                torch.linalg.lu_solve(lu[:, :-1], pivots[:, :-1], v[:, :-1]),
            )

            B[:, 2:] = -torch.matmul(
                B[:, 2:],
                torch.linalg.lu_solve(lu[:, 1:-1], pivots[:, 1:-1], B[:, 1:-1]),
            )

            v = self._cyclic_shift(v, i)
            B = self._cyclic_shift(B, i)
            lu = self._cyclic_shift(lu, i)
            pivots = self._cyclic_shift(pivots, i)

        return B.squeeze(1)[1:], v.squeeze(1)[1:].squeeze(-1)

    @staticmethod
    def _pow2(n):
        def sz(n):
            return 2 ** floor(log2(n))

        start = [0]
        end = [sz(n)]
        n -= end[-1]

        while n > 0:
            cz = sz(n + 1)
            start.append(end[-1] - 1)
            end.append(start[-1] + cz)
            n -= cz - 1

        return start, end

    @staticmethod
    def _cyclic_shift(A, n):
        return A.as_strided(
            (A.shape[0] * 2, A.shape[1] // 2) + A.shape[2:],
            (prod(A.shape[2:]), 2 ** (n + 1) * prod(A.shape[2:])) + A.stride()[2:],
        )


class BidiagonalHybridFactorizationImpl(BidiagonalPCRFactorization):
    """Hybrid PCR/Thomas factorization."""

    def __init__(self, *args, min_size=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_size = min_size + 1

    def matvec(self, v):
        if self._dense_pcr:
            return self._matvec_dense_hybrid(v)
        return self._matvec_generic_hybrid(v)

    def _matvec_dense_hybrid(self, v):
        B = pad(self.B.data, (0, 0, 0, 0, 0, 0, 1, 0))
        v_work = v.clone()

        start, end, last = self._pcr_blocks()

        for s, e in zip(start, end):
            B[s + 1 : e], v_work[s + 1 : e] = self._solve_block(
                self.A.lu[s:e], self.A.pivots[s:e], B[s:e], v_work[s:e]
            )

        # critical fix: remove the leading padded zero block
        B = B[1:]

        # solve the already-reduced front part directly
        v_work[:last] = torch.linalg.lu_solve(
            self.A.lu[:last], self.A.pivots[:last], v_work[:last].unsqueeze(-1)
        ).squeeze(-1)

        # Thomas continuation on the remaining tail
        for i in range(last, self.nblk):
            v_work[i] = torch.linalg.lu_solve(
                self.A.lu[i],
                self.A.pivots[i],
                v_work[i].unsqueeze(-1)
                - torch.bmm(B[i - 1], v_work[i - 1].clone().unsqueeze(-1)),
            ).squeeze(-1)

        return v_work

    def _matvec_generic_hybrid(self, v):
        A = self.A
        B = self.B
        rhs = v
        hist = []

        while A.nblk > self.min_size:
            hist.append((A, B, rhs))
            A, B, rhs = self._reduce_generic(A, B, rhs)

        x = thomas_solve(A, B, rhs)

        for A_prev, B_prev, rhs_prev in reversed(hist):
            x = self._expand_generic(A_prev, B_prev, rhs_prev, x)

        return x

    def _pcr_blocks(self):
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
    A batched block banded matrix of the form:

    .. math::

        \\begin{bmatrix}
        A_1 & 0 & 0 & 0 & \\cdots  & 0\\\\
        B_1 & A_2 & 0 & 0 & \\cdots & 0\\\\
        0 & B_2 & A_3 & 0 & \\cdots & 0\\\\
        \\vdots & \\vdots & \\ddots & \\ddots & \\ddots  & \\vdots \\\\
        0 & 0 & 0 & B_{n-2} & A_{n-1} & 0\\\\
        0 & 0 & 0 & 0 & B_{n-1} & A_n
        \\end{bmatrix}

    that is, a blocked banded system with the main
    diagonal and first lower block diagonal filled

    We use the following sizes:
        - nblk: number of blocks in the square matrix
        - sblk: size of each block
        - sbat: batch size

    Now handles abstractions
    """

    def __init__(self, *args, inverse_operator=BidiagonalThomasFactorization, **kwargs):
        super().__init__(*args, **kwargs)
        self.inverse_operator = inverse_operator

    def forward(self, v):
        return self.matvec(v)

    def matvec(self, v):
        out = self.A.matvec(v)
        if self.nblk > 1:
            tail = out[1:] + self.B.matvec(v[:-1])
            out = torch.cat([out[:1], tail], dim=0)
        return out

    def vecmat(self, v):
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
    A batched block diagonal matrix of the type

    .. math::

        \\begin{bmatrix}
        A_1 & B_1 & 0 & 0\\\\
        C_1 & A_2 & B_2 & 0 \\\\
        0 & C_2 & A_3 & B_3\\\\
        0 & 0 & C_3 & A_4
        \\end{bmatrix}

    where the matrix has diagonal blocks of non-zeros and
    can have arbitrary numbers of filled diagonals

    Additionally, this matrix is batched.

    We use the following sizes:
        - nblk: number of blocks in the each direction
        - sblk: size of each block
        - sbat: batch size

    Args:
        data (list of tensors):     list of tensors of length ndiag.
                                    Each tensor
                                    has shape :code:`(nblk-abs(d),sbat,sblk,sblk)`
                                    where d is the diagonal number
                                    provided in the next input
        diags (list of ints):       list of ints of length ndiag.
                                    Each entry gives the diagonal
                                    for the data in the corresponding
                                    tensor.  These values d can
                                    range from -(n-1) to (n-1)
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
