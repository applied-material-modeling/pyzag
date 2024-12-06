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

    def solve(self, fn, x0):
        """Actually solve the system

        Args:
            fn (function): function that returns the residual and Jacobian (as appropriate chunked operators)
            x0 (torch.tensor): initial guess, again properly chunked

        Returns:
            torch.tensor:   solution
        """
        x = x0
        R, J = fn(x)

        nR = torch.norm(R, dim=-1)
        nR0 = nR.clone()
        i = 0

        while i < self.miter:
            not_converged = self.not_converged(nR, nR0)
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
            self._store_failed(self.not_converged(nR, nR0))

        return x

    def not_converged(self, nR, nR0):
        """The logical to determine if we've converged in a particular time/batch

        Args:
            nR (torch.tensor): current residual
            nR0 (torch.tensor): original residual
        """
        return torch.logical_or(
            torch.logical_and(nR > self.atol, nR / nR0 > self.rtol), torch.isnan(nR)
        )

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

    def step(self, x, J, fn, R, take_step):
        """Take a simple Newton step

        Args:
            x (torch.tensor): current solution
            dx (torch.tensor): newton increment
            fn (function): function
            R0 (torch.tensor): current residual
            take_step (torch.tensor): which entries to take a step with
        """
        final_steps = torch.any(take_step, dim=0)

        dx = J.inverse().matvec(R)

        x[:, final_steps] = x[:, final_steps] - dx[:, final_steps]
        R, J = fn(x)
        nR = torch.norm(R, dim=-1)

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

    Args:
        A (torch.tensor): tensor of shape (nblk,sbat,sblk,sblk)
            storing the nblk main diagonal blocks
        B (torch.tensor): tensor of shape (nblk-1,sbat,sblk,sblk)
            storing the nblk-1 off diagonal blocks
    """

    def __init__(self, A, B, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.A = A
        self.B = B
        self.nblk = A.shape[0]
        self.sbat = A.shape[1]
        self.sblk = A.shape[2]

    @property
    def dtype(self):
        """
        dtype, which is just the dtype of self.diag
        """
        return self.A.dtype

    @property
    def device(self):
        """
        device, which is just the device of self.diag
        """
        return self.A.device

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


class LUFactorization(BidiagonalOperator):
    """A factorization that uses the LU decomposition of A

    Args:
        A (torch.tensor): tensor of shape (nblk,sbat,sblk,sblk) with the main diagonal
        B (torch.tensor): tensor of shape (nblk-1,sbat,sblk,sblk) with the off diagonal
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._setup_factorization()

    def _setup_factorization(self):
        """
        Form the factorization...

        Args:
            diag (torch.tensor): diagonal blocks of shape (nblk, sbat, sblk, sblk)
        """
        self.lu, self.pivots, _ = torch.linalg.lu_factor_ex(self.A)

    def forward(self, v):
        """
        Run the solve using the linear algebra type interface
        with the number of blocks and block size squeezed

        Args:
            v (torch.tensor): tensor of shape (sbat, sblk*nblk)
        """
        return self.matvec(v)


def thomas_solve(lu, pivots, B, v):
    """Simple function implementing a Thomas solve

    Solves in place of v

    Args:
        lu (torch.tensor): factorized diagonal blocks, (nblk,sbat,sblk,sblk)
        pivots (torch.tensor): pivots for factorization
        B (torch.tensor): lower diagonal blocks (nblk-1,sbat,sblk,sblk)
        v (torch.tensor): right hand side (nblk,sbat,sblk)
    """
    i = 0
    v[i] = torch.linalg.lu_solve(lu[i], pivots[i], v[i].unsqueeze(-1)).squeeze(-1)
    for i in range(1, lu.shape[0]):
        v[i] = torch.linalg.lu_solve(
            lu[i],
            pivots[i],
            v[i].unsqueeze(-1) - torch.bmm(B[i - 1], v[i - 1].unsqueeze(-1).clone()),
        ).squeeze(-1)

    return v


class BidiagonalThomasFactorization(LUFactorization):
    """
    Manages the data needed to solve our bidiagonal system via Thomas
    factorization

    Args:
        A (torch.tensor): tensor of shape (nblk,sbat,sblk,sblk) with the main diagonal
        B (torch.tensor): tensor of shape (nblk-1,sbat,sblk,sblk) with the off diagonal
    """

    def matvec(self, v):
        """
        Complete the backsolve for a given right hand side

        Args:
            v (torch.tensor): tensor of shape (nblk, sbat, sblk)
        """
        return thomas_solve(self.lu, self.pivots, self.B, v)


class BidiagonalPCRFactorization(LUFactorization):
    """
    Manages the data needed to solve our bidiagonal system via parallel cyclic reduction

    Args:
        A (torch.tensor): tensor of shape (nblk,sbat,sblk,sblk) with the main diagonal
        B (torch.tensor): tensor of shape (nblk-1,sbat,sblk,sblk) with the off diagonal
    """

    def matvec(self, v):
        """
        Complete the backsolve for a given right hand side

        Args:
            v (torch.tensor): tensor of shape (nblk, sbat, sblk)
        """
        # We could do this in place if it wasn't for the pad
        self.B = pad(self.B, (0, 0, 0, 0, 0, 0, 1, 0))

        # Now figure out how many powers of 2 we need to complete our matrix
        for s, e in zip(*self._pow2(self.nblk)):
            self.B[s + 1 : e], v[s + 1 : e] = self._solve_block(
                self.lu[s:e], self.pivots[s:e], self.B[s:e], v[s:e]
            )

        # To retain consistent sizes
        self.B = self.B[1:]

        return torch.linalg.lu_solve(self.lu, self.pivots, v.unsqueeze(-1)).squeeze(-1)

    def _solve_block(self, lu, pivots, B, v):
        """Solve a subsection of the matrix via PCR

        Args:
            lu (torch.tensor): (ncurr,sbat,sblk,sblk)
            pivots (torch.tensor): (ncurr,sbat,sblk)
            B (torch.tensor): (ncurr,sbat,sblk,sblk)
            v (torch.tensor): (ncurr,sbat,sblk)
        """
        # Number of iterations required to reduce this block
        niter = lu.shape[0].bit_length() - 1

        # Add the extra working dimension to the start of everything
        lu = lu.unsqueeze(0)
        pivots = pivots.unsqueeze(0)
        B = B.unsqueeze(0)
        v = v.unsqueeze(0).unsqueeze(-1)

        # Actually start reduction!
        for i in range(niter):
            # Reduce RHS
            v[:, 1:] -= torch.matmul(
                B[:, 1:],
                torch.linalg.lu_solve(lu[:, :-1], pivots[:, :-1], v[:, :-1]),
            )

            # Reduce off diagonal coefficients
            B[:, 2:] = -torch.matmul(
                B[:, 2:],
                torch.linalg.lu_solve(lu[:, 1:-1], pivots[:, 1:-1], B[:, 1:-1]),
            )

            # Shuffle dimensions
            v = self._cyclic_shift(v, i)
            B = self._cyclic_shift(B, i)
            lu = self._cyclic_shift(lu, i)
            pivots = self._cyclic_shift(pivots, i)

        return B.squeeze(1)[1:], v.squeeze(1)[1:].squeeze(-1)

    @staticmethod
    def _pow2(n):
        """Calculate submatrix sizes

        Args:
            n (int): number of blocks

        Returns:
            two lists, one giving start block indices and the
            second giving end block indices.

        The first (start,end) pair is the largest power of 2 that fits in
        n.  Subsequent pairs are the largest power of 2 that fit in the remainder
        *with one overlap between the next increment and the previous*.
        """

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
        """Provide a view of the input with a cyclic shift applied

        Args:
            A (torch.tensor): input tensor
            n (int): number of cyclic shifts
        """
        return A.as_strided(
            (A.shape[0] * 2, A.shape[1] // 2) + A.shape[2:],
            (prod(A.shape[2:]), 2 ** (n + 1) * prod(A.shape[2:])) + A.stride()[2:],
        )


# Cheater wrapper
def BidiagonalHybridFactorization(min_size=1):
    """Apply the hybrid factorization with a given min_size"""
    return lambda A, B, min_size=min_size: BidiagonalHybridFactorizationImpl(
        A, B, min_size=min_size
    )


class BidiagonalHybridFactorizationImpl(BidiagonalPCRFactorization):
    """A factorization approach that switches from PCR to Thomas

    Specifically, this class uses PCR until the PCR chunk size is
    smaller than user provided minimum chunk size.  Then it switches
    to Thomas.

    Args:
        A (torch.tensor): tensor of shape (nblk,sbat,sblk,sblk) with the main diagonal
        B (torch.tensor): tensor of shape (nblk-1,sbat,sblk,sblk) with the off diagonal

    Keyword Args:
        min_size (int): minimum block size, default is zero
    """

    def __init__(self, *args, min_size=0, **kwargs):
        super().__init__(*args, **kwargs)

        # I use < below...
        self.min_size = min_size + 1

    def matvec(self, v):
        """
        Complete the backsolve for a given right hand side

        Args:
            v (torch.tensor): tensor of shape (nblk, sbat, sblk, 1)
        """
        # We could do this in place if it wasn't for the pad
        self.B = pad(self.B, (0, 0, 0, 0, 0, 0, 1, 0))

        # Get the PCR blocks to actually use
        start, end, last = self._pcr_blocks()

        # Do PCR
        for s, e in zip(start, end):
            self.B[s + 1 : e], v[s + 1 : e] = self._solve_block(
                self.lu[s:e], self.pivots[s:e], self.B[s:e], v[s:e]
            )

        # To retain consistent sizes
        self.B = self.B[1:]

        # We still need to solve the first block even if last is 0

        # The actual LU solve for the solution
        v[:last] = torch.linalg.lu_solve(
            self.lu[:last], self.pivots[:last], v[:last].unsqueeze(-1)
        ).squeeze(-1)

        # Now take over for Thomas
        for i in range(last, self.nblk):
            # The .clone() here should not be necessary, but for whatever
            # reason torch autograd give the usual "in place" complaint
            # without it...
            v[i] = torch.linalg.lu_solve(
                self.lu[i],
                self.pivots[i],
                v[i].unsqueeze(-1)
                - torch.bmm(self.B[i - 1], v[i - 1].clone().unsqueeze(-1)),
            ).squeeze(-1)

        return v

    def _pcr_blocks(self):
        """Figure out the PCR blocks we are actually going to use"""
        # Figure out which blocks we're going to use
        start, end = self._pow2(self.nblk)
        # These are sorted...
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

    Args:
        A (torch.tensor): tensor of shape (nblk,sbat,sblk,sblk)
            storing the nblk diagonal blocks
        B (torch.tensor): tensor of shape (nblk-1,sbat,sblk,sblk)
            storing the nblk-1 off diagonal blocks
    """

    def __init__(self, *args, inverse_operator=BidiagonalThomasFactorization, **kwargs):
        super().__init__(*args, **kwargs)
        self.inverse_operator = inverse_operator

    def to_diag(self):
        """
        Convert to a SquareBatchedBlockDiagonalMatrix, for testing
        or legacy purposes
        """
        return SquareBatchedBlockDiagonalMatrix([self.A, self.B], [0, -1])

    def forward(self, v):
        """
        :math:`A \\cdot v` in an efficient manner

        Args:
            v (torch.tensor):   batch of vectors
        """
        # Reshaped v
        vp = v.reshape(self.sbat * self.nblk, self.sblk).unsqueeze(-1)

        # Actual calculation
        b = torch.bmm(self.A.view(-1, self.sblk, self.sblk), vp)
        b[self.sbat :] += torch.bmm(
            self.B.view(-1, self.sblk, self.sblk), vp[: -self.sbat]
        )

        return b.squeeze(-1).view(self.nblk, self.sbat, self.sblk)

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
