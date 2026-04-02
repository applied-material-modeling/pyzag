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
Packed block operators with dense tensor storage and direct linear algebra implementations.
"""

from math import prod

import torch
from torch.nn.functional import pad

from .base import (
    BlockOperatorBuilder,
    BlockViewOps,
    PCRFactorizedDiagonalOps,
    SolvableBlockOperator,
)


def batch_lu_solve(lu, pivots, rhs):
    """
    Batched version of torch.linalg.lu_solve that accepts separate LU and pivot tensors.
    """
    return torch.linalg.lu_solve(lu, pivots, rhs)


def _dense_pcr_cyclic_shift(A, level):
    """
    Perform a cyclic shift on a dense tensor for PCR reduction.

    Args:
        A (torch.Tensor): Input tensor.
        level (int): PCR reduction level.

    Returns:
        torch.Tensor: Shifted tensor.
    """
    return A.as_strided(
        (A.shape[0] * 2, A.shape[1] // 2) + A.shape[2:],
        (prod(A.shape[2:]), 2 ** (level + 1) * prod(A.shape[2:])) + A.stride()[2:],
    )


class DenseBlockOperator(SolvableBlockOperator, BlockViewOps):
    """Dense tensor-backed packed block operator.

    Args:
        data (torch.Tensor): shape (nblk, sbat, sblk, sblk)
    """

    def __init__(self, data):
        if data.ndim != 4:
            raise ValueError(
                "DenseBlockOperator expects shape (nblk, sbat, sblk, sblk)."
            )
        if data.shape[-1] != data.shape[-2]:
            raise ValueError("DenseBlockOperator requires square blocks.")
        self.data = data

    @property
    def device(self):
        """Return the device of the underlying data tensor."""
        return self.data.device

    @property
    def dtype(self):
        """Return the dtype of the underlying data tensor."""
        return self.data.dtype

    @property
    def nblk(self):
        """Return the number of logical blocks."""
        return self.data.shape[0]

    @property
    def batch_size(self):
        """Return the logical batch size."""

        return self.data.shape[1]

    @property
    def block_shape(self):
        """Return the shape of one operator block."""
        return self.data.shape[-2:]

    def matvec(self, x):
        """Apply the operator to a block-major vector `x`."""
        if x.ndim != 3:
            raise ValueError("matvec expects x with shape (nblk, sbat, sblk).")
        return torch.matmul(self.data, x.unsqueeze(-1)).squeeze(-1)

    def t_matvec(self, x):
        """Apply the transpose of the operator to a block-major vector `x`."""
        if x.ndim != 3:
            raise ValueError("t_matvec expects x with shape (nblk, sbat, sblk).")
        return torch.matmul(self.data.transpose(-1, -2), x.unsqueeze(-1)).squeeze(-1)

    def solve(self, rhs):
        """Solve the linear system with this operator as the coefficient matrix."""
        if rhs.ndim != 3:
            raise ValueError("solve expects rhs with shape (nblk, sbat, sblk).")
        return torch.linalg.solve(self.data, rhs.unsqueeze(-1)).squeeze(-1)

    def clone(self):
        """Return a safe copy of the operator."""
        return DenseBlockOperator(self.data.clone())

    def block(self, i):
        """Return logical block i as an operator with nblk == 1."""
        return DenseBlockOperator(self.data[i : i + 1])

    def window(self, start, end):
        """Return logical block window [start:end)."""
        return DenseBlockOperator(self.data[start:end])

    def pad_front(self, n=1):
        """Return an operator with `n` leading dummy logical blocks."""
        if n < 0:
            raise ValueError("n must be nonnegative.")
        if n == 0:
            return self.clone()
        data = pad(self.data, (0, 0, 0, 0, 0, 0, n, 0))
        return DenseBlockOperator(data)

    def trim_front(self, n=1):
        """Return an operator with the first `n` logical blocks removed."""
        if n < 0:
            raise ValueError("n must be nonnegative.")
        return DenseBlockOperator(self.data[n:])

    def update_window(self, start, end, other):
        """Overwrite logical block window `[start:end)` with `other`."""

        if not isinstance(other, DenseBlockOperator):
            raise TypeError("other must be DenseBlockOperator.")
        self.data[start:end].copy_(other.data)
        return self


class DenseBlockLUFactorizedOperator(PCRFactorizedDiagonalOps):
    """Dense tensor-backed packed block operator with cached LU.

    Args:
        data (torch.Tensor): shape (nblk, sbat, sblk, sblk)
    """

    def __init__(self, data):
        if data.ndim != 4:
            raise ValueError(
                "DenseBlockLUFactorizedOperator expects shape (nblk, sbat, sblk, sblk)."
            )
        if data.shape[-1] != data.shape[-2]:
            raise ValueError("DenseBlockLUFactorizedOperator requires square blocks.")

        self.data = data
        self.lu, self.pivots, _ = torch.linalg.lu_factor_ex(data)

    @classmethod
    def from_factored(cls, data, lu, pivots):
        """
        Construct without recomputing LU.
        """
        obj = cls.__new__(cls)
        obj.data = data
        obj.lu = lu
        obj.pivots = pivots
        return obj

    @property
    def device(self):
        """Return the device of the underlying data tensor."""
        return self.data.device

    @property
    def dtype(self):
        """Return the dtype of the underlying data tensor."""
        return self.data.dtype

    @property
    def nblk(self):
        """Return the number of logical blocks."""
        return self.data.shape[0]

    @property
    def batch_size(self):
        """Return the logical batch size."""
        return self.data.shape[1]

    @property
    def block_shape(self):
        """Return the shape of one operator block."""
        return self.data.shape[-2:]

    def matvec(self, x):
        """Apply the operator to a block-major vector `x`."""
        if x.ndim != 3:
            raise ValueError("matvec expects x with shape (nblk, sbat, sblk).")
        return torch.matmul(self.data, x.unsqueeze(-1)).squeeze(-1)

    def t_matvec(self, x):
        """Apply the transpose of the operator to a block-major vector `x`."""
        if x.ndim != 3:
            raise ValueError("t_matvec expects x with shape (nblk, sbat, sblk).")
        return torch.matmul(self.data.transpose(-1, -2), x.unsqueeze(-1)).squeeze(-1)

    def solve(self, rhs):
        """Solve the linear system with this operator as the coefficient matrix."""
        if rhs.ndim != 3:
            raise ValueError("solve expects rhs with shape (nblk, sbat, sblk).")
        return batch_lu_solve(self.lu, self.pivots, rhs.unsqueeze(-1)).squeeze(-1)

    def clone(self):
        """Return a safe copy of the operator."""
        return DenseBlockLUFactorizedOperator.from_factored(
            self.data.clone(),
            self.lu.clone(),
            self.pivots.clone(),
        )

    def block(self, i):
        """Return logical block i as an operator with nblk == 1."""
        return DenseBlockLUFactorizedOperator.from_factored(
            self.data[i : i + 1],
            self.lu[i : i + 1],
            self.pivots[i : i + 1],
        )

    def window(self, start, end):
        """Return logical block window [start:end)."""
        return DenseBlockLUFactorizedOperator.from_factored(
            self.data[start:end],
            self.lu[start:end],
            self.pivots[start:end],
        )

    def pad_front(self, n=1):
        """Return an operator with `n` leading dummy logical blocks."""

        if n != 0:
            raise NotImplementedError(
                "Padding factored diagonal operators is not needed for PCR."
            )
        return self.clone()

    def trim_front(self, n=1):
        """Return an operator with the first `n` logical blocks removed."""
        if n != 0:
            raise NotImplementedError(
                "Trimming factored diagonal operators is not needed for PCR."
            )
        return self.clone()

    def update_window(self, start, end, other):
        """Overwrite logical block window `[start:end)` with `other`."""
        if not isinstance(other, DenseBlockLUFactorizedOperator):
            raise TypeError("other must be DenseBlockLUFactorizedOperator.")
        self.data[start:end].copy_(other.data)
        self.lu[start:end].copy_(other.lu)
        self.pivots[start:end].copy_(other.pivots)
        return self

    def reduce_block(self, B, rhs):
        """
        Dense PCR reduction kernel that preserves the original tensor working layout.

        Args:
            B (DenseBlockOperator): shape (nblk, sbat, sblk, sblk) if padded,
                or (nblk-1, sbat, sblk, sblk) if unpadded.
            rhs (torch.Tensor): shape (nblk, sbat, sblk)

        Returns:
            (DenseBlockOperator, torch.Tensor):
                reduced B with shape (nblk-1, sbat, sblk, sblk)
                reduced rhs with shape (nblk-1, sbat, sblk)
        """
        if not isinstance(B, DenseBlockOperator):
            raise TypeError("B must be DenseBlockOperator.")
        if rhs.ndim != 3:
            raise ValueError("rhs must have shape (nblk, sbat, sblk).")
        if rhs.shape[0] != self.nblk:
            raise ValueError("rhs first dimension must match A.nblk.")

        # Accept both padded and unpadded B, but normalize to padded form
        if B.nblk == self.nblk:
            bdata = B.data
        elif B.nblk == self.nblk - 1:
            bdata = pad(B.data, (0, 0, 0, 0, 0, 0, 1, 0))
        else:
            raise ValueError(
                f"B must have nblk == {self.nblk} or {self.nblk - 1}, got {B.nblk}."
            )

        lu = self.lu.unsqueeze(0)
        pivots = self.pivots.unsqueeze(0)
        b = bdata.unsqueeze(0)
        v = rhs.unsqueeze(0).unsqueeze(-1)

        niter = self.nblk.bit_length() - 1

        for i in range(niter):
            v[:, 1:] -= torch.matmul(
                b[:, 1:],
                torch.linalg.lu_solve(lu[:, :-1], pivots[:, :-1], v[:, :-1]),
            )

            b[:, 2:] = -torch.matmul(
                b[:, 2:],
                torch.linalg.lu_solve(lu[:, 1:-1], pivots[:, 1:-1], b[:, 1:-1]),
            )

            v = _dense_pcr_cyclic_shift(v, i)
            b = _dense_pcr_cyclic_shift(b, i)
            lu = _dense_pcr_cyclic_shift(lu, i)
            pivots = _dense_pcr_cyclic_shift(pivots, i)

        return (
            DenseBlockOperator(b.squeeze(1)[1:].clone()),
            v.squeeze(1)[1:].squeeze(-1).clone(),
        )


class DenseBlockOperatorBuilder(BlockOperatorBuilder):
    """
    Block operator builder for dense tensor-backed operators.
    """

    def make_forward_blocks(self, J):
        """Return `(A_ops, B_ops)` for the forward lower block-bidiagonal system."""
        A_ops = DenseBlockLUFactorizedOperator(J[1])
        B_ops = DenseBlockOperator(J[0, 1:])
        return A_ops, B_ops

    def make_adjoint_blocks(self, J):
        """Return `(A_ops, B_ops)` for the adjoint upper block-bidiagonal system."""
        A_ops = DenseBlockLUFactorizedOperator(J[1, 1:].transpose(-1, -2))
        B_ops = DenseBlockOperator(J[0, 1:-1].transpose(-1, -2))
        return A_ops, B_ops
