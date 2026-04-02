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

from abc import ABC, abstractmethod


class BlockOperator(ABC):
    """Abstract packed block-operator interface.
    dimensions should be (nblk , ...) where nblk is the number of blocks
    and the remaining dimensions are the block shape.
    """

    @property
    @abstractmethod
    def device(self):
        """Execution device for torch-backed implementations."""
        pass

    @property
    @abstractmethod
    def dtype(self):
        """Data type for torch-backed implementations."""
        pass

    @property
    @abstractmethod
    def nblk(self):
        """Number of packed blocks."""
        pass

    @property
    @abstractmethod
    def batch_size(self):
        """Batch size of each packed block."""
        pass

    @property
    @abstractmethod
    def block_shape(self):
        """Per-block operator shape. (ignoring the first dimensions)"""
        pass

    @abstractmethod
    def matvec(self, x):
        """A x"""
        pass

    @abstractmethod
    def t_matvec(self, x):
        """A^T x"""
        pass

    @abstractmethod
    def matmat(self, X):
        """A B X"""
        pass

    @abstractmethod
    def t_matmat(self, X):
        """A^T B X"""
        pass

    @abstractmethod
    def compose(self, other):
        """Return the operator self @ other."""
        pass

    @abstractmethod
    def add(self, other):
        """Return self + other."""
        pass

    @abstractmethod
    def sub(self, other):
        """Return self - other."""
        pass

    @abstractmethod
    def neg(self):
        """Return -self."""
        pass

    @abstractmethod
    def clone(self):
        """Return a safe copy of the operator."""
        pass

    @abstractmethod
    def slice_blocks(self, start=None, end=None, step=None):
        """Return a packed view/copy of a subset of blocks."""
        pass

    @abstractmethod
    def empty_like(self, nlbk):
        """Return an empty operator with the same block shape and batch size."""
        pass

    def __len__(self):
        return self.nblk


class SolvableBlockOperator(BlockOperator):
    """Abstract block operator that supports linear solves."""

    @abstractmethod
    def solve(self, rhs):
        """Solve A x = rhs."""
        pass

    @abstractmethod
    def t_solve(self, rhs):
        """Solve A^T x = rhs."""
        pass

    @abstractmethod
    def solve_mat(self, rhs):
        """Solve A X = rhs for multiple rhs columns."""
        pass

    @abstractmethod
    def t_solve_mat(self, rhs):
        """Solve A^T X = rhs for multiple rhs columns."""
        pass

    @abstractmethod
    def inv_compose(self, other):
        """Return the operator A^{-1} @ other."""
        pass

    @abstractmethod
    def t_inv_compose(self, other):
        """Return the operator A^{-T} @ other."""
        pass

    # for a generic approach, should be overriden
    def solve_lower_bidiagonal(self, B, rhs):
        out = [self.slice_blocks(0, 1).solve(rhs[0:1])[0]]

        for i in range(1, self.nblk):
            Bi = B.slice_blocks(i - 1, i)
            Ai = self.slice_blocks(i, i + 1)
            ri = rhs[i : i + 1] - Bi.matvec(out[i - 1].unsqueeze(0))
            out.append(Ai.solve(ri)[0])

        return __import__("torch").stack(out, dim=0)

    # for a generic approach, should be overriden
    def solve_lower_bidiagonal_transpose(self, B, rhs):
        out = rhs.clone()
        n = self.nblk

        out[-1] = self.slice_blocks(n - 1, n).t_solve(rhs[-1:])[0]
        for i in range(n - 2, -1, -1):
            Ai = self.slice_blocks(i, i + 1)
            Bi = B.slice_blocks(i, i + 1)
            ri = rhs[i : i + 1] - Bi.t_matvec(out[i + 1].unsqueeze(0))
            out[i] = Ai.t_solve(ri)[0]

        return out


class PCRBlockViewOps(ABC):
    """
    Storage/layout contract for fast PCR backends.
    Implementations must guarantee that:
    - pcr_window returns writable views or copies
    - pcr_update_window is safe under internal storage layout
    """

    @abstractmethod
    def pcr_pad_front(self, n=1):
        """Return a new operator/view with n leading dummy blocks."""
        pass

    @abstractmethod
    def pcr_trim_front(self, n=1):
        """Drop n leading blocks."""
        pass

    @abstractmethod
    def pcr_window(self, start, end):
        """Return contiguous block window [start:end)."""
        pass

    @abstractmethod
    def pcr_update_window(self, start, end, other):
        """Write other into [start:end)."""
        pass


class PCRFactorizedDiagonalOps(SolvableBlockOperator, PCRBlockViewOps):
    """Factored diagonal blocks suitable for fast PCR."""

    @abstractmethod
    def pcr_reduce_block(self, B, rhs):
        """Perform one backend-native PCR reduction block."""
        pass


class BlockOperatorBuilder(ABC):
    """Convert a model/Jacobian representation into block operators.
    Use in nonlinear.py
    """

    @abstractmethod
    def make_forward_blocks(self, J):
        """Return (A_ops, B_ops) for the forward bidiagonal system."""
        pass

    @abstractmethod
    def make_adjoint_blocks(self, J):
        """Return (A_ops, B_ops) for the adjoint bidiagonal system."""
        pass
