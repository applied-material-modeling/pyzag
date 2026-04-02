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
Abstract block operator interfaces.
"""

from abc import ABC, abstractmethod


class BlockOperator(ABC):
    """
    Abstract interface for a logical packed block operator.

    This interface defines the solver-facing contract for a block operator.
    It does not require any particular internal storage layout.

    Logical conventions
    -------------------
    The operator consists of `nblk` logical blocks. The solver treats vectors
    passed to this operator as block-major: axis 0 indexes logical blocks in
    solver order.

    For the current bidiagonal solvers, vector inputs are expected to satisfy

        x.shape[0] == self.nblk

    and are typically shaped

        (nblk, batch_size, block_vec_size)

    where:
        - axis 0 is the logical block / time index
        - axis 1 is the batch index
        - axis 2 is the local vector entries for one block

    Backend freedom
    ---------------
    A backend may store its blocks in any representation it wants
    (dense, structured dense, factored, sparse-like packed, etc.) as long as:
        - `nblk` reports the correct logical number of blocks
        - `batch_size` and `block_shape` describe the logical block action
        - all methods below preserve the same logical block ordering
    """

    @property
    @abstractmethod
    def device(self):
        """Execution device for torch-backed implementations."""

    @property
    @abstractmethod
    def dtype(self):
        """Data type for torch-backed implementations."""

    @property
    @abstractmethod
    def nblk(self):
        """Number of logical blocks in this operator."""

    @property
    @abstractmethod
    def batch_size(self):
        """Logical batch size expected in block-major vector inputs."""

    @property
    @abstractmethod
    def block_shape(self):
        """Logical shape of one operator block.
        This is the mathematical shape of a single block.
        """

    @abstractmethod
    def matvec(self, x):
        """Apply the operator to a block-major vector `x`.

        Required solver-side convention:
            x.shape[0] == self.nblk

        The backend must interpret `x[i]` as the vector for logical block `i`
        and return the result in the same logical block order.
        """

    @abstractmethod
    def t_matvec(self, x):
        """Apply the transpose of the operator to a block-major vector `x`.

        Required solver-side convention:
            x.shape[0] == self.nblk

        The backend must interpret `x[i]` as the vector for logical block `i`
        and return the result in the same logical block order.
        """

    @abstractmethod
    def clone(self):
        """Return a safe copy of the operator."""

    def __len__(self):
        return self.nblk


class SolvableBlockOperator(BlockOperator):
    """Block operator supporting direct block solves.

    The `solve` and `solve_lower_bidiagonal` methods are solver-facing
    contracts. They operate on block-major right-hand sides whose leading
    dimension is the logical block index.
    """

    @abstractmethod
    def solve(self, rhs):
        """Solve the block system `A x = rhs`.

        Required solver-side convention:
            rhs.shape[0] == self.nblk

        For the current bidiagonal solvers, `rhs` is typically shaped
            (nblk, batch_size, block_vec_size)

        Returns a solution with the same leading block convention.
        """


class BlockViewOps(ABC):
    """Logical window/update contract required by the PCR and Thomas solver.

    PCR does not assume dense storage. It only assumes that a backend can
    expose and update contiguous logical block windows in solver order.

    These methods may return views or copies.
    """

    @abstractmethod
    def block(self, i):
        """Return logical block i as an operator with nblk == 1."""

    @abstractmethod
    def pad_front(self, n=1):
        """Return an operator with `n` leading dummy logical blocks."""

    @abstractmethod
    def trim_front(self, n=1):
        """Return an operator with the first `n` logical blocks removed."""

    @abstractmethod
    def window(self, start, end):
        """Return logical block window `[start:end)`.

        The returned operator must preserve the original logical block order.
        If `end - start = m`, the returned operator must represent exactly `m`
        logical blocks.
        """

    @abstractmethod
    def update_window(self, start, end, other):
        """Overwrite logical block window `[start:end)` with `other`.

        Required compatibility:
            other.nblk == end - start

        The update must affect the same logical block range that would be
        returned by `window(start, end)`.
        """


class PCRFactorizedDiagonalOps(SolvableBlockOperator, BlockViewOps):
    """Diagonal-block contract for fast PCR solves.
    This interface is for the diagonal operator used by PCR-based bidiagonal
    solves.
    """

    @abstractmethod
    def reduce_block(self, B, rhs):
        """Perform one backend-native PCR reduction on a contiguous block window.

        Let `m = self.nblk` for the current PCR window. The solver expects:
            rhs.shape[0] == m

        and expects the returned reduced system to satisfy:
            B_red.nblk == m - 1
            rhs_red.shape[0] == m - 1
        """


class BlockOperatorBuilder(ABC):
    """Convert a model/Jacobian representation into block operators.
    Use in nonlinear.py
    """

    @abstractmethod
    def make_forward_blocks(self, J):
        """Return `(A_ops, B_ops)` for the forward lower block-bidiagonal system.

        Expected logical meaning:
            - `A_ops` contains the diagonal blocks
            - `B_ops` contains the lower off-diagonal blocks
            - `B_ops.nblk == A_ops.nblk - 1`
        """

    @abstractmethod
    def make_adjoint_blocks(self, J):
        """Return `(A_ops, B_ops)` for the adjoint system.

        The returned operators must follow the same logical block-ordering
        convention expected by the adjoint solver.
        """
