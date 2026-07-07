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

"""Generic ODE integration as :class:`NonlinearFunctionOperator` factories.

The Euler integration math is fully abstract: it operates on
:class:`BlockVector` objects and produces :class:`BlockOperator` blocks via a
user-supplied :class:`ODEWrapper`. This decouples the integration scheme
from any particular tensor backend (dense, sparse, structured, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import torch

from pyzag.nonlinear import (
    ChunkOp,
    NonlinearFunctionOperator,
    NonlinearFunctionOperatorFactory,
)
from pyzag.operators.base import BlockJacobian, BlockVector
from pyzag.operators.dense import DenseBlockJacobian, DenseBlockVector


class ODEWrapper(ABC):
    """Bridge from raw user-ODE outputs to the abstract block types.

    A user's ODE module returns raw tensors ``(x_dot, J_dot) = ode(t, x)``.
    The wrapper converts these into the :class:`BlockVector` /
    :class:`BlockOperator` family that the Euler operator and downstream
    solver consume.
    """

    @abstractmethod
    def wrap_vector(self, raw: torch.Tensor) -> BlockVector:
        """Wrap a raw tensor into a :class:`BlockVector`."""

    @abstractmethod
    def unwrap_vector(self, bv: BlockVector) -> torch.Tensor:
        """Extract the raw tensor representation from a :class:`BlockVector`.

        Required because the user's ODE module operates on raw tensors and
        the chunk's combined state must be assembled before the call.
        """

    @abstractmethod
    def wrap_jacobian(self, diag: torch.Tensor, sub: torch.Tensor) -> BlockJacobian:
        """Wrap raw per-step diagonal and subdiagonal tensors into a
        backend-typed :class:`BlockJacobian`.

        ``diag[k] = dR[k]/dx[k]`` and ``sub[k] = dR[k]/dx[k-1]``, both of
        shape ``(nblk_steps, ..., nstate, nstate)`` for the dense
        backend; other backends define their own raw contract.
        """


class DenseODEWrapper(ODEWrapper):
    """Default wrapper for dense torch tensors.

    Uses :class:`DenseBlockVector` and :class:`DenseBlockJacobian`.
    """

    def wrap_vector(self, raw: torch.Tensor) -> BlockVector:
        return DenseBlockVector(raw)

    def unwrap_vector(self, bv: BlockVector) -> torch.Tensor:
        if not isinstance(bv, DenseBlockVector):
            raise TypeError("DenseODEWrapper requires DenseBlockVector input.")
        return bv.data

    def wrap_jacobian(self, diag: torch.Tensor, sub: torch.Tensor) -> BlockJacobian:
        return DenseBlockJacobian(diag=diag, sub=sub)


class IntegrateODE(torch.nn.Module, NonlinearFunctionOperatorFactory):
    """Base class for ODE integration factories.

    Extends ``torch.nn.Module`` so the user's ODE parameters are discovered
    by the enclosing solver via ``parameters()``.

    Args:
        ode: ``torch.nn.Module`` whose ``forward(t, x)`` returns
            ``(x_dot, J_dot)`` as raw tensors.
        wrapper: bridge between the raw ODE outputs and the abstract block
            types used by the solver.
    """

    def __init__(
        self,
        ode: torch.nn.Module,
        wrapper: ODEWrapper | None = None,
    ) -> None:
        super().__init__()
        self.ode = ode
        self._wrapper = wrapper if wrapper is not None else DenseODEWrapper()

    @property
    def lookback(self) -> int:
        return 1

    @property
    def wrapper(self) -> ODEWrapper:
        return self._wrapper

    @wrapper.setter
    def wrapper(self, value: ODEWrapper) -> None:
        self._wrapper = value


class BackwardEulerODE(IntegrateODE):
    """Backward Euler integration as a :class:`NonlinearFunctionOperator` factory."""

    def evaluate_raw(
        self,
        x_full: torch.Tensor,
        forces: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, BlockJacobian]:
        t = forces[0]
        x_dot, J_dot = self.ode(t, x_full)
        dt = torch.diff(t, dim=0)
        if dt.dim() == x_full.dim():
            dt = dt.squeeze(-1)
        sblk = x_full.shape[-1]
        I_eye = torch.eye(sblk, dtype=x_full.dtype, device=x_full.device)

        R = x_full[1:] - x_full[:-1] - x_dot[1:] * dt[..., None]
        diag = I_eye - J_dot[1:] * dt[..., None, None]
        sub = -I_eye.expand_as(J_dot[1:])
        return R, self.wrapper.wrap_jacobian(diag, sub)

    def make_operator(
        self,
        prev_solution: torch.Tensor,
        forces: Sequence[torch.Tensor],
        inverse_operator,
    ) -> NonlinearFunctionOperator:
        return ChunkOp(self, prev_solution, forces, inverse_operator)


class ForwardEulerODE(IntegrateODE):
    """Forward Euler integration as a :class:`NonlinearFunctionOperator` factory."""

    def evaluate_raw(
        self,
        x_full: torch.Tensor,
        forces: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, BlockJacobian]:
        t = forces[0]
        x_dot, J_dot = self.ode(t, x_full)
        dt = torch.diff(t, dim=0)
        if dt.dim() == x_full.dim():
            dt = dt.squeeze(-1)
        sblk = x_full.shape[-1]
        I_eye = torch.eye(sblk, dtype=x_full.dtype, device=x_full.device)

        R = x_full[1:] - x_full[:-1] - x_dot[:-1] * dt[..., None]
        diag = I_eye.expand_as(J_dot[:-1]).contiguous()
        sub = -I_eye.expand_as(J_dot[:-1]) - J_dot[:-1] * dt[..., None, None]
        return R, self.wrapper.wrap_jacobian(diag, sub)

    def make_operator(
        self,
        prev_solution: torch.Tensor,
        forces: Sequence[torch.Tensor],
        inverse_operator,
    ) -> NonlinearFunctionOperator:
        return ChunkOp(self, prev_solution, forces, inverse_operator)
