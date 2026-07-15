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

r"""Solving recursive nonlinear equations and computing parameter sensitivities
via the adjoint method.

The solver integrates a recursive system whose residual couples consecutive
steps (lookback 1):

.. math::

    R_k = f(x_k, x_{k-1};\, p) = 0, \qquad k = 1, \dots, n,

solving the trajectory in chunks with a blocked Newton iteration. Parameter
gradients are obtained by a reverse adjoint sweep that reuses the same chunk
Jacobians, giving memory cost independent of ``n`` rather than backpropagating
through every step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Sequence

import warnings

import torch

from pyzag import chunktime
from pyzag.operators.base import BlockJacobian, BlockVector


def _disable_donated_buffers() -> None:
    """Lower torch's AOTAutograd ``donated_buffer`` default so pyzag's adjoint works
    with ``torch.compile``.

    pyzag's adjoint reuses the autograd graph across the reverse sweep via
    ``torch.autograd.grad(..., retain_graph=True)`` (see
    :meth:`RecursiveNonlinearEquationSolver.accumulate`). On torch>=2.12 AOTAutograd
    collects "donated buffers" for a ``torch.compile``'d graph, and a backward compiled
    with non-empty donated buffers *requires* ``retain_graph=False``. So a compiled model
    differentiated through pyzag's adjoint raises
    ``RuntimeError: ... compiled with non-empty donated buffers ...``.

    ``donated_buffer`` is a ``ContextVar``-backed torch config. AOTAutograd compiles and
    runs the backward under its own contextvars contexts, where a normal
    ``torch._functorch.config.donated_buffer = False`` (or ``config.patch``) override is
    not visible -- those contexts read the config *default*. So the only setting that
    actually reaches them is the default, which we lower here, process-wide.

    This is deliberately blunt: it lowers a global torch default the first time a
    :class:`RecursiveNonlinearEquationSolver` is constructed. It is scoped to actual
    pyzag use -- merely importing pyzag does not touch the default, so code
    that never runs the adjoint is unaffected. A ``UserWarning`` is emitted once, when it
    takes effect. To keep donated buffers enabled, restore the default::

        import torch._functorch.config as c
        c._config["donated_buffer"].default = True

    The consequence of restoring it: any ``torch.compile``'d model differentiated through
    pyzag's adjoint will raise the donated-buffer error again. (Donation saves a little
    backward memory, but is fundamentally incompatible with ``retain_graph=True``.)
    """
    try:
        import torch._functorch.config as functorch_config

        # pylint: disable=protected-access
        entry = functorch_config._config["donated_buffer"]
    except (ImportError, KeyError, AttributeError):
        return  # older/other torch without the flag -- nothing to disable

    if getattr(entry, "default", False):
        entry.default = False
        warnings.warn(
            "pyzag lowered torch._functorch.config.donated_buffer's default to False "
            "process-wide. AOTAutograd 'donated buffers' (torch>=2.12) are incompatible "
            "with pyzag's retain_graph=True adjoint over torch.compile'd models, which "
            "otherwise raises 'compiled with non-empty donated buffers'. To keep donated "
            "buffers enabled, restore "
            "torch._functorch.config._config['donated_buffer'].default = True -- "
            "compiled models differentiated through the adjoint will then error.",
            UserWarning,
            stacklevel=3,
        )


class NonlinearFunctionOperator(ABC):
    r"""Abstract operator passed to the Newton solver.

    The solver integrates a recursive nonlinear system whose per-step residual
    has lookback ``L``:

        R_k = f(x_k, x_{k-1}, \dots, x_{k-L};\, u_k;\, p) = 0,
        \qquad x_k \in \mathbb{R}^n

    A chunk covers ``m`` consecutive steps. The forces ``u_k`` and the lookback
    tail ``x_0`` (the previous solution) are captured at construction; Newton
    only varies the chunk unknowns ``x = (x_1, ..., x_m)``. Calling the operator
    maps that trial state to ``(R, J)``:

    - ``R(x) = (R_1, ..., R_m)`` (a :class:`BlockVector`), with
      ``R_k = f(x_k, x_{k-1}; u_k)``.
    - ``J = dR/dx``. For ``L == 1`` this is block lower-bidiagonal, with
      diagonal blocks ``A_k = dR_k/dx_k`` and subdiagonal blocks
      ``B_k = dR_k/dx_{k-1}``, so that ``(J @ d)_k = A_k d_k + B_{k-1} d_{k-1}``
      and all other blocks are zero.

    Newton iterates ``x <- x - J^{-1} R(x)``, solving ``J d = R`` each iteration
    via the bidiagonal (Thomas/PCR) solver.

    Forces and time are captured at construction. Newton only passes the
    state ``x`` as a :class:`BlockVector` and receives back ``(R, J)``
    where ``R`` is a :class:`BlockVector` and ``J`` is a
    :class:`chunktime.BidiagonalForwardOperator`.
    """

    @abstractmethod
    def __call__(
        self, x: BlockVector
    ) -> tuple[BlockVector, "chunktime.BidiagonalForwardOperator"]:
        """Return ``(residual, Jacobian operator)`` for current state ``x``."""


class NonlinearFunctionOperatorFactory(ABC):
    """Factory for chunk-level :class:`NonlinearFunctionOperator` objects.

    The forward solve calls :meth:`make_operator` once per chunk to create
    a stateful operator that captures the chunk's previous-solution lookback
    and forces. The adjoint pass calls :meth:`evaluate_raw` to recompute
    the residual and Jacobian; the latter as a backend-typed
    :class:`BlockJacobian` consumed by the adjoint walk.

    Subclasses must expose ``lookback`` and ``wrapper`` (typically as
    properties) and implement :meth:`make_operator` and
    :meth:`evaluate_raw`. The canonical :class:`ChunkOp` below is a
    drop-in implementation that most subclasses can return from
    ``make_operator`` without modification.
    """

    @property
    @abstractmethod
    def lookback(self) -> int:
        """Number of previous solution steps the residual depends on.

        Declared abstract so a subclass that forgets it fails at instantiation
        rather than with a late ``AttributeError`` deep in the solver.
        """

    @property
    @abstractmethod
    def wrapper(self):
        """Backend bridge (an :class:`~pyzag.ode.ODEWrapper`-like object) that
        wraps/unwraps raw tensors as backend :class:`BlockVector` objects."""

    @abstractmethod
    def make_operator(
        self,
        prev_solution: torch.Tensor,
        forces: Sequence[torch.Tensor],
        inverse_operator,
    ) -> NonlinearFunctionOperator:
        """Build a :class:`NonlinearFunctionOperator` for a single chunk."""

    @abstractmethod
    def evaluate_raw(
        self,
        x_full: torch.Tensor,
        forces: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, BlockJacobian]:
        """Return ``(R, J)`` for adjoint reconstruction.

        ``x_full`` includes the lookback steps prepended to the chunk
        state. ``R`` is a raw torch tensor; ``J`` is a backend-typed
        :class:`BlockJacobian`.
        """


class ChunkOp(NonlinearFunctionOperator):
    """Generic per-chunk operator for any :class:`NonlinearFunctionOperatorFactory`.

    Holds the chunk's previous-solution lookback and forces; on each
    Newton call, assembles the full state, asks the factory for raw
    ``(R, J)``, then wraps R via the factory's :class:`ODEWrapper` and
    builds the chunk's forward bidiagonal system from the
    :class:`BlockJacobian`. Most factories can return ``ChunkOp(self, ...)``
    from their ``make_operator`` without modification.
    """

    def __init__(
        self,
        factory: NonlinearFunctionOperatorFactory,
        prev_solution: torch.Tensor,
        forces: Sequence[torch.Tensor],
        inverse_operator,
    ) -> None:
        self.factory = factory
        self.prev_solution = prev_solution
        self.forces = forces
        self.inverse_operator = inverse_operator

    def __call__(
        self, x: BlockVector
    ) -> tuple[BlockVector, "chunktime.BidiagonalForwardOperator"]:
        x_chunk_raw = self.factory.wrapper.unwrap_vector(x)
        x_full_raw = torch.cat([self.prev_solution, x_chunk_raw])

        R_raw, J = self.factory.evaluate_raw(x_full_raw, self.forces)

        R = self.factory.wrapper.wrap_vector(R_raw)
        return R, J.forward_system(self.inverse_operator)


class FullTrajectoryPredictor:
    """Predict steps using a complete user-provided trajectory."""

    def __init__(self, history: torch.Tensor) -> None:
        self.history = history

    def predict(self, results: torch.Tensor, k: int, kinc: int) -> torch.Tensor:
        # pylint: disable=W0613
        return self.history[k : k + kinc]


class ZeroPredictor:
    """Predict steps just using zeros."""

    def predict(self, results: torch.Tensor, k: int, kinc: int) -> torch.Tensor:
        return torch.zeros_like(results[k : k + kinc])


class PreviousStepsPredictor:
    """Predict by providing the values from the previous chunk of steps."""

    def predict(self, results: torch.Tensor, k: int, kinc: int) -> torch.Tensor:
        if k - kinc < 0:
            res = torch.zeros_like(results[k : k + kinc])
            res[kinc - k :] = results[0:k]
            res[: kinc - k] = results[0]
            return res

        return results[(k - kinc) : k]


class LastStepPredictor:
    """Predict by providing the values from the previous single step."""

    def predict(self, results: torch.Tensor, k: int, kinc: int) -> torch.Tensor:
        if k < 1:
            return torch.zeros_like(results[k : k + kinc])

        return results[k - 1].unsqueeze(0).expand((kinc,) + results.shape[1:])


class StepExtrapolatingPredictor:
    """Predict by extrapolating using the previous chunks of steps."""

    def predict(self, results: torch.Tensor, k: int, kinc: int) -> torch.Tensor:
        if k < 2:
            return torch.zeros_like(results[k : k + kinc])

        dinc = (results[k - 1] - results[k - 2]) + results[k - 1]

        return dinc.unsqueeze(0).expand((kinc,) + results.shape[1:])


class ExtrapolatingPredictor:
    """Predict by extrapolating the values from the previous single steps."""

    def predict(self, results: torch.Tensor, k: int, kinc: int) -> torch.Tensor:
        if k - kinc < 0:
            res = torch.zeros_like(results[k : k + kinc])
            res[kinc - k :] = results[0:k]
            res[: kinc - k] = results[0]
            return res
        if k - 2 * kinc - 1 < 0:
            return results[(k - kinc) : k]

        inc = results[(k - kinc) : k] - results[(k - 2 * kinc) : k - kinc]

        return results[(k - kinc) : k] + inc


class StepGenerator:
    """Generate chunks of recursive steps to produce at once."""

    def __init__(self, block_size: int = 1, first_block_size: int = 0) -> None:
        self.block_size = block_size
        self.offset_step = first_block_size
        self.back = False

        self.n = 0
        self.steps: list[int] = []
        self.pairs: list[tuple[int, int]] = []
        self.i = 0

    def __call__(self, n: int) -> "StepGenerator":
        self.back = False
        self.n = n
        self.steps = [1]
        if self.offset_step > 0:
            self.steps += [self.offset_step + 1]
        self.steps += list(range(self.steps[-1], n, self.block_size))[1:] + [n]

        self.pairs = list(zip(self.steps[:-1], self.steps[1:]))

        self.i = 0

        return self

    def __iter__(self) -> "StepGenerator":
        return self

    def reverse(self) -> "StepGenerator":
        """Reverse the iterator to yield chunks starting from the end."""
        self.back = True
        rev = [
            (self.n - k2, self.n - k1)
            for k1, k2 in zip(self.steps[:-1], self.steps[1:])
        ][:-1]
        if not rev:
            # Single forward chunk: the ``[:-1]`` above drops the only reversed
            # pair, so re-anchor the sole reverse chunk at block 1 (both the
            # terminal seed ``k2 + 1 == n`` and the first block). Needs n >= 3
            # (two transitions) for a non-degenerate chunk; a 2-point trajectory
            # has a single transition the chunked adjoint cannot represent.
            if self.n < 3:
                raise NotImplementedError(
                    "The adjoint pass requires at least 3 time steps in a single "
                    f"chunk (got n={self.n}). Use more time steps."
                )
            self.pairs = [(1, self.n - 1)]
        else:
            if rev[-1][0] != 1:
                rev += [(1, rev[-1][0])]
            self.pairs = rev

        self.i = 0

        return self

    def __next__(self) -> tuple[int, int]:
        self.i += 1
        if self.i <= len(self.pairs):
            return self.pairs[self.i - 1]
        raise StopIteration


class InitialOffsetStepGenerator(StepGenerator):
    """Generate chunks of recursive steps with optional user-provided initial chunks."""

    def __init__(
        self, *args, initial_steps: Sequence[int] | None = None, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        if initial_steps is None:
            self.initial_steps: list[int] = []
        else:
            self.initial_steps = list(initial_steps)

    def __call__(self, n: int) -> "InitialOffsetStepGenerator":
        self.back = False
        self.n = n
        self.steps = [1]
        if len(self.initial_steps) > 0:
            self.steps += [i + 1 for i in self.initial_steps]
        self.steps += list(range(self.steps[-1], n, self.block_size))[1:] + [n]

        self.pairs = list(zip(self.steps[:-1], self.steps[1:]))

        self.i = 0

        return self


class RecursiveNonlinearEquationSolver(torch.nn.Module):
    """Generates a time series from a recursive nonlinear equation and (optionally) uses the adjoint method to provide derivatives.

    ``func`` is a :class:`NonlinearFunctionOperatorFactory`. The factory
    knows how to create chunk-level operators (for the forward solve) and
    how to recompute residuals / Jacobians (for the adjoint). Built-in
    factories include :class:`pyzag.ode.BackwardEulerODE` and
    :class:`pyzag.ode.ForwardEulerODE`; users can also subclass
    :class:`NonlinearFunctionOperatorFactory` directly and return
    :class:`ChunkOp` from their ``make_operator``.
    """

    def __init__(
        self,
        func: NonlinearFunctionOperatorFactory,
        step_generator: StepGenerator = StepGenerator(1),
        predictor=ZeroPredictor(),
        direct_solve_operator=chunktime.BidiagonalThomasFactorization,
        nonlinear_solver: chunktime.ChunkNewtonRaphson = chunktime.ChunkNewtonRaphson(),
        callbacks: Sequence[Callable[[torch.Tensor], torch.Tensor]] | None = None,
        convert_nan_gradients: bool = True,
    ) -> None:
        super().__init__()
        # Lower torch's AOTAutograd donated-buffer default so the adjoint (which uses
        # retain_graph=True) works with torch.compile'd models. Scoped here to actual
        # solver use, and runs before any solve compiles a backward. Process-global and
        # emits a one-time warning; see _disable_donated_buffers.
        _disable_donated_buffers()
        self.func = func

        self.direct_solve_operator = direct_solve_operator
        self.step_generator = step_generator
        self.predictor = predictor
        self.nonlinear_solver = nonlinear_solver

        # Backward cache
        self.n = 0
        self.forces: list[torch.Tensor] = []
        self.result: torch.Tensor | None = None
        self.adjoint_params: Sequence[torch.Tensor] = []

        if self.func.lookback != 1:
            raise ValueError(
                f"The function factory has lookback = {self.func.lookback}, "
                "but the current solver only handles lookback = 1!"
            )

        self.callbacks = callbacks
        self.convert_nan_gradients = convert_nan_gradients

    def forward(self, *args, **kwargs):
        """Alias for solve."""
        return self.solve(*args, **kwargs)

    def solve(
        self,
        y0: torch.Tensor,
        n: int,
        *args: torch.Tensor,
        adjoint_params: Sequence[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Solve the recursive equations for n steps."""
        self._check_shapes(y0, n, args)

        result = torch.empty(n, *y0.shape, dtype=y0.dtype, device=y0.device)
        result[0] = y0

        for k1, k2 in self.step_generator(n):
            result[k1:k2] = self.block_update(
                result[k1 - self.func.lookback : k1].clone(),
                self.predictor.predict(result, k1, k2 - k1).clone(),
                [arg[k1 - self.func.lookback : k2].clone() for arg in args],
            )
            if self.callbacks is not None:
                for fn in self.callbacks:
                    result[k1:k2] = fn(result[k1:k2])

        # Cache the trajectory whenever an adjoint pass is possible. ``None`` is
        # the sentinel for a plain (non-adjoint) solve; an *empty* tuple is a
        # valid adjoint solve with no differentiable parameters (e.g. computing
        # only d(loss)/dy0), so it must still cache — hence ``is not None``
        # rather than a truthiness check.
        if adjoint_params is not None:
            self.n = n
            self.forces = [arg.clone() for arg in args]
            self.result = result.clone()
            self.adjoint_params = adjoint_params

        return result

    def block_update(
        self,
        prev_solution: torch.Tensor,
        solution: torch.Tensor,
        forces: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Solve one chunk via Newton.

        Builds a chunk-level operator from the factory, wraps the initial
        guess as a :class:`BlockVector`, runs Newton, and unwraps back to
        a raw tensor for the result array.
        """
        fn = self.func.make_operator(prev_solution, forces, self.direct_solve_operator)
        x0 = self.func.wrapper.wrap_vector(solution)
        result = self.nonlinear_solver.solve(fn, x0)
        return self.func.wrapper.unwrap_vector(result)

    def rewind(
        self, output_grad: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Rewind through an adjoint pass to provide the dot product for each quantity in output_grad."""
        grad_result = tuple(
            torch.zeros(p.shape, device=output_grad.device) for p in self.adjoint_params
        )

        for k1, k2 in self.step_generator(len(self.result)).reverse():
            with torch.enable_grad():
                R, J = self.func.evaluate_raw(
                    self.result[k1 - 1 : k2 + 1],
                    [f[k1 - 1 : k2 + 1] for f in self.forces],
                )
                R = R.flip(0)
                J_walk = J.as_adjoint_walk()

            if k2 + 1 == len(self.result):
                adjoint = J_walk.solve_terminal_adjoint(output_grad[-1])
                with torch.enable_grad():
                    grad_result = self.accumulate(
                        grad_result, adjoint, R[:1], retain=True
                    )

            adjoint = self.block_update_adjoint(
                J_walk,
                self.func.wrapper.wrap_vector(output_grad[k1:k2].flip(0)),
                adjoint[-1:],
            )

            with torch.enable_grad():
                grad_result = self.accumulate(grad_result, adjoint, R[1:])

        adj_last = self.func.wrapper.unwrap_vector(adjoint[-1:]).squeeze(0)

        if self.convert_nan_gradients:
            return tuple(torch.nan_to_num(g) for g in grad_result), torch.nan_to_num(
                adj_last
            )

        return grad_result, adj_last

    def accumulate(
        self,
        grad_result: tuple[torch.Tensor, ...],
        adjoint: BlockVector,
        R: torch.Tensor,
        retain: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Accumulate the updated gradient values."""
        # No differentiable parameters (e.g. an IC-only adjoint solve): there is
        # nothing to accumulate, and ``torch.autograd.grad`` rejects an empty
        # ``inputs``. The y0 gradient is carried entirely by the adjoint
        # recursion (``adj_last``), independent of this accumulation.
        if not self.adjoint_params:
            return grad_result
        full_adjoint = self.func.wrapper.unwrap_vector(adjoint)
        g = torch.autograd.grad(
            R,
            self.adjoint_params,
            full_adjoint,
            retain_graph=retain,
            allow_unused=True,
        )
        return tuple(
            pi + gi if gi is not None else pi for pi, gi in zip(grad_result, g)
        )

    def block_update_adjoint(
        self,
        J: BlockJacobian,
        grads: BlockVector,
        a_prev: BlockVector,
    ) -> BlockVector:
        """Do the blocked adjoint solve.

        ``J`` is the chunk's :class:`BlockJacobian` in adjoint-walk order
        (i.e. the result of :meth:`BlockJacobian.as_adjoint_walk`). The
        single-block inter-chunk coupling is delegated to
        :meth:`BlockJacobian.couple_prev_chunk` so the solver never
        touches raw tensor indexing.
        """
        operator = J.adjoint_system(self.direct_solve_operator)

        # ``-grads`` already allocates a fresh vector (__neg__ copies), so the
        # in-place boundary update below is safe without an extra clone.
        rhs = -grads
        rhs[0:1] = rhs[0:1] - J.couple_prev_chunk(a_prev)

        return operator.matvec(rhs)

    def _check_shapes(
        self, y0: torch.Tensor, n: int, forces: Sequence[torch.Tensor]
    ) -> None:
        """Check the shapes of everything before starting the calculation."""
        correct_force_batch_shape = (n,) + y0.shape[:-1]
        for f in forces:
            if f.shape[:-1] != correct_force_batch_shape:
                raise ValueError(
                    "One of the provided driving forces does not have the correct shape. "
                    "The batch shape should be "
                    + str(correct_force_batch_shape)
                    + " but is instead "
                    + str(f.shape[:-1])
                )


class AdjointWrapper(torch.autograd.Function):
    # pylint: disable=all
    """Defines the backward pass for pytorch, allowing us to mix the adjoint calculation with AD."""

    @staticmethod
    def forward(solver, y0, n, forces, *params):
        with torch.no_grad():
            y = solver.solve(y0, n, *forces, adjoint_params=params)
            return y

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.solver = inputs[0]

    @staticmethod
    def backward(ctx, output_grad):
        with torch.no_grad():
            grad_res, adj_last = ctx.solver.rewind(output_grad)
            if ctx.needs_input_grad[1]:
                # ``y[0] == y0``, so the total y0 gradient is the adjoint
                # propagated back through the recursion (``-adj_last``) PLUS the
                # objective's *direct* dependence on the returned first block
                # (``output_grad[0]``). Omitting the direct term under-counts
                # d(loss)/dy0 when the loss depends on the returned initial state.
                return (None, -adj_last + output_grad[0], None, None, *grad_res)
            return (None, None, None, None, *grad_res)


def solve(solver, y0, n, *forces):
    """Solve a :class:`RecursiveNonlinearEquationSolver` without the adjoint method."""
    return solver.solve(y0, n, *forces)


def solve_adjoint(solver, y0, n, *forces):
    """Apply a :class:`RecursiveNonlinearEquationSolver` to solve adjoint-differentiably."""
    additional_params = []
    for m in solver.modules():
        if hasattr(m, "converted_params"):
            additional_params.extend(getattr(m, p) for p in m.converted_params)

    all_params = list(solver.parameters()) + additional_params

    return AdjointWrapper.apply(solver, y0, n, forces, *all_params)
