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

"""Contract tests: prove that RecursiveNonlinearEquationSolver and the
chunktime Newton solvers consume their inputs solely through the
abstract BlockJacobian / BlockVector interfaces with no reach-through
to backend storage.

Two complementary checks:

1. ``BlockJacobian`` contract: a thin opaque BlockJacobian (with
   ``__slots__`` forbidding ``.diag`` / ``.sub`` access) runs a full
   solve + solve_adjoint and matches the dense reference.

2. ``BlockVector`` contract: a static grep guarantees no ``.data[`` patterns
   in the solver core (``chunktime.py`` / ``nonlinear.py``), and a
   behavioral check confirms ``ChunkNewtonRaphson`` actually exercises
   the new abstract primitives (``where``, ``flat_norm``,
   ``scale_batches``) rather than working around them via ``.data``.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import torch

from pyzag import chunktime, nonlinear, ode
from pyzag.ode import DenseODEWrapper
from pyzag.operators.base import BlockJacobian, BlockVector
from pyzag.operators.dense import DenseBlockJacobian, DenseBlockVector

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)


class LogisticODE(torch.nn.Module):
    """Logistic growth ODE rate: dy/dt = r * (1 - y/K) * y."""

    def __init__(self, r, K):
        super().__init__()
        self.r = torch.tensor(r)
        self.K = torch.tensor(K)

    def forward(self, t, y):
        y_dot = self.r * (1.0 - y / self.K) * y
        J_dot = (self.r - (2 * self.r * y) / self.K)[..., None]
        return y_dot, J_dot

    def y0(self, nbatch):
        return torch.linspace(0, 1, nbatch).reshape(nbatch, 1)


class LinearSystem(torch.nn.Module):
    """Linear ODE rate: dy/dt = A y, with A symmetric positive."""

    def __init__(self, n, seed=None):
        super().__init__()
        self.n = n
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(seed)
            Ap = torch.rand((n, n), generator=g)
        else:
            Ap = torch.rand((n, n))
        self.A = torch.nn.Parameter(Ap.transpose(0, 1) * Ap)

    def forward(self, t, y):
        y_dot = torch.matmul(self.A.unsqueeze(0).unsqueeze(0), y.unsqueeze(-1)).squeeze(
            -1
        )
        J_dot = (
            self.A.unsqueeze(0)
            .unsqueeze(0)
            .expand(t.shape[0], t.shape[1], self.n, self.n)
        )
        return y_dot, J_dot

    def y0(self, nbatch):
        return torch.outer(torch.linspace(-1, 1, nbatch), torch.linspace(1, 2, self.n))


class OpaqueBlockJacobian(BlockJacobian):
    """A BlockJacobian that exposes ONLY the abstract interface.

    Internally delegates to a DenseBlockJacobian, but uses ``__slots__``
    to forbid storing or accessing ``diag`` / ``sub`` / any other
    backend-specific attribute. If the solver ever reached for
    ``J.diag`` or ``J.sub`` or sliced ``J[...]``, the test would
    fail with AttributeError.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: DenseBlockJacobian) -> None:
        object.__setattr__(self, "_inner", inner)

    @property
    def device(self) -> torch.device:
        return self._inner.device

    @property
    def dtype(self) -> torch.dtype:
        return self._inner.dtype

    @property
    def nblk_steps(self) -> int:
        return self._inner.nblk_steps

    @property
    def batch_size(self) -> int:
        return self._inner.batch_size

    @property
    def block_size(self) -> int:
        return self._inner.block_size

    def forward_system(self, inverse_operator):
        return self._inner.forward_system(inverse_operator)

    def adjoint_system(self, inverse_operator):
        return self._inner.adjoint_system(inverse_operator)

    def solve_terminal_adjoint(self, g_terminal: torch.Tensor) -> BlockVector:
        return self._inner.solve_terminal_adjoint(g_terminal)

    def couple_prev_chunk(self, a_first: BlockVector) -> BlockVector:
        return self._inner.couple_prev_chunk(a_first)

    def as_adjoint_walk(self) -> "OpaqueBlockJacobian":
        return OpaqueBlockJacobian(self._inner.as_adjoint_walk())


class OpaqueWrapper(DenseODEWrapper):
    """ODEWrapper that produces OpaqueBlockJacobian instead of
    DenseBlockJacobian. Vector wrap/unwrap is still dense; only the
    Jacobian path is opaque to the solver.
    """

    def wrap_jacobian(self, diag: torch.Tensor, sub: torch.Tensor):
        inner = DenseBlockJacobian(diag=diag, sub=sub)
        return OpaqueBlockJacobian(inner)


class TestBlockJacobianContract(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.n = 3
        self.nbatch = 3
        self.nchunk = 5
        self.ntime = 50

    def _build(self, wrapper):
        rate = LinearSystem(self.n, seed=42)
        model = ode.BackwardEulerODE(rate, wrapper=wrapper)
        times = (
            torch.linspace(0, 1.0, self.ntime)
            .unsqueeze(-1)
            .expand(-1, self.nbatch)
            .unsqueeze(-1)
        )
        y0 = rate.y0(self.nbatch)
        solver = nonlinear.RecursiveNonlinearEquationSolver(
            model, step_generator=nonlinear.StepGenerator(self.nchunk)
        )
        return solver, y0, times

    def test_solve_matches_dense(self):
        ref_solver, y0, times = self._build(DenseODEWrapper())
        opaque_solver, _, _ = self._build(OpaqueWrapper())

        ref = nonlinear.solve(ref_solver, y0, self.ntime, times)
        opaque = nonlinear.solve(opaque_solver, y0, self.ntime, times)

        self.assertTrue(torch.allclose(ref, opaque))

    def test_solve_adjoint_matches_dense(self):
        ref_solver, y0, times = self._build(DenseODEWrapper())
        opaque_solver, _, _ = self._build(OpaqueWrapper())

        # Adjoint gradients via opaque path
        res_opaque = nonlinear.solve_adjoint(opaque_solver, y0, self.ntime, times)
        torch.linalg.norm(res_opaque).backward()
        grads_opaque = [p.grad.clone() for p in opaque_solver.parameters()]

        # Reference: full AD through the dense path
        res_ref = nonlinear.solve(ref_solver, y0, self.ntime, times)
        torch.linalg.norm(res_ref).backward()
        grads_ref = [p.grad.clone() for p in ref_solver.parameters()]

        for g_ref, g_opaque in zip(grads_ref, grads_opaque):
            self.assertTrue(torch.allclose(g_ref, g_opaque))

    def test_opaque_jacobian_has_no_diag_or_sub(self):
        # Sanity check that the opaque type really doesn't expose backend storage.
        wrapper = OpaqueWrapper()
        diag = torch.zeros(2, 1, 3, 3)
        sub = torch.zeros(2, 1, 3, 3)
        J = wrapper.wrap_jacobian(diag, sub)
        self.assertIsInstance(J, OpaqueBlockJacobian)
        with self.assertRaises(AttributeError):
            _ = J.diag
        with self.assertRaises(AttributeError):
            _ = J.sub


class TestBlockVectorContract(unittest.TestCase):
    """Static + behavioral contract that the solver core uses BlockVector
    only through its abstract interface (no ``.data[`` reach-through).

    The grep is scoped to the actual solver methods via :mod:`inspect` so
    sibling utility classes that have their own unrelated ``self.data``
    (e.g., :class:`SquareBatchedBlockDiagonalMatrix`) don't false-positive.
    """

    # Method bodies that must NOT contain `.data[` or `torch.norm(...data...)`.
    SOLVER_CORE_METHODS = [
        chunktime.ChunkNewtonRaphson.solve,
        chunktime.ChunkNewtonRaphson.step,
        chunktime.ChunkNewtonRaphsonLineSearch.step,
        nonlinear.RecursiveNonlinearEquationSolver.solve,
        nonlinear.RecursiveNonlinearEquationSolver.block_update,
        nonlinear.RecursiveNonlinearEquationSolver.rewind,
        nonlinear.RecursiveNonlinearEquationSolver.block_update_adjoint,
        nonlinear.RecursiveNonlinearEquationSolver.accumulate,
        nonlinear.ChunkOp.__call__,
    ]

    def _solver_sources(self):
        import inspect

        return [
            (fn.__qualname__, inspect.getsource(fn)) for fn in self.SOLVER_CORE_METHODS
        ]

    def test_no_data_indexing_in_solver_methods(self):
        # Reach-through would look like `x.data[`, `R.data[`, etc.
        pattern = re.compile(r"\.data\[")
        offenders = [
            f"{name}: {line.rstrip()}"
            for name, src in self._solver_sources()
            for line in src.splitlines()
            if pattern.search(line)
        ]
        self.assertEqual(
            offenders,
            [],
            "Solver core methods must not index into BlockVector .data; found:\n"
            + "\n".join(offenders),
        )

    def test_no_torch_norm_on_data_in_solver_methods(self):
        pattern = re.compile(r"torch\.norm\([^)]*\.data")
        offenders = [
            f"{name}: {line.rstrip()}"
            for name, src in self._solver_sources()
            for line in src.splitlines()
            if pattern.search(line)
        ]
        self.assertEqual(
            offenders,
            [],
            "Solver core methods must not call torch.norm on .data; found:\n"
            + "\n".join(offenders),
        )

    def _drive_newton_with_counters(self, method_names, nonlinear_solver=None):
        """Patch DenseBlockVector methods to count calls, then drive a
        small forward solve. Returns a dict of method_name -> call count.
        Patching at the class level catches calls regardless of which
        DenseBlockVector instance fires them (subclasses, arithmetic
        results, etc.)."""
        counts = {name: 0 for name in method_names}
        originals = {name: getattr(DenseBlockVector, name) for name in method_names}

        def make_counter(name, orig):
            def wrapper(self, *args, **kwargs):
                counts[name] += 1
                return orig(self, *args, **kwargs)

            return wrapper

        for name, orig in originals.items():
            setattr(DenseBlockVector, name, make_counter(name, orig))

        try:
            rate = LogisticODE(1.0, 1.0)
            model = ode.BackwardEulerODE(rate)
            nbatch, ntime = 4, 30
            times = (
                torch.linspace(0, 1.0, ntime)
                .unsqueeze(-1)
                .expand(-1, nbatch)
                .unsqueeze(-1)
            )
            y0 = rate.y0(nbatch)
            kwargs = {"step_generator": nonlinear.StepGenerator(5)}
            if nonlinear_solver is not None:
                kwargs["nonlinear_solver"] = nonlinear_solver
            solver = nonlinear.RecursiveNonlinearEquationSolver(model, **kwargs)
            nonlinear.solve(solver, y0, ntime, times)
        finally:
            for name, orig in originals.items():
                setattr(DenseBlockVector, name, orig)

        return counts

    def test_newton_exercises_where_norm_clone(self):
        counts = self._drive_newton_with_counters(["where", "norm", "clone"])

        self.assertGreater(
            counts["where"], 0, "Newton step did not call BlockVector.where()"
        )
        self.assertGreater(
            counts["norm"], 0, "Newton step did not call BlockVector.norm()"
        )
        # `clone` is used inside the bidiagonal solvers (Thomas/PCR).
        self.assertGreater(counts["clone"], 0)

    def test_line_search_exercises_flat_norm_and_scale_batches(self):
        counts = self._drive_newton_with_counters(
            ["flat_norm", "scale_batches"],
            nonlinear_solver=chunktime.ChunkNewtonRaphsonLineSearch(),
        )

        self.assertGreater(
            counts["flat_norm"],
            0,
            "Line search did not call BlockVector.flat_norm()",
        )
        self.assertGreater(
            counts["scale_batches"],
            0,
            "Line search did not call BlockVector.scale_batches()",
        )


if __name__ == "__main__":
    unittest.main()
