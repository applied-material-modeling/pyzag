# Migration Guide: pyzag 1.x → 2.0

pyzag 2.0 introduces a block-operator abstraction layer alongside the existing
dense implementation. The dense path remains the default and is functionally
equivalent to 1.x, but the public Python API has changed in ways that will break
code written against 1.x.

This document lists every user-visible break and shows the minimal port for
each.

---

## At a glance

| Area | Change |
|---|---|
| User-defined residuals | `NonlinearRecursiveFunction` removed; subclass `NonlinearFunctionOperatorFactory` and return a `ChunkOp` instead |
| Block storage | New `pyzag.operators` subpackage with `BlockVector` / `BlockOperator` / `BlockJacobian` abstractions; dense backend is `pyzag.operators.dense` |
| Backends | pyzag ships the dense backend; the `pyzag.operators` abstractions let alternate backends be implemented against the same interfaces |
| `IntegrateODE` family | Now extends `torch.nn.Module, NonlinearFunctionOperatorFactory`; takes an optional `wrapper` argument |
| `thomas_solve` | Signature operates on `BlockOperator` / `BlockVector` rather than LU factors and raw tensors |
| `StepExtrapolatingPredictor` | Bug fix: silent index wrap when `k == 1` is now caught by the guard |

---

## 1. Defining the residual

**1.x**

```python
from pyzag import nonlinear

class MyResidual(nonlinear.NonlinearRecursiveFunction):
    def forward(self, x_np1, x_n, *forces):
        ...
        return residual, jacobian
```

**2.0**

The base class `NonlinearRecursiveFunction` has been removed. Residuals are
now produced by a *factory* (`NonlinearFunctionOperatorFactory`) that returns
a *chunk operator* (`ChunkOp`). The factory pattern is what lets the solver
swap in different backends without changing the residual.

A `NonlinearFunctionOperatorFactory` subclass must provide four members: the
`lookback` and `wrapper` properties, `make_operator` (builds the per-chunk
operator the Newton solve calls), and `evaluate_raw` (recomputes `(R, J)` for
the adjoint pass). `ChunkOp` is a ready-made operator most factories can return
from `make_operator` unchanged.

```python
from pyzag.nonlinear import NonlinearFunctionOperatorFactory, ChunkOp

class MyResidualFactory(NonlinearFunctionOperatorFactory):
    @property
    def lookback(self) -> int: ...

    @property
    def wrapper(self): ...        # ODEWrapper-like backend bridge

    def make_operator(self, prev_solution, forces, inverse_operator) -> ChunkOp:
        return ChunkOp(self, prev_solution, forces, inverse_operator)

    def evaluate_raw(self, x_full, forces):
        ...
        return R, J             # R: raw tensor, J: BlockJacobian
```

If you were subclassing `IntegrateODE` (i.e. you only cared about ODE
integration, not arbitrary recursive systems), see section 3 — the port is
shorter.

---

## 2. Block storage: the `pyzag.operators` subpackage

2.0 introduces an abstract block-operator API in `pyzag/operators/base.py`:

- `BlockVector` — abstract block vector (axis 0 is the time/block index)
- `BlockOperator`, `SolvableBlockOperator` — abstract block operators
- `BlockJacobian` — bidiagonal Jacobian abstraction
- `PCRState` — backend state for parallel cyclic reduction

The dense implementation (drop-in for 1.x behavior) lives in
`pyzag.operators.dense`:

- `DenseBlockVector`
- `DenseBlockOperator`
- `DenseBlockJacobian`

If you were passing raw tensors directly to internal `chunktime` machinery,
wrap them in the dense types instead. For most users this is invisible — the
solver constructs them internally.

---

## 3. `IntegrateODE` / `BackwardEulerODE` / `ForwardEulerODE`

**1.x** — `IntegrateODE` inherited from `NonlinearRecursiveFunction`:

```python
from pyzag import ode

stepper = ode.BackwardEulerODE(my_ode_module)
```

**2.0** — `IntegrateODE` inherits from `torch.nn.Module` and
`NonlinearFunctionOperatorFactory`, and takes an optional `wrapper` to select
the backend bridge. The default is the dense backend, so the call site is
unchanged for the common case:

```python
from pyzag import ode

stepper = ode.BackwardEulerODE(my_ode_module)        # dense (default)
# or, opting into a different backend:
# stepper = ode.BackwardEulerODE(my_ode_module, wrapper=my_wrapper)
```

The `ode` module ABC for the bridge is `ODEWrapper`; `DenseODEWrapper` is the
default.

---

## 4. `chunktime.thomas_solve`

**1.x**

```python
def thomas_solve(lu, pivots, B, v):
    ...
```

Took a pre-LU-decomposed diagonal and raw `B` / `v` tensors.

**2.0**

```python
def thomas_solve(A: BlockOperator, B: BlockOperator, v: BlockVector) -> BlockVector:
    ...
```

Operates on block types directly. The LU step is delegated to the operator's
own `factored()` constructor (see `DenseBlockOperator.factored`).

If you were calling `thomas_solve` directly with raw tensors, you now need to
construct the appropriate `BlockOperator` first.

---

## 5. Bug fix: `StepExtrapolatingPredictor.predict()`

`StepExtrapolatingPredictor.predict()` previously guarded with `if k < 1`,
which let `k == 1` fall through to `results[k - 2] == results[-1]` — a silent
index wrap that read the *last* element of the trajectory rather than failing
or returning zeros. The guard is now `if k < 2`. A dead expression on the
same path (a value computed and discarded) has also been removed.

If you were relying on the buggy behavior, you can replicate it explicitly,
but the math was almost certainly not what you intended.

---

## 6. Removed / unchanged

- All three predictors that existed in 1.x but had no callers
  (`LastStepPredictor`, `StepExtrapolatingPredictor`, `ExtrapolatingPredictor`)
  are retained in 2.0 for users who may have wired them in externally.
- `FullTrajectoryPredictor`, `ZeroPredictor`, `PreviousStepsPredictor` are
  unchanged.
- `RangeRescale`, `Reparameterizer` (in `pyzag.reparametrization`) are
  unchanged.
- `MapNormal`, `HierarchicalStatisticalModel` (in `pyzag.stochastic`) are
  unchanged.
- `solve()` and `solve_adjoint()` top-level helpers are unchanged.
