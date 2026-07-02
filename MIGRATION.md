# Migration Guide: pyzag 1.x → 2.0

pyzag 2.0 introduces a block-operator abstraction layer and a NEML2 backend
alongside the existing dense implementation. The dense path remains the default
and is functionally equivalent to 1.x, but the public Python API has changed in
ways that will break code written against 1.x.

This document lists every user-visible break and shows the minimal port for
each.

---

## At a glance

| Area | Change |
|---|---|
| User-defined residuals | `NonlinearRecursiveFunction` removed; subclass `NonlinearFunctionOperatorFactory` and return a `ChunkOp` instead |
| Block storage | New `pyzag.operators` subpackage with `BlockVector` / `BlockOperator` / `BlockJacobian` abstractions; dense backend is `pyzag.operators.dense` |
| Backends | New `pyzag.operators.neml2` backend implementing the same abstractions on top of NEML2-compatible types (zero `import neml2` in pyzag itself) |
| `IntegrateODE` family | Now extends `torch.nn.Module, NonlinearFunctionOperatorFactory`; takes an optional `wrapper` argument |
| `thomas_solve` | Signature operates on `BlockOperator` / `BlockVector` rather than LU factors and raw tensors |
| Block operator indexing | `block(i)` and `window(start, end)` replaced by `__getitem__` (`A[i:i+1]`, `A[i:j]`) |
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
swap in different backends (dense, NEML2) without changing the residual.

```python
from pyzag.nonlinear import NonlinearFunctionOperatorFactory, ChunkOp

class MyResidualFactory(NonlinearFunctionOperatorFactory):
    def make_operator(self, ...) -> ChunkOp:
        ...
        return ChunkOp(...)
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

### Block window API

The methods `block(i)` and `window(start, end)` on operators have been
replaced by Python's standard `__getitem__`:

```python
# 1.x
A.block(i)
A.window(start, end)

# 2.0
A[i:i+1]   # single-block operator
A[start:end]
```

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
own `factored()` constructor (see `DenseBlockOperator.factored`,
`NEML2SolvableBlockOperator.factored`).

If you were calling `thomas_solve` directly with raw tensors, you now need to
construct the appropriate `BlockOperator` first.

---

## 5. NEML2 backend (new in 2.0)

A NEML2-compatible backend now lives in the `pyzag/operators/neml2/`
subpackage (`_assembly.py`, `_operator.py`, `_pcr.py`, `_vector.py`,
`_containers.py`, `_jacobian.py`, `_wrapper.py`). It implements pyzag's
`BlockVector` / `SolvableBlockOperator` / `BlockJacobian` ABCs on top of
NEML2-style assembled types (`AssembledMatrix`, `AxisLayout`, `Tensor`).

Important: this subpackage contains **no `import neml2`** — it is a pure-pyzag
implementation that re-creates the relevant container types. NEML2 users get
the backend for free; non-NEML2 users see no new runtime dependency.

The user-facing factory that adapts a NEML2 `NonlinearSystem` to pyzag lives
on the NEML2 side, in `neml2.pyzag.interface.NEML2PyzagFactory`.

---

## 6. Bug fix: `StepExtrapolatingPredictor.predict()`

`StepExtrapolatingPredictor.predict()` previously guarded with `if k < 1`,
which let `k == 1` fall through to `results[k - 2] == results[-1]` — a silent
index wrap that read the *last* element of the trajectory rather than failing
or returning zeros. The guard is now `if k < 2`. A dead expression on the
same path (a value computed and discarded) has also been removed.

If you were relying on the buggy behavior, you can replicate it explicitly,
but the math was almost certainly not what you intended.

---

## 7. Removed / unchanged

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

---

## Test-suite changes (internal-facing)

If you were running pyzag's test suite directly, two changes:

- The vestigial `try/except` + `_NEML2_AVAILABLE` + `@unittest.skipUnless`
  scaffolding has been removed from the NEML2-backend test files. Those tests
  now run unconditionally and pass without NEML2 being installed (the backend
  has no `import neml2`).
- The NEML2 AD-graph / adjoint tests have been moved out of pyzag and into
  the NEML2 repository (`tests/unit/pyzag/`, e.g. `test_adjoint.py`,
  `test_adjoint_multigroup.py`, `test_neml2_pyzag_factory.py`). They exercise
  the NEML2-side adapter (`neml2.pyzag.interface.NEML2PyzagFactory`) rather
  than anything in pyzag itself, so they live with the code they test.
