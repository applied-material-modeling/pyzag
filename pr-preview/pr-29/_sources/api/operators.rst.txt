pyzag.operators
===============

The ``pyzag.operators`` subpackage defines the abstract block-operator
interfaces the solver runs against, plus the concrete dense backend. New in
pyzag 2.0.

Mathematical model
------------------

The solver works on *chunks* of a recursive nonlinear system. Within a chunk of
:math:`m` steps the unknowns are stacked into a **block vector**

.. math::

    x = (x_1, \dots, x_m), \qquad x_k \in \mathbb{R}^{n},

with logical shape ``(nblk, batch_size, block_size)`` -- axis 0 is the block /
time index, axis 1 is the batch, and the last axis holds one block's entries. A
backend is free to store this however it likes (dense tensor, per-group list,
sparse) as long as it honours that logical shape.

A **block operator** :math:`A` is a linear map defined purely by its *action* --
it need not materialize a matrix. The two required actions are the matrix-vector
product and its transpose,

.. math::

    y = A\,x, \qquad y = A^{\mathsf{T}} x,

exposed as :meth:`~pyzag.operators.base.BlockOperator.matvec` and
:meth:`~pyzag.operators.base.BlockOperator.t_matvec`. A
**solvable block operator** additionally solves the block system

.. math::

    A\,x = b \;\Longrightarrow\; x = A^{-1} b,

via :meth:`~pyzag.operators.base.SolvableBlockOperator.solve`.

For lookback 1 the chunk residual Jacobian is block **lower-bidiagonal**: with
diagonal blocks :math:`A_k = \partial R_k/\partial x_k` and subdiagonal blocks
:math:`B_k = \partial R_k/\partial x_{k-1}`,

.. math::

    (J\,x)_k = A_k\,x_k + B_{k-1}\,x_{k-1},

and all other blocks are zero. Newton solves :math:`J\,\Delta = R` each iteration.
This bidiagonal solve is provided two ways: **Thomas** (sequential forward
substitution) and **parallel cyclic reduction** (PCR), which repeatedly halves a
power-of-two window by eliminating the odd blocks using
:math:`A^{-1}`-products. The PCR working state is carried opaquely across levels
by a :class:`~pyzag.operators.base.PCRState`. The
:class:`~pyzag.operators.base.BlockJacobian` abstraction owns the per-chunk
Jacobian and knows how to build both the forward system (for Newton) and the
transposed adjoint system (for the reverse/adjoint sweep).

.. note::

   The concrete bidiagonal *solve* operators that consume these interfaces --
   ``BidiagonalForwardOperator`` and the Thomas / PCR / hybrid factorizations
   that implement the :math:`J^{-1}` action described above -- live in
   :doc:`chunktime`, since they are solver machinery rather than storage
   backends.

Abstract interfaces
-------------------

Every backend implements these ABCs from ``pyzag.operators.base``. The contract
is behavioral: a backend may use any storage as long as the logical block
ordering and shapes are preserved.

Block vectors
~~~~~~~~~~~~~~

.. autoclass:: pyzag.operators.base.BlockVector
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Block operators
~~~~~~~~~~~~~~~~

.. autoclass:: pyzag.operators.base.BlockOperator
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. autoclass:: pyzag.operators.base.SolvableBlockOperator
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. autoclass:: pyzag.operators.base.PCRState
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Block Jacobians
~~~~~~~~~~~~~~~

.. autoclass:: pyzag.operators.base.BlockJacobian
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Dense backend
-------------

The dense backend is the default implementation and reproduces the pyzag 1.x
behavior. Block vectors are stored as ``(nblk, batch, state)`` tensors and block
operators as ``(nblk, sbat, sblk, sblk)`` tensors. Each abstract interface above
has a dense counterpart below.

.. autoclass:: pyzag.operators.dense.DenseBlockVector
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. autoclass:: pyzag.operators.dense.DenseBlockOperator
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. autoclass:: pyzag.operators.dense.DensePCRState
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. autoclass:: pyzag.operators.dense.DenseBlockJacobian
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Dense helpers
~~~~~~~~~~~~~

.. autofunction:: pyzag.operators.dense.batch_lu_solve
