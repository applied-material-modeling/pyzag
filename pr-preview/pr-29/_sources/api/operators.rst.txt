pyzag.operators
===============

The ``pyzag.operators`` subpackage defines the abstract block-operator
interfaces the solver runs against, plus the concrete dense backend. New in
pyzag 2.0.

Abstract interfaces
-------------------

.. automodule:: pyzag.operators.base
    :members:

Dense backend
-------------

The dense backend is the default implementation and reproduces the pyzag 1.x
behavior. Block vectors are stored as ``(nblk, batch, state)`` tensors and block
operators as ``(nblk, sbat, sblk, sblk)`` tensors.

.. automodule:: pyzag.operators.dense
    :members:
