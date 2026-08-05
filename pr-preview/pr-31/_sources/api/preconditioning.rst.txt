pyzag.preconditioning
=====================

Two ways to use the Gauss-Newton curvature of a calibration residual, sharing one
estimator. Pick by whether the curvature drifts over the fit:

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - lever
     - what it does
     - optimizers
     - refreshes?
   * - :class:`GaussNewtonPreconditioner`
     - reshapes the gradient each step
     - SGD-family only
     - yes
   * - :func:`gauss_newton_rescalers`
     - changes coordinates once
     - **any**, incl. Adam
     - no

Adam and its relatives divide each coordinate by its own running gradient RMS, so
they are *exactly invariant* to gradient preconditioning -- the preconditioner
rejects them rather than silently doing nothing. Use the reparametrization lever
with those.

.. automodule:: pyzag.preconditioning
    :members:

The scaler
----------

:func:`gauss_newton_rescalers` returns
:class:`pyzag.reparametrization.CurvatureRescale` objects, installed with the
existing :class:`pyzag.reparametrization.Reparameterizer`. See :doc:`reparametrization`.

Preconditioning an SVI fit
--------------------------

The Pyro-facing half lives in :doc:`stochastic`, because those classes subclass
Pyro types and this module is deliberately importable without Pyro. Use
:class:`pyzag.stochastic.PyroGaussNewtonOptim` with
:class:`pyzag.stochastic.PreconditionedSVI`, and build the residual with
:func:`pyzag.stochastic.gaussian_map_residual`.
