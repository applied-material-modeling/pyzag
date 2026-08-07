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

"""Gauss-Newton curvature for least-squares model calibration: two levers.

When calibrating a model against data by minimizing a least-squares loss
``L(theta) = 1/2 r(theta)^T W r(theta)`` (``r`` the residual, ``W`` a diagonal
weight), the raw gradient is badly scaled whenever the parameters have
heterogeneous magnitudes or sensitivities -- which slows first-order optimizers.
The usual fix, :mod:`pyzag.reparametrization`, rescales each parameter by a
hand-picked range ``(ub - lb)``; that works but demands prior knowledge of every
parameter's range.

This module offers data-driven alternatives that need **no ranges**, both built on
the Gauss-Newton curvature ``H = J^T W J`` (``J = dr/dtheta``) and both sharing one
estimator, :class:`CurvatureEstimator`. It never forms the dense Jacobian: ``H``
comes from a small **subsample** of residual rows, one reverse-mode sweep each.

**Lever 1 -- preconditioning** (:class:`GaussNewtonPreconditioner`), a wrapper
around a torch optimizer that reshapes the gradient every step::

    theta <- optimizer_update( (H + lam * diag(H))^{-1} @ grad )

It can refresh ``H`` as the fit moves, and at ``lr=1`` with SGD the update is
exactly the damped Gauss-Newton step, so there is nothing to tune. It requires an
optimizer whose step is proportional to its gradient -- see the note below.

**Lever 2 -- static reparametrization**
(:func:`gauss_newton_rescalers` + :class:`pyzag.reparametrization.CurvatureRescale`),
which estimates ``H`` **once** and changes coordinates by ``1 / sqrt(diag H)``::

    scalers = gauss_newton_rescalers(model.named_parameters(), residual_closure)
    Reparameterizer(scalers)(model)
    opt = torch.optim.Adam(model.parameters(), lr=...)   # any optimizer

Because it is a reparametrization the optimizer's own state moves into the scaled
coordinates with the metric, so it works with **any** optimizer, Adam included.
It cannot refresh, though: re-scaling mid-run would invalidate a stateful
optimizer's moments.

Which to reach for:

===========================  ==================================================
curvature over the fit       lever
===========================  ==================================================
stable                       **reparametrization** -- one estimate, any optimizer
drifts                       **preconditioning** -- refreshes, SGD-family only
===========================  ==================================================

Both are alternatives to a hand-picked ``RangeRescale``, and lever 2 additionally
*separates* the two jobs a range width does today: it takes the step scale from
the data, leaving ``lb`` / ``ub`` free to be honest bounds rather than a
compromise between bounding and conditioning.

Damping in lever 1 is Marquardt (``lam * diag(H)``), invariant to parameter scales.

The **gain ratio** -- observed loss reduction over the reduction the quadratic
model predicted -- drives two independent mechanisms, and it is worth keeping
them apart:

=============  ======================  ==========================  ===============
knob           question                action                      cost
=============  ======================  ==========================  ===============
``rho``        is the cached H stale?  recompute H                 ``nsub`` sweeps
``lam_adapt``  is the step too long?   adapt lam, reject bad step  free
=============  ======================  ==========================  ===============

``rho=None`` turns off refreshing (compute ``H`` once and reuse it) but leaves
damping active -- which matters *more* in that configuration, since refreshing is
no longer available as a corrective. Damping costs nothing extra: it reuses loss
values the training loop already has.

.. important::

   The base optimizer's step must be **proportional to the gradient it is
   given** -- SGD, with or without momentum. That proportionality is what makes
   the update the damped Gauss-Newton step.

   **Adam and its relatives do not qualify.** They divide each coordinate by its
   own running gradient RMS, so multiplying the gradient by a diagonal leaves
   their step exactly unchanged; wrapping one is a no-op, not a weaker effect.
   The constructor rejects them for that reason. To condition a problem you
   intend to optimize with Adam, change coordinates instead --
   :mod:`pyzag.reparametrization` composes with Adam precisely because a
   reparametrization also moves the optimizer's own state into the new space,
   which a preconditioner applied to the gradient cannot do.

Typical use (no parameter ranges required)::

    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    pre = GaussNewtonPreconditioner(opt, model.parameters(), rho=0.25, nsub=8)
    for _ in range(niter):
        loss = pre.step(lambda: model(time, temperature, loading) - data)

The closure returns the **residual vector** ``r(theta)`` (differentiable w.r.t. the
parameters); the preconditioner computes the gradient and curvature from it. At
``lr=1`` with SGD the update is exactly the damped Gauss-Newton step.

Forming ``H`` lives in :mod:`pyzag.curvature`
(:class:`~pyzag.curvature.CurvatureEstimator`), shared by both levers.
:class:`GaussNewtonCurvature` extends it with the caching, staleness and damping a
training loop needs. :class:`GaussNewtonPreconditioner` drives a plain torch loop;
:class:`pyzag.stochastic.PyroGaussNewtonOptim` drives Pyro SVI, where the training
loop belongs to somebody else.

Scope / notes:

* Physical **bounds are intentionally not handled here** -- a preconditioner only
  rescales the step direction. Enforce bounds separately (e.g. a projection after
  ``step`` or a bounds-only :mod:`pyzag.reparametrization`).
* Non-finite residuals or preconditioned gradients (which stiff models can
  produce) cause the update to be **skipped with a warning**, never silently.
* For ``mode="full"`` use ``nsub`` (or ``len(sample_indices)``) at least the number
  of parameters, otherwise the sampled ``H`` is rank deficient.
"""

import copy
import math
import warnings

import torch

from pyzag.curvature import CurvatureEstimator
from pyzag.reparametrization import CurvatureRescale


def _snapshot_optimizer(optimizer):
    """Deep copy of an optimizer's state, so a rejected step can be undone."""
    return copy.deepcopy(optimizer.state_dict())


class GaussNewtonCurvature(
    CurvatureEstimator
):  # pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-public-methods
    """A :class:`CurvatureEstimator` plus the state a training loop needs.

    Caches ``H``, judges when it has gone stale (:meth:`should_refresh`), and
    runs the Levenberg-Marquardt damping schedule (:meth:`adapt_damping`). A
    driver calls :meth:`begin_step`, :meth:`gain_ratio`, :meth:`adapt_damping`,
    :meth:`should_refresh`, :meth:`refresh`, :meth:`precondition` and
    :meth:`note_step` in that order.

    Keyword Args:
        rho (float or None): gain-ratio threshold; ``None`` never refreshes after
            the first step (a fixed preconditioner).
        lam (float): initial Marquardt damping factor.
        lam_adapt (bool): adapt ``lam`` from the gain ratio -- grow it when a step
            underdelivers, shrink it when a step delivers, reject a step that
            increased the loss. See :meth:`adapt_damping`.
        min_refresh_interval (int): steps that must reuse the cached curvature
            before it may be refreshed again. The default 1 allows a refresh at
            most every other step; pass 0 to allow one on every step.
        on_refresh (callable, optional): called on every refresh as
            ``on_refresh(step=..., gain_ratio=..., n_refresh=...)``.

    Every other keyword is forwarded to :class:`CurvatureEstimator`.
    """

    # Standard Levenberg-Marquardt damping schedule. A rejected step damps by
    # _LAM_UP**2 so that backing out is decisively more conservative than merely
    # underdelivering.
    _LAM_UP = 3.0
    _LAM_DOWN = 3.0
    _LAM_MIN = 1e-12
    _LAM_MAX = 1e8

    def __init__(
        self,
        parameters,
        *,
        rho=0.25,
        lam=1e-2,
        lam_adapt=True,
        min_refresh_interval=1,
        on_refresh=None,
        nsub=8,
        **estimator_kwargs,
    ):
        # A cheap subsample, unlike the exact CurvatureEstimator default, and the
        # asymmetry is earned rather than assumed. `precondition` divides by
        # `clamp(H) + lam*max|H|`, so a near-zero entry still yields a bounded
        # step -- the damping absorbs a bad estimate, and the next refresh
        # replaces it. A frozen reparametrization has neither defence: it divides
        # by sqrt(H) directly and lives with the result for the whole fit.
        # Measured on the NEML2 calibration, nsub=8 produces an unusable estimate
        # on 5 draws in 12; that is fine here and fatal there.
        super().__init__(parameters, nsub=nsub, **estimator_kwargs)
        self.rho = None if rho is None else float(rho)
        self.lam = float(lam)
        self.lam0 = float(lam)
        self.lam_adapt = bool(lam_adapt)
        self.min_refresh_interval = int(min_refresh_interval)
        self.on_refresh = on_refresh

        # State carried across steps.
        self._H = None  # cached curvature (vector if diag, matrix if full)
        self._prev = None  # {"g", "dtheta", "H", "loss"} of the last accepted step
        # The last point the objective was successfully evaluated at, kept apart
        # from `_prev` because the two have different lifetimes. Backing a step
        # out invalidates the *quadratic model* (`g`/`dtheta`/`H` describe a step
        # that no longer happened) but not the *fallback point* -- which the undo
        # has just restored the parameters to, making it the best-established
        # point available. Clearing both together left a rollback with nothing
        # behind it, so a second failure in a row had nowhere to go.
        self._good = None  # {"theta", "loss", "opt_state"}
        self._since_refresh = 0
        self._force_refresh = False
        self._t = -1  # 0-based index of the current step

        # Diagnostics.
        self.n_steps = 0
        self.n_refreshes = 0
        self.n_refresh_sweeps = 0
        self.n_rejected = 0
        self.n_failed = 0
        self.lam_history = []
        self.refresh_steps = []  # step indices at which the curvature was refreshed

    @property
    def H(self):
        """The cached curvature: a length-``p`` vector (diag) or ``p x p`` matrix
        (full), or ``None`` before the first refresh."""
        return self._H

    def refresh(self, residual, w_flat=None):
        """Re-estimate ``H`` from ``residual`` and cache it.

        ``w_flat`` is accepted for backwards compatibility and ignored -- the
        weights come from ``weights``.
        """
        del w_flat  # recomputed from self.weights; kept for call-signature compat
        self._H, nsweeps = self.estimate(residual)
        self.n_refreshes += 1
        self.n_refresh_sweeps += nsweeps

    def set_H(self, H, sweeps=0):
        """Install a curvature computed elsewhere -- e.g. a structure-aware
        estimator that assembles ``H`` from several blocks, each estimated by the
        method that suits it -- counting it as a refresh.

        Args:
            H (Tensor): length-``p`` vector in ``"diag"`` mode, ``p x p`` matrix in
                ``"full"`` mode.
            sweeps (int): number of reverse-mode sweeps it cost, for accounting.
        """
        expected = (self._p,) if self.mode == "diag" else (self._p, self._p)
        if tuple(H.shape) != expected:
            raise ValueError(
                f"H has shape {tuple(H.shape)}, expected {expected} for mode={self.mode!r}"
            )
        self._H = H
        self.n_refreshes += 1
        self.n_refresh_sweeps += int(sweeps)

    def precondition(self, g):
        """Return (H + lam * diag(H))^{-1} g, Marquardt-damped and scale-invariant."""
        H = self._H
        assert H is not None  # always refreshed before the first step
        floor = 1e-12
        if self.mode == "diag":
            scale = torch.clamp(H.abs().max(), min=floor)
            return g / (torch.clamp(H, min=floor) + self.lam * scale)
        d = torch.clamp(torch.diagonal(H), min=floor)
        eye = torch.eye(self._p, device=self._device, dtype=self._dtype)
        A = H + self.lam * torch.diag(d)
        ridge = floor * torch.clamp(d.max(), min=floor)
        for _ in range(8):
            try:
                L = torch.linalg.cholesky(A + ridge * eye)
                return torch.cholesky_solve(g.unsqueeze(-1), L).squeeze(-1)
            except RuntimeError:
                ridge *= 10.0
        return g / (
            torch.clamp(torch.diagonal(H), min=floor) + self.lam
        )  # diagonal fallback

    def _quad(self, H, d):
        """``H @ d``, dispatched on mode."""
        return H * d if self.mode == "diag" else H @ d

    def begin_step(self):
        """Advance the step counter. Returns the new 0-based step index."""
        self._t += 1
        return self._t

    def reset_anchor(self):
        """Drop the cached quadratic model (its anchor point is no longer valid).

        Deliberately leaves ``_good`` alone: the last *evaluable* point stays
        valid even when the quadratic model built at it does not.
        """
        self._prev = None

    def request_refresh(self):
        """Force a refresh on the next step."""
        self._force_refresh = True

    def gain_ratio(self, loss_v):
        """Gain ratio of the PREVIOUS step: observed vs. predicted loss reduction.

        Returns ``None`` when there is no anchored quadratic model, or when the
        prediction is too small to divide by. Only meaningful when the previous
        step's quadratic model predicted a non-negligible change; at/near
        convergence pred_red -> 0 and the ratio is 0/0 noise, which must not be
        mistaken for staleness.
        """
        if self._prev is None:
            return None
        actual_red = self._prev["loss"] - loss_v
        Hd = self._quad(self._prev["H"], self._prev["dtheta"])
        pred_red = -(
            torch.dot(self._prev["g"], self._prev["dtheta"])
            + 0.5 * torch.dot(self._prev["dtheta"], Hd)
        ).item()
        eps = 1e-10 * max(abs(self._prev["loss"]), 1.0)
        if pred_red > eps:
            return actual_red / pred_red
        if pred_red < -eps:
            return -1.0  # model predicted an increase -> curvature stale
        return None

    def adapt_damping(self, loss_v, gain_ratio):
        """Levenberg-Marquardt damping update, driven by the gain ratio.

        Gauss-Newton drops the ``sum(r_i * grad^2 r_i)`` term of the true
        Hessian, and that term is not always small. For a Gaussian scale
        parameter in unconstrained coordinates (``sigma = exp(u)``) the dropped
        term is ``r^2`` -- exactly as large as the Gauss-Newton term ``r^2``
        itself -- so the undamped step is a systematic factor-of-two overshoot in
        every scale coordinate. Damping is what absorbs that: on a step whose
        observed reduction falls short of the quadratic model's prediction,
        ``lam`` grows and the step shrinks towards gradient descent; on a step
        that delivers, ``lam`` decays back towards the full Newton step.

        Rejection keys off the **observed** loss increase, not off
        ``gain_ratio``. The two are not interchangeable: :meth:`gain_ratio`
        returns a ``-1.0`` sentinel when the *quadratic model* predicted an
        increase, which says the curvature is stale, not that the step was bad --
        the actual loss may well have fallen. Reading that sentinel as a
        regression would back out perfectly good steps.

        Args:
            loss_v (float): objective at the current parameters.
            gain_ratio (float or None): from :meth:`gain_ratio`.

        Returns:
            bool: ``True`` if the previous step should be **rejected** -- the
            loss went up, so the caller should call :meth:`undo_last_step`.
        """
        # A non-finite objective is not a comparison, it is the absence of one:
        # every `nan > tol` below is False, so without this branch a NaN reads as
        # an *accepted* step. It then lands in `_prev["loss"]`, after which
        # `nan - nan > tol` is False forever -- the damping freezes and the run
        # spins at a dead point for the rest of its iterations, reporting a stale
        # loss. Treat it as the strongest evidence of an over-long step, exactly
        # as `reject_failed_evaluation` treats a raised one. Rejection here is a
        # correctness matter, not an adaptation preference, so it fires even with
        # `lam_adapt` off -- only the `lam` bump is conditional. It gates on
        # `_good`, not `_prev`: a rollback clears the anchor, and a NaN arriving
        # on the step straight after one must still be able to rewind.
        if not math.isfinite(loss_v):
            if self._good is None or self._good["theta"] is None:
                return False
            if self.lam_adapt:
                self.lam = min(self.lam * self._LAM_UP**3, self._LAM_MAX)
            self.n_rejected += 1
            return True
        if not self.lam_adapt or self._prev is None:
            return False
        # Reject on a *meaningful* increase only. At convergence the loss is
        # flat to within roundoff, and a bare `loss_v > prev` then backs out
        # good steps forever on floating-point noise.
        tol = 1e-10 * max(abs(self._prev["loss"]), 1.0)
        if loss_v - self._prev["loss"] > tol:
            # The step made things worse: back it out and damp harder.
            self.lam = min(self.lam * self._LAM_UP**2, self._LAM_MAX)
            self.n_rejected += 1
            return True
        if gain_ratio is None:
            return False
        if gain_ratio < 0.25:
            self.lam = min(self.lam * self._LAM_UP, self._LAM_MAX)
        elif gain_ratio > 0.75:
            self.lam = max(self.lam / self._LAM_DOWN, self._LAM_MIN)
        return False

    def undo_last_step(self, optimizer):
        """Restore the parameters *and* the optimizer state to before the last step.

        Restoring the parameters alone is not enough: a stateful optimizer's
        momentum buffer still holds the rejected direction and would immediately
        re-apply it, so the same step is proposed, rejected, and re-proposed
        forever while ``lam`` ratchets to its ceiling. The optimizer state has to
        rewind with the parameters.
        """
        # Rewind to the last *evaluable* point rather than to `_prev`. The two
        # agree on the common path, but `_prev` is cleared by every rollback
        # while `_good` survives one, which is what lets a second consecutive
        # failure still find somewhere to land.
        good = self._good
        if good is None:
            return
        if good["theta"] is not None:
            self._write_params(good["theta"])
        if good["opt_state"] is not None:
            optimizer.load_state_dict(good["opt_state"])
        self.reset_anchor()

    def reject_failed_evaluation(self, optimizer):
        """Back out the last step after the model could not be evaluated at all.

        :meth:`adapt_damping` compares two loss values, so it cannot react to a
        point where the objective does not exist -- a stiff forward model driven
        outside its valid region raises instead of returning a number, and the
        exception escapes before any comparison is possible. That is simply the
        limiting case of a step that was too long, so it is handled the same way,
        with a harder damping bump because the evidence is stronger.

        Returns:
            bool: ``True`` if a good point was available to fall back to. When
            ``False`` there is nothing to undo -- the very first evaluation
            failed -- and the caller should let the error propagate rather than
            pretend to recover.
        """
        if self._good is None or self._good["theta"] is None:
            return False
        self.lam = min(self.lam * self._LAM_UP**3, self._LAM_MAX)
        self.n_failed += 1
        self.undo_last_step(optimizer)
        return True

    def should_refresh(self, gain_ratio):
        """Whether the curvature must be (re)computed on this step. With
        ``rho=None`` it is computed once and never refreshed."""
        if self._H is None or self._force_refresh:
            return True
        if (
            self.rho is not None
            and gain_ratio is not None
            and self._since_refresh >= self.min_refresh_interval
        ):
            return gain_ratio < self.rho
        return False

    def note_refresh(self, gain_ratio, refreshed):
        """Record this step's refresh decision.

        Takes the decision rather than being one of a pair of methods, so a
        caller cannot handle one branch and forget the other.
        """
        if not refreshed:
            self._since_refresh += 1
            return
        self._since_refresh = 0
        self._force_refresh = False
        self.refresh_steps.append(self._t)
        if self.on_refresh is not None:
            self.on_refresh(
                step=self._t, gain_ratio=gain_ratio, n_refresh=self.n_refreshes
            )

    def note_step(self, g, dtheta, loss_v, theta=None, opt_state=None):
        """Anchor the quadratic model at the step that just completed.

        ``dtheta`` must be the **actual** parameter displacement the optimizer
        produced, not the proposed direction, so the gain ratio accounts for the
        base optimizer's learning rate and momentum.
        """
        self._prev = {
            "g": g,
            "dtheta": dtheta,
            "H": self._H,
            "loss": loss_v,
            "theta": theta,
            "opt_state": opt_state,
        }
        # `loss_v` was measured *at* `theta`, so this records a point known to be
        # evaluable. It outlives `_prev` deliberately: see `_good` in __init__.
        if theta is not None and math.isfinite(loss_v):
            self._good = {"theta": theta, "loss": loss_v, "opt_state": opt_state}
        self.lam_history.append(self.lam)
        self.n_steps += 1


# Optimizers whose per-coordinate step does not scale with the gradient they are
# given. They divide each coordinate by its own running gradient RMS, so a
# diagonal preconditioner cancels out exactly and this API silently does nothing.
_SCALE_INVARIANT = (
    "Adam",
    "AdamW",
    "NAdam",
    "RAdam",
    "Adamax",
    "Adagrad",
    "Adadelta",
    "RMSprop",
    "ClippedAdam",
    "Lion",
    "Adafactor",
)


def gauss_newton_rescalers(  # pylint: disable=too-many-locals
    named_parameters,
    residual_closure,
    *,
    normalize=True,
    cond_max=1e12,
    bounds=None,
    offsets=None,
    **estimator_kwargs,
):
    """Build a static, curvature-derived reparametrization -- the second lever.

    Estimates ``H`` **once** at the current parameter values and returns one
    :class:`~pyzag.reparametrization.CurvatureRescale` per parameter, scaled by
    ``1 / sqrt(diag H)``. Feed the result straight to
    :class:`~pyzag.reparametrization.Reparameterizer`::

        scalers = gauss_newton_rescalers(model.named_parameters(), residual_closure)
        Reparameterizer(scalers)(model)
        opt = torch.optim.Adam(model.parameters(), lr=...)   # any optimizer

    Use this rather than :class:`GaussNewtonPreconditioner` when you want to keep
    an Adam-family optimizer (which is invariant to gradient preconditioning), or
    when the curvature is stable enough that one estimate serves the whole fit.
    Use the preconditioner instead when the curvature drifts, since only it can
    refresh. Cost here is a single estimate -- ``nsub`` reverse sweeps -- and
    nothing per step.

    Args:
        named_parameters: ``(name, parameter)`` pairs, or a dict. Names must be
            the dotted paths ``Reparameterizer`` matches on.
        residual_closure (callable): returns the residual at the current
            parameters, differentiable w.r.t. them.

    Keyword Args:
        cond_max (float): reject the estimate if ``max(H) / min(H)`` exceeds this.
            An under-sampled row set can leave a parameter's curvature many orders
            of magnitude below the rest, and ``1 / sqrt(H)`` then hands that
            parameter an essentially infinite step. The damage is invisible in
            ``H`` itself -- on this calibration ``nsub=8`` gives an ``H`` 17% off
            whose implied scale is wrong by twelve orders of magnitude -- so the
            check has to be on the conditioning, not on the norm. Raise it only if
            your parameters genuinely differ that much in sensitivity.
        normalize (bool): divide the scales by their geometric mean, so the
            learning rate sets the *average* physical step and the curvature sets
            only how that step is **distributed** across parameters. On by
            default, and you almost always want it: ``1 / sqrt(H)`` carries units
            of ``theta / residual``, so its absolute level is a property of the
            residual's units rather than of the parameters. Measured on the NEML2
            calibration it sits ~920x below the hand-picked range widths, which
            would freeze an optimizer tuned for those. Normalizing makes the
            learning rate mean the same thing it does for any other coordinate
            choice; the curvature still supplies the whole relative profile.
        bounds (dict, optional): ``name -> (lb, ub)`` in natural units. Bounds are
            *only* bounds here -- they do not influence the step scale, so they
            can be generous. Omitting them means no clamping at all, which is a
            real behavioural change if you are replacing a ``RangeRescale``.
        offsets (dict, optional): ``name -> offset``, the natural value at scaled
            zero. Defaults to 0.
        **estimator_kwargs: forwarded to :class:`CurvatureEstimator` (``nsub``,
            ``sample_indices``, ``cotangents``, ``weights``, ``generator``).

    Returns:
        dict: ``name -> CurvatureRescale``.

    Raises:
        ValueError: if any parameter has non-positive curvature. That means the
            residual does not depend on it (or not detectably), so there is no
            scale to derive and ``1 / sqrt(H)`` is not defined -- silently
            flooring it would hand back an arbitrary scale dressed up as a
            measurement.
    """
    if isinstance(named_parameters, dict):
        named_parameters = named_parameters.items()
    names, params = zip(*named_parameters)

    estimator = CurvatureEstimator(params, mode="diag", **estimator_kwargs)
    H, _ = estimator.estimate(residual_closure())
    # Report before normalizing -- a log of a non-positive entry is nan and would
    # poison every other scale.
    _check_curvature_usable(names, params, H, cond_max)
    scales = torch.rsqrt(H.detach())
    if normalize:
        scales = scales / torch.exp(torch.log(scales).mean())

    bounds = bounds or {}
    offsets = offsets or {}
    scalers, off = {}, 0
    for name, param in zip(names, params):
        block = scales[off : off + param.numel()].reshape(param.shape)
        off += param.numel()
        lb, ub = bounds.get(name, (None, None))
        scalers[name] = CurvatureRescale(
            block, offset=offsets.get(name, 0.0), lb=lb, ub=ub
        )
    return scalers


def _check_curvature_usable(names, params, H, cond_max):
    """Refuse an estimate whose implied scale would be meaningless.

    Two ways that happens, both fatal for a *frozen* reparametrization and both
    invisible if you only look at ``||H||``:

    * an entry is non-positive -- the residual does not constrain it at all;
    * an entry is positive but so far below the rest that ``1/sqrt(H)`` gives it
      a runaway step. That is what an under-sampled row set produces, and unlike
      the first case it does not announce itself.
    """

    def blocks():
        off = 0
        for name, param in zip(names, params):
            yield name, param, H[off : off + param.numel()]
            off += param.numel()

    dead = [
        f"{n} ({int((b <= 0).sum())}/{p.numel()} entries)"
        for n, p, b in blocks()
        if bool((b <= 0).any())
    ]
    if dead:
        raise ValueError(
            "non-positive Gauss-Newton curvature for: "
            + "; ".join(dead)
            + ". The residual does not depend on those entries, so 1/sqrt(H) has "
            "no meaning for them. Either the estimate is under-sampled (raise "
            "nsub, or leave it at None for an exact estimate), or those "
            "parameters genuinely do not affect the residual and should be "
            "dropped from the calibration."
        )

    cond = float(H.max() / H.min())
    if cond > cond_max:
        worst = min(blocks(), key=lambda t: float(t[2].min()))[0]
        raise ValueError(
            f"Gauss-Newton curvature spans {cond:.3g} (limit cond_max={cond_max:.3g}), "
            f"smallest at {worst!r}. 1/sqrt(H) would give that parameter a step "
            f"{math.sqrt(cond):.3g}x the largest, which is far more likely to be an "
            "under-sampled estimate than real physics: raise nsub, or leave it at "
            "None for an exact estimate. If the spread is genuine, raise cond_max."
        )


def check_optimizer_is_memoryless(optimizer):
    """Warn if ``optimizer`` carries state that damping cannot reach.

    Levenberg-Marquardt damping assumes the step is a function of the *current*
    damped gradient: raise ``lam``, get a shorter step. A momentum buffer breaks
    that -- it contributes its own accumulated velocity, so a step that
    overshoots keeps overshooting no matter how hard ``lam`` is raised. In
    practice the run enters a reject/restore cycle and ``lam`` ratchets to its
    ceiling while the loss stalls. Restoring the optimizer state alongside the
    parameters does not help, because the restored state is what holds the
    offending velocity.
    """
    for group in getattr(optimizer, "param_groups", []):
        if group.get("momentum", 0) or group.get("dampening", 0):
            warnings.warn(
                f"{type(optimizer).__name__} has momentum, which Levenberg-Marquardt "
                "damping cannot shorten: raising lam shrinks the gradient term but "
                "not the accumulated velocity, so rejected steps repeat and lam "
                "ratchets to its ceiling. Use momentum=0, or lam_adapt=False.",
                stacklevel=3,
            )
            return


def check_optimizer_respects_gradient_scale(optimizer):
    """Raise if ``optimizer`` would annihilate the preconditioner.

    Gauss-Newton preconditioning replaces the gradient with ``(H + lam)^-1 g``
    and relies on the optimizer's step being **proportional** to what it is
    handed -- that is what makes the update the damped Gauss-Newton step. Adam
    and its relatives normalize each coordinate by its own running gradient RMS,
    so multiplying the gradient by any fixed diagonal leaves their step exactly
    unchanged: wrapping one is a no-op, not a weaker effect.

    That failure is silent and easy to mistake for a working configuration, so
    it is rejected outright rather than warned about. Use SGD (optionally with
    momentum), or reach for :mod:`pyzag.reparametrization`, which conditions the
    problem by changing coordinates and therefore composes with Adam.
    """
    name = type(optimizer).__name__
    if name in _SCALE_INVARIANT:
        raise TypeError(
            f"{name} normalizes each coordinate by its own gradient RMS, so it is "
            "invariant to Gauss-Newton preconditioning -- wrapping it would have "
            "no effect at all. Use an optimizer whose step is proportional to the "
            "gradient (e.g. torch.optim.SGD), or condition the problem by "
            "reparametrizing instead (pyzag.reparametrization), which does "
            "compose with Adam. Pass check_optimizer=False to override."
        )


def apply_preconditioned_update(
    curvature, g, optimizer, loss_v=None, theta_before=None
):
    """Precondition ``g``, install it as ``.grad``, and let ``optimizer`` step.

    The tail every driver shares, whoever owns the training loop. The optimizer
    is handed ``(H + lam)^-1 g`` -- the damped Gauss-Newton direction -- and its
    own rule (learning rate, momentum) scales it. The gradient is **overwritten**,
    not accumulated.

    This requires an optimizer whose step is proportional to the gradient it is
    given; see :func:`check_optimizer_respects_gradient_scale`.

    Args:
        curvature (GaussNewtonCurvature): the engine holding the parameters.
        g (Tensor): the raw packed gradient.
        optimizer (torch.optim.Optimizer): applies the update.
        loss_v (float, optional): objective at the current parameters. When
            given, the quadratic model is anchored here for the next step's
            gain-ratio test; when ``None`` the trigger stays inert.
        theta_before (Tensor, optional): parameter vector before this step,
            recorded so a step that turns out to increase the loss can be backed
            out. Read from the parameters when omitted.

    Returns:
        bool: whether the update was applied. ``False`` means the curvature
        produced a non-finite direction, so the parameters were left untouched
        and a refresh is forced for the next step.
    """
    pg = curvature.precondition(g)
    if not bool(torch.isfinite(pg).all()):
        warnings.warn(
            "non-finite preconditioned gradient (ill-conditioned or stale "
            "curvature); skipping this update and forcing a refresh next step.",
            stacklevel=3,
        )
        curvature.request_refresh()
        return False

    if theta_before is None:
        theta_before = curvature.theta()
    # Only pay for the snapshot when a step can actually be rejected.
    opt_state = _snapshot_optimizer(optimizer) if curvature.lam_adapt else None
    curvature.write_grads(pg)
    optimizer.step()
    if loss_v is not None:
        curvature.note_step(
            g,
            curvature.theta() - theta_before,
            loss_v,
            theta=theta_before,
            opt_state=opt_state,
        )
    return True


class GaussNewtonPreconditioner:  # pylint: disable=too-many-arguments
    """Wrap a torch optimizer and precondition its gradient with the Gauss-Newton
    curvature, refreshed on a gain-ratio trigger.

    Args:
        optimizer (torch.optim.Optimizer): the base optimizer to wrap. Its
            ``param_groups`` must cover exactly ``parameters``.
        parameters (iterable of Parameter): the parameters being calibrated.

    Keyword Args:
        mode (str): ``"diag"`` (default) preconditions with the diagonal of ``H``;
            ``"full"`` uses the full ``p x p`` matrix.
        nsub (int): number of residual rows to subsample when estimating ``H``
            (one reverse-mode sweep each). Ignored if ``sample_indices`` is given.
        sample_indices (array-like of int, optional): explicit rows of the
            flattened residual to sample (overrides ``nsub``). Use this to sample
            *representatively* across conditions; a poorly chosen stride can alias
            with batch structure and starve some parameters, so when in doubt leave
            it ``None`` and let the estimator draw ``nsub`` rows at random.
        cotangents (callable or sequence, optional): custom cotangent vectors,
            overriding ``nsub`` and ``sample_indices``. Each costs one sweep and
            may select a *group* of rows -- valid only when those rows have
            disjoint parameter support. See :meth:`GaussNewtonCurvature.refresh`.
        rho (float or None): gain-ratio threshold. ``H`` is refreshed when the
            observed loss reduction of the previous step is below ``rho`` times the
            value the cached curvature predicted -- i.e. when the curvature model
            has gone stale. Larger ``rho`` refreshes more eagerly. Set to ``None``
            to never refresh after the first step: a **fixed preconditioner**, the
            cheapest option and often sufficient when the curvature profile is
            stable (e.g. a diagonal), where it acts as a data-driven analogue of a
            static per-parameter rescaling.
        lam (float): initial Marquardt damping factor; the solve uses
            ``H + lam * diag(H)``.
        lam_adapt (bool): Levenberg-Marquardt damping adaptation -- see
            :meth:`GaussNewtonCurvature.adapt_damping`. On by default because the
            second-order term Gauss-Newton drops is not always small.
        weights (Tensor, optional): diagonal of ``W``, broadcastable to the residual
            shape (e.g. inverse variances). ``None`` means unweighted (``W = I``).
        min_refresh_interval (int): number of steps that must *reuse* the cached
            curvature before it may be refreshed again -- a hard floor on refresh
            cost. The default 1 therefore allows a refresh at most every other
            step; pass 0 to allow one on every step.
        generator (torch.Generator, optional): RNG for reproducible subsampling.
        on_refresh (callable, optional): called whenever the curvature is
            refreshed, as ``on_refresh(step=<0-based step index>,
            gain_ratio=<float or None>, n_refresh=<count>)``. The refreshed step
            indices are also recorded on ``self.refresh_steps`` (useful for
            annotating plots with the recompute iterations).
        check_optimizer (bool): reject a base optimizer that is invariant to
            gradient preconditioning (Adam and relatives), which would make the
            whole wrapper a silent no-op. See
            :func:`check_optimizer_respects_gradient_scale`.
    """

    def __init__(
        self,
        optimizer,
        parameters,
        *,
        mode="diag",
        nsub=8,
        sample_indices=None,
        cotangents=None,
        rho=0.25,
        lam=1e-2,
        lam_adapt=True,
        weights=None,
        min_refresh_interval=1,
        generator=None,
        on_refresh=None,
        check_optimizer=True,
    ):
        if check_optimizer:
            check_optimizer_respects_gradient_scale(optimizer)
            if lam_adapt:
                check_optimizer_is_memoryless(optimizer)
        self.optimizer = optimizer
        self.curvature = GaussNewtonCurvature(
            parameters,
            mode=mode,
            nsub=nsub,
            sample_indices=sample_indices,
            cotangents=cotangents,
            rho=rho,
            lam=lam,
            lam_adapt=lam_adapt,
            weights=weights,
            min_refresh_interval=min_refresh_interval,
            generator=generator,
            on_refresh=on_refresh,
        )

    # ---- delegation to the curvature engine ---------------------------------
    # The engine owns the parameter list, the configuration and the diagnostics;
    # these forwarders keep the driver's original public surface intact.
    @property
    def parameters(self):
        """The parameters being calibrated."""
        return self.curvature.parameters

    @property
    def mode(self):
        """``"diag"`` or ``"full"``."""
        return self.curvature.mode

    @property
    def lam(self):
        """Marquardt damping factor."""
        return self.curvature.lam

    @property
    def rho(self):
        """Gain-ratio refresh threshold (``None`` = fixed preconditioner)."""
        return self.curvature.rho

    @property
    def n_steps(self):
        """Number of completed updates."""
        return self.curvature.n_steps

    @property
    def n_refreshes(self):
        """Number of curvature refreshes."""
        return self.curvature.n_refreshes

    @property
    def n_refresh_sweeps(self):
        """Total reverse-mode sweeps spent refreshing the curvature."""
        return self.curvature.n_refresh_sweeps

    @property
    def refresh_steps(self):
        """Step indices at which the curvature was refreshed."""
        return self.curvature.refresh_steps

    @property
    def _H(self):
        return self.curvature.H

    def _refresh(self, residual, w_flat=None):
        return self.curvature.refresh(residual, w_flat)

    # ---- the wrapped step ---------------------------------------------------
    def step(self, residual_closure):
        """Evaluate the residual, precondition the gradient, and step the wrapped
        optimizer. Returns the scalar loss (float) at the current parameters, or
        ``nan`` if the residual was non-finite and the update was skipped.

        Args:
            residual_closure (callable): returns the residual ``r(theta)`` as a
                tensor that is differentiable w.r.t. ``parameters``.
        """
        gn = self.curvature
        gn.begin_step()
        try:
            residual = residual_closure()
        except (ValueError, RuntimeError) as err:
            if not gn.reject_failed_evaluation(self.optimizer):
                raise
            warnings.warn(
                f"the residual could not be evaluated ({type(err).__name__}: {err}); "
                "backing out the last step and increasing the damping.",
                stacklevel=2,
            )
            return float("nan")
        w_flat = gn.weights_flat(residual)
        r_flat = residual.reshape(-1)
        loss = 0.5 * torch.sum(w_flat * r_flat * r_flat)

        if not torch.isfinite(loss):
            # Same evidence as a raised evaluation -- a finite point existed one
            # step ago and the step from it was too long -- so the same response.
            # Merely resetting the anchor leaves the parameters *at* the dead
            # point, where every subsequent step recomputes NaN and skips again:
            # the run burns its whole iteration budget without moving.
            if not gn.reject_failed_evaluation(self.optimizer):
                raise ValueError(
                    "non-finite residual at the initial parameters, with no "
                    "previous point to fall back to; check the starting values."
                )
            warnings.warn(
                "non-finite residual at the current parameters; backing out the "
                "last step and increasing the damping.",
                stacklevel=2,
            )
            return float("nan")
        loss_v = float(loss.detach())

        gain_ratio = gn.gain_ratio(loss_v)
        if gn.adapt_damping(loss_v, gain_ratio):
            # The previous step increased the loss. Back it out, keep the larger
            # damping adapt_damping just set, and re-try from the good point.
            gn.undo_last_step(self.optimizer)
            return loss_v

        refreshed = gn.should_refresh(gain_ratio)
        if refreshed:
            gn.refresh(residual)
        gn.note_refresh(gain_ratio, refreshed)

        # Gradient g = J^T (W r) (one sweep), then the preconditioned update.
        theta_before = gn.theta()
        g = gn.vjp(residual, (w_flat * r_flat).reshape(residual.shape))
        apply_preconditioned_update(gn, g, self.optimizer, loss_v, theta_before)
        return loss_v

    def zero_grad(self, *args, **kwargs):
        """Delegate to the wrapped optimizer (provided for drop-in familiarity)."""
        self.optimizer.zero_grad(*args, **kwargs)
