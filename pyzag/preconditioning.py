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

"""Gauss-Newton gradient preconditioning for least-squares model calibration.

When calibrating a model against data by minimizing a least-squares loss
``L(theta) = 1/2 r(theta)^T W r(theta)`` (``r`` the residual, ``W`` a diagonal
weight), the raw gradient is badly scaled whenever the parameters have
heterogeneous magnitudes or sensitivities -- which slows first-order optimizers.
The usual fix, :mod:`pyzag.reparametrization`, rescales each parameter by a
hand-picked range ``(ub - lb)``; that works but demands prior knowledge of every
parameter's range.

This module offers a data-driven alternative that needs **no ranges**: precondition
the gradient with the Gauss-Newton curvature ``H = J^T W J`` (``J = dr/dtheta``),

    theta <- optimizer_update( (H + lam * diag(H))^{-1} @ grad ).

:class:`GaussNewtonPreconditioner` is a thin **wrapper around an ordinary torch
optimizer** -- it only reshapes the gradient, so the base optimizer (Adam, SGD,
...) is fully swappable. It never forms the dense Jacobian: ``H`` is estimated from
a small **subsample** of residual rows (each row is one reverse-mode sweep), and it
is refreshed only when a free **gain-ratio** test says the cached curvature has
gone stale. Damping is Marquardt (``lam * diag(H)``), which is invariant to the
parameter scales.

Typical use (any optimizer, no parameter ranges)::

    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    pre = GaussNewtonPreconditioner(opt, model.parameters(), rho=0.25, nsub=8)
    for _ in range(niter):
        loss = pre.step(lambda: model(time, temperature, loading) - data)

The closure returns the **residual vector** ``r(theta)`` (differentiable w.r.t. the
parameters); the preconditioner computes the gradient and curvature from it.

Scope / notes:

* Physical **bounds are intentionally not handled here** -- a preconditioner only
  rescales the step direction. Enforce bounds separately (e.g. a projection after
  ``step`` or a bounds-only :mod:`pyzag.reparametrization`).
* Non-finite residuals or preconditioned gradients (which stiff models can
  produce) cause the update to be **skipped with a warning**, never silently.
* For ``mode="full"`` use ``nsub`` (or ``len(sample_indices)``) at least the number
  of parameters, otherwise the sampled ``H`` is rank deficient.
"""

import warnings

import torch


class GaussNewtonPreconditioner:  # pylint: disable=too-many-instance-attributes
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
        rho (float or None): gain-ratio threshold. ``H`` is refreshed when the
            observed loss reduction of the previous step is below ``rho`` times the
            value the cached curvature predicted -- i.e. when the curvature model
            has gone stale. Larger ``rho`` refreshes more eagerly. Set to ``None``
            to never refresh after the first step: a **fixed preconditioner**, the
            cheapest option and often sufficient when the curvature profile is
            stable (e.g. a diagonal), where it acts as a data-driven analogue of a
            static per-parameter rescaling.
        lam (float): Marquardt damping factor; the solve uses ``H + lam * diag(H)``.
        weights (Tensor, optional): diagonal of ``W``, broadcastable to the residual
            shape (e.g. inverse variances). ``None`` means unweighted (``W = I``).
        min_refresh_interval (int): minimum number of steps between refreshes, a
            hard floor on refresh cost (default 1, i.e. no restriction).
        generator (torch.Generator, optional): RNG for reproducible subsampling.
        on_refresh (callable, optional): called whenever the curvature is
            refreshed, as ``on_refresh(step=<0-based step index>,
            gain_ratio=<float or None>, n_refresh=<count>)``. The refreshed step
            indices are also recorded on ``self.refresh_steps`` (useful for
            annotating plots with the recompute iterations).
    """

    def __init__(
        self,
        optimizer,
        parameters,
        *,
        mode="diag",
        nsub=8,
        sample_indices=None,
        rho=0.25,
        lam=1e-2,
        weights=None,
        min_refresh_interval=1,
        generator=None,
        on_refresh=None,
    ):
        if mode not in ("diag", "full"):
            raise ValueError(f"mode must be 'diag' or 'full', got {mode!r}")
        self.optimizer = optimizer
        self.parameters = list(parameters)
        if not self.parameters:
            raise ValueError("parameters is empty")
        self.mode = mode
        self.nsub = int(nsub)
        self.sample_indices = (
            None
            if sample_indices is None
            else torch.as_tensor(sample_indices).reshape(-1).long()
        )
        self.rho = None if rho is None else float(rho)
        self.lam = float(lam)
        self.weights = weights
        self.min_refresh_interval = int(min_refresh_interval)
        self.generator = generator
        self.on_refresh = on_refresh

        self._device = self.parameters[0].device
        self._dtype = self.parameters[0].dtype
        self._sizes = [p.numel() for p in self.parameters]
        self._p = sum(self._sizes)

        # State carried across steps.
        self._H = None  # cached curvature (vector if diag, matrix if full)
        self._prev = None  # {"g", "dtheta", "H", "loss"} of the last accepted step
        self._since_refresh = 0
        self._force_refresh = False
        self._t = -1  # 0-based index of the current step() call

        # Diagnostics.
        self.n_steps = 0
        self.n_refreshes = 0
        self.n_refresh_sweeps = 0
        self.refresh_steps = []  # step indices at which the curvature was refreshed

    # ---- packing helpers ----------------------------------------------------
    def _pack(self, grads):
        out = []
        for g, n in zip(grads, self._sizes):
            out.append(
                torch.zeros(n, device=self._device, dtype=self._dtype)
                if g is None
                else g.reshape(-1)
            )
        return torch.cat(out)

    def _vjp(self, residual, cotangent):
        """One reverse-mode sweep: J^T cotangent, packed to a length-p vector."""
        grads = torch.autograd.grad(
            residual,
            self.parameters,
            grad_outputs=cotangent,
            retain_graph=True,
            allow_unused=True,
        )
        return self._pack(grads)

    def _weights_flat(self, residual):
        if self.weights is None:
            return torch.ones(
                residual.numel(), device=residual.device, dtype=residual.dtype
            )
        return (
            torch.broadcast_to(self.weights, residual.shape)
            .reshape(-1)
            .to(residual.dtype)
        )

    # ---- curvature ----------------------------------------------------------
    def _refresh(self, residual, w_flat):
        """Estimate H = J^T W J from a subsample of residual rows (exact rows,
        rescaled to the full sum) -- one reverse-mode sweep per sampled row."""
        n = residual.numel()
        if self.sample_indices is not None:
            idx = self.sample_indices.to(residual.device)
            if int(idx.max()) >= n or int(idx.min()) < 0:
                raise IndexError(
                    f"sample_indices out of range for residual of size {n}"
                )
        else:
            k = min(self.nsub, n)
            idx = torch.randperm(n, generator=self.generator, device=residual.device)[
                :k
            ]
        if self.mode == "full" and idx.numel() < self._p:
            warnings.warn(
                f"mode='full' with only {idx.numel()} sampled rows < {self._p} parameters: "
                "the sampled Gauss-Newton matrix is rank deficient (Marquardt damping "
                "regularizes it, but consider nsub >= number of parameters).",
                stacklevel=3,
            )

        flat = residual.reshape(-1)
        scale = n / idx.numel()  # unbiased estimate of the full sum
        if self.mode == "diag":
            acc = torch.zeros(self._p, device=self._device, dtype=self._dtype)
        else:
            acc = torch.zeros(self._p, self._p, device=self._device, dtype=self._dtype)
        for ki in idx.tolist():
            e = torch.zeros_like(flat)
            e[ki] = 1.0
            row = self._vjp(residual, e.reshape(residual.shape))  # J_k (row k of J)
            wk = w_flat[ki]
            if self.mode == "diag":
                acc += wk * row * row
            else:
                acc += wk * torch.outer(row, row)
        self._H = acc * scale
        self.n_refreshes += 1
        self.n_refresh_sweeps += int(idx.numel())

    def _precondition(self, g):
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
        return H * d if self.mode == "diag" else H @ d

    # ---- the wrapped step ---------------------------------------------------
    def step(self, residual_closure):
        """Evaluate the residual, precondition the gradient, and step the wrapped
        optimizer. Returns the scalar loss (float) at the current parameters, or
        ``nan`` if the residual was non-finite and the update was skipped.

        Args:
            residual_closure (callable): returns the residual ``r(theta)`` as a
                tensor that is differentiable w.r.t. ``parameters``.
        """
        self._t += 1
        residual = residual_closure()
        w_flat = self._weights_flat(residual)
        r_flat = residual.reshape(-1)
        loss = 0.5 * torch.sum(w_flat * r_flat * r_flat)

        if not torch.isfinite(loss):
            warnings.warn(
                "non-finite residual at the current parameters; skipping update "
                "(a previous step likely left the model's valid region).",
                stacklevel=2,
            )
            self._prev = None  # the quadratic model is no longer anchored
            return float("nan")
        loss_v = float(loss.detach())

        # Gain ratio of the PREVIOUS step: observed vs. predicted loss reduction.
        # Only meaningful when the previous step's quadratic model predicted a
        # non-negligible change; at/near convergence pred_red -> 0 and the ratio is
        # 0/0 noise, which must not be mistaken for staleness.
        gain_ratio = None
        if self._prev is not None:
            actual_red = self._prev["loss"] - loss_v
            Hd = self._quad(self._prev["H"], self._prev["dtheta"])
            pred_red = -(
                torch.dot(self._prev["g"], self._prev["dtheta"])
                + 0.5 * torch.dot(self._prev["dtheta"], Hd)
            ).item()
            eps = 1e-10 * max(abs(self._prev["loss"]), 1.0)
            if pred_red > eps:
                gain_ratio = actual_red / pred_red
            elif pred_red < -eps:
                gain_ratio = -1.0  # model predicted an increase -> curvature stale

        # Refresh the curvature if it is missing, forced, or judged stale. With
        # rho=None the curvature is computed once and never refreshed.
        need = self._H is None or self._force_refresh
        if (
            not need
            and self.rho is not None
            and gain_ratio is not None
            and self._since_refresh >= self.min_refresh_interval
        ):
            need = gain_ratio < self.rho
        if need:
            self._refresh(residual, w_flat)
            self._since_refresh = 0
            self._force_refresh = False
            self.refresh_steps.append(self._t)
            if self.on_refresh is not None:
                self.on_refresh(
                    step=self._t, gain_ratio=gain_ratio, n_refresh=self.n_refreshes
                )
        else:
            self._since_refresh += 1

        # Gradient g = J^T (W r) (one sweep) and the preconditioned direction.
        g = self._vjp(residual, (w_flat * r_flat).reshape(residual.shape))
        pg = self._precondition(g)
        if not bool(torch.isfinite(pg).all()):
            warnings.warn(
                "non-finite preconditioned gradient (ill-conditioned or stale "
                "curvature); skipping this update and forcing a refresh next step.",
                stacklevel=2,
            )
            self._force_refresh = True
            return loss_v

        # Overwrite .grad with the preconditioned gradient and let the base
        # optimizer apply its own update rule to it.
        theta_before = torch.cat([p.detach().reshape(-1) for p in self.parameters])
        off = 0
        for p, n in zip(self.parameters, self._sizes):
            p.grad = pg[off : off + n].reshape(p.shape).clone()
            off += n
        self.optimizer.step()
        theta_after = torch.cat([p.detach().reshape(-1) for p in self.parameters])

        self._prev = {
            "g": g,
            "dtheta": theta_after - theta_before,
            "H": self._H,
            "loss": loss_v,
        }
        self.n_steps += 1
        return loss_v

    def zero_grad(self, *args, **kwargs):
        """Delegate to the wrapped optimizer (provided for drop-in familiarity)."""
        self.optimizer.zero_grad(*args, **kwargs)
