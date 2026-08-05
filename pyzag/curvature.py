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


"""Gauss-Newton curvature extraction, shared by everything that needs ``H``.

``H = J^T W J`` for a residual ``r(theta)``, formed without ever materializing
the dense Jacobian: each sampled row costs one reverse-mode sweep. Two consumers
build on this, and they want quite different things from it:

* :class:`pyzag.preconditioning.GaussNewtonCurvature` -- caching, staleness and
  Levenberg-Marquardt damping, to precondition a training loop step by step;
* :func:`pyzag.preconditioning.gauss_newton_rescalers` -- a single estimate, to
  build a static reparametrization.

Keeping the extraction here means one implementation of the row sampling, the
cotangent grouping and the packing, rather than one per consumer.
"""

import warnings

import torch


class CurvatureEstimator:  # pylint: disable=too-many-instance-attributes,too-many-arguments
    """Form the Gauss-Newton curvature ``H = J^T W J`` of a residual.

    The estimation half of Gauss-Newton, with no notion of an optimizer, a
    training loop, or a cached value that might go stale. It packs a parameter
    list into a flat vector, takes reverse-mode products against a residual, and
    turns a chosen set of rows into ``H``. Two rather different consumers need
    exactly that and nothing more:

    * :class:`GaussNewtonCurvature` adds caching, staleness and damping, to
      precondition a training loop step by step;
    * :func:`gauss_newton_rescalers` calls :meth:`estimate` exactly once, to
      build a static reparametrization.

    Args:
        parameters (iterable of Tensor): leaf tensors with ``requires_grad=True``.
            Their shapes and order are frozen here.

    Keyword Args:
        mode (str): ``"diag"`` (default) or ``"full"``.
        nsub (int or None): residual rows to subsample, one reverse sweep each.
            ``None`` (default) uses **every** row, which is exact. Subsampling is
            an opt-in trade: see :meth:`estimate` for what it costs you.
            Ignored if ``sample_indices`` or ``cotangents`` is given.
        sample_indices (array-like of int, optional): explicit rows to sample.
        cotangents (callable or sequence, optional): custom cotangents; see
            :meth:`estimate` for the precondition they carry.
        weights (Tensor, optional): diagonal of ``W``, broadcastable to the
            residual shape. ``None`` means ``W = I``.
        generator (torch.Generator, optional): RNG for reproducible subsampling.
        sampling (str): ``"random"`` (default) draws uniformly without
            replacement; ``"stratified"`` draws one row from each of ``nsub``
            equal blocks of the flattened residual.

            Measured on the NEML2 calibration the two are equivalent: both give a
            usable scale above ``nsub ~ 32`` and both fail below it, because a
            parameter whose sensitivity is concentrated in a small part of the
            residual can be missed either way. Stratified is offered because it
            cannot alias against a ``(ntime, nbatch)`` layout the way an explicit
            stride can, but it is not a substitute for a large enough ``nsub``.

            **Watch the scale, not H.** ``1 / sqrt(H)`` amplifies a near-zero
            entry without bound, so ``||H - H_exact||`` badly understates the
            damage: at ``nsub=8`` that norm is 17% off while the scale it implies
            is wrong by twelve orders of magnitude.
    """

    def __init__(
        self,
        parameters,
        *,
        mode="diag",
        nsub=None,
        sample_indices=None,
        cotangents=None,
        weights=None,
        generator=None,
        sampling="random",
    ):
        if mode not in ("diag", "full"):
            raise ValueError(f"mode must be 'diag' or 'full', got {mode!r}")
        self.parameters = list(parameters)
        if not self.parameters:
            raise ValueError("parameters is empty")
        self.mode = mode
        self.nsub = None if nsub is None else int(nsub)
        self.sample_indices = (
            None
            if sample_indices is None
            else torch.as_tensor(sample_indices).reshape(-1).long()
        )
        # Normalized to a callable so the estimate path has a single shape; a
        # precomputed sequence is accepted and simply returned every time.
        if cotangents is None or callable(cotangents):
            self.cotangents = cotangents
        else:
            _fixed = list(cotangents)
            self.cotangents = lambda _residual: _fixed
        if sampling not in ("stratified", "random"):
            raise ValueError(
                f"sampling must be 'stratified' or 'random', got {sampling!r}"
            )
        self.sampling = sampling
        self.weights = weights
        self.generator = generator

        self._device = self.parameters[0].device
        self._dtype = self.parameters[0].dtype
        self._sizes = [p.numel() for p in self.parameters]
        self._p = sum(self._sizes)

    @property
    def nparam(self):
        """Total number of scalar parameters ``p``."""
        return self._p

    def pack(self, grads):
        """Flatten a per-parameter gradient tuple to a length-``p`` vector,
        substituting zeros for ``None`` (the ``allow_unused`` case)."""
        out = []
        for g, n in zip(grads, self._sizes):
            out.append(
                torch.zeros(n, device=self._device, dtype=self._dtype)
                if g is None
                else g.reshape(-1)
            )
        return torch.cat(out)

    def pack_grads(self):
        """Flatten the parameters' current ``.grad`` to a length-``p`` vector."""
        return self.pack([p.grad for p in self.parameters])

    def write_grads(self, vec):
        """Overwrite every parameter's ``.grad`` with the matching slice of
        ``vec``. Any previously accumulated gradient is discarded."""
        off = 0
        for p, n in zip(self.parameters, self._sizes):
            p.grad = vec[off : off + n].reshape(p.shape).clone()
            off += n

    def _write_params(self, vec):
        """Overwrite the parameters in place from a flat vector.

        Used to back out a rejected step; see :meth:`adapt_damping`.
        """
        off = 0
        with torch.no_grad():
            for p, n in zip(self.parameters, self._sizes):
                p.copy_(vec[off : off + n].reshape(p.shape))
                off += n

    def theta(self):
        """Flat, detached copy of the current parameter values."""
        return torch.cat([p.detach().reshape(-1) for p in self.parameters])

    def vjp(self, residual, cotangent):
        """One reverse-mode sweep: J^T cotangent, packed to a length-p vector."""
        grads = torch.autograd.grad(
            residual,
            self.parameters,
            grad_outputs=cotangent,
            retain_graph=True,
            allow_unused=True,
        )
        return self.pack(grads)

    def weights_flat(self, residual):
        """The diagonal of ``W``, flattened to match the flattened residual."""
        if self.weights is None:
            return torch.ones(
                residual.numel(), device=residual.device, dtype=residual.dtype
            )
        return (
            torch.broadcast_to(self.weights, residual.shape)
            .reshape(-1)
            .to(residual.dtype)
        )

    def _row_cotangents(self, residual):
        """Yield ``(cotangent, weight, nrows)`` triples for one refresh.

        The default path emits one one-hot per sampled row, so ``weight`` is that
        row's ``W`` entry. A custom ``cotangents`` callable folds ``sqrt(W)`` into
        the cotangent itself and reports ``weight = 1``.
        """
        n = residual.numel()
        w_flat = self.weights_flat(residual)

        if self.cotangents is not None:
            cots = list(self.cotangents(residual))
            if not cots:
                raise ValueError("cotangents produced no vectors")
            # Custom cotangents describe their own row grouping, so there is no
            # subsample to extrapolate from: they are taken as-is.
            return [(c.reshape(residual.shape), 1.0, len(cots)) for c in cots], 1.0

        if self.sample_indices is not None:
            idx = self.sample_indices.to(residual.device)
            if int(idx.max()) >= n or int(idx.min()) < 0:
                raise IndexError(
                    f"sample_indices out of range for residual of size {n}"
                )
        else:
            k = n if self.nsub is None else min(self.nsub, n)
            idx = self._sample_rows(n, k, residual.device)
        if self.mode == "full" and idx.numel() < self._p:
            warnings.warn(
                f"mode='full' with only {idx.numel()} sampled rows < {self._p} parameters: "
                "the sampled Gauss-Newton matrix is rank deficient (Marquardt damping "
                "regularizes it, but consider nsub >= number of parameters).",
                stacklevel=4,
            )

        flat = residual.reshape(-1)
        out = []
        for ki in idx.tolist():
            e = torch.zeros_like(flat)
            e[ki] = 1.0
            out.append((e.reshape(residual.shape), w_flat[ki], idx.numel()))
        return out, n / idx.numel()  # unbiased estimate of the full sum

    def _sample_rows(self, n, k, device):
        """``k`` row indices, stratified or uniform per ``sampling``."""
        if self.sampling == "random":
            return torch.randperm(n, generator=self.generator, device=device)[:k]
        # One row from each of k equal blocks: full coverage, no aliasing.
        edge = torch.arange(k + 1, device=device) * n // k
        lo, width = edge[:-1], edge[1:] - edge[:-1]
        draw = torch.rand(k, generator=self.generator, device=device)
        return lo + (draw * width).long().clamp_(max=int(width.max()) - 1)

    def estimate(self, residual):
        """Return ``(H, nsweeps)``, sampling rows as configured.

        With a custom ``cotangents`` sequence each cotangent costs one sweep, but
        the vector it returns is the **sum** of the rows it selects. That sum is
        the correct contribution only when those rows have **disjoint parameter
        support** -- otherwise the estimate picks up cross terms. Valid for
        block-diagonal Jacobians (one cotangent per time step across a
        ``pyro.plate`` of independent specimens); invalid for parameters shared
        across the grouped rows, which must be handled separately.

        Custom cotangents also carry their own weighting: put ``sqrt(w_k)`` in
        entry ``k`` rather than 1, since ``(sum_k sqrt(w_k) J_k)^2`` equals
        ``sum_k w_k J_k^2`` when the supports are disjoint. They are used as
        given, with no ``n / nsampled`` extrapolation -- a custom grouping
        describes its own coverage.
        """
        rows, scale = self._row_cotangents(residual)
        if self.mode == "diag":
            acc = torch.zeros(self._p, device=self._device, dtype=self._dtype)
        else:
            acc = torch.zeros(self._p, self._p, device=self._device, dtype=self._dtype)
        nsweeps = 0
        for cot, wk, _ in rows:
            row = self.vjp(residual, cot)  # J_k (row k of J), or a sum of rows
            if self.mode == "diag":
                acc += wk * row * row
            else:
                acc += wk * torch.outer(row, row)
            nsweeps += 1
        return acc * scale, nsweeps
