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

"""Tools for converting deterministc models implemented in pytorch to stochastic models

Besides the model-conversion machinery (:class:`MapNormal`,
:class:`HierarchicalStatisticalModel`), this module hosts the Pyro side of
Gauss-Newton preconditioning: :class:`PyroGaussNewtonOptim` and
:class:`PreconditionedSVI` let :mod:`pyzag.preconditioning` drive an SVI fit.
They live here rather than next to the curvature engine because they must
subclass Pyro types at class-definition time; :mod:`pyzag.preconditioning`
itself stays importable without Pyro.
"""

from __future__ import annotations

import warnings

import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
from pyro.infer.util import torch_item, zero_grads
from pyro.nn import PyroSample
from pyro.poutine.subsample_messenger import _Subsample
import torch

from pyzag import preconditioning


class MapNormal:
    """A map between a deterministic torch parameter and a two-scale normal distribution

    Args:
        cov: coefficient of variation used to define the scale priors

    Keyword Args:
        sep (str): seperator character in names
        loc_suffix: suffix to add to parameter name to give the upper-level distribution for the scale
        scale_suffix: suffix to add to the parameter name to give the lower-level distribution for the scale
    """

    def __init__(
        self,
        cov: float,
        loc_suffix: str = "_loc",
        scale_suffix: str = "_scale",
    ) -> None:
        self.cov = cov

        self.loc_suffix = loc_suffix
        self.scale_suffix = scale_suffix

    def __call__(
        self,
        pyro_module: pyro.nn.module.PyroModule,
        name: str,
        value: torch.nn.Parameter,
        prefix: str,
    ) -> tuple[list[str], str]:
        """Apply the mapped conversion to a normal distribution.

        Args:
            pyro_module (pyro.nn.PyroModule): new pyro module to contain parameters
            mod_name (str): string name of module to help disambiguate
            name (str): named of parameter in module
            value (torch.nn.Parameter): value of the parameter
            prefix (str): prefix name to append to the parameter name

        Returns:
            list of names of the new top-level parameters
        """
        dim = value.dim()
        mean = value.detach().clone()
        scale = torch.abs(mean) * self.cov
        setattr(
            pyro_module,
            prefix + name + self.loc_suffix,
            PyroSample(dist.Normal(mean, scale).to_event(dim)),
        )
        setattr(
            pyro_module,
            prefix + name + self.scale_suffix,
            PyroSample(dist.HalfNormal(scale).to_event(dim)),
        )

        setattr(
            pyro_module,
            prefix + name,
            PyroSample(
                lambda m, name=name, dim=dim: dist.Normal(
                    getattr(m, prefix + name + self.loc_suffix),
                    getattr(m, prefix + name + self.scale_suffix),
                ).to_event(dim)
            ),
        )

        return [
            prefix + name + self.loc_suffix,
            prefix + name + self.scale_suffix,
        ], prefix + name


class HierarchicalStatisticalModel(pyro.nn.module.PyroModule):
    """Converts a torch model over to being a Pyro-based hierarchical statistical model

    Args:
        base (torch.nn.Module):     base torch module
        parameter_mapper (MapParameter): mapper class describing how to convert from Parameter to Distribution
        noise_prior (float): scale prior for white noise

    Keyword Args:
        update_mask (bool): if True, update the mask to remove samples that are not valid
    """

    def __init__(
        self,
        base: torch.nn.Module,
        parameter_mapper: MapNormal,
        noise_prior: torch.Tensor,
        update_mask: bool = False,
    ) -> None:
        super().__init__()

        self.base = base

        # Map each parameter to a distribution; record sample sites for each level.
        self.top = []
        self.bot = []
        for nm, m in self.base.named_modules():
            converted_params = []
            for n, val in list(m.named_parameters(recurse=False)):
                upper_params, lower_param = parameter_mapper(self, n, val, nm + ".")
                delattr(m, n)
                setattr(m, n, val.detach().clone())
                self.top.extend(upper_params)
                self.bot.append((m, n, lower_param))
                converted_params.append(n)

            # Record original parameter names on the module so the adjoint
            # method can introspect which parameters to track.
            m.converted_params = converted_params

        if noise_prior.dim() == 0:
            self.sample_noise_outside = True
            self.eps = PyroSample(dist.HalfNormal(noise_prior))
        else:
            self.sample_noise_outside = False
            self.eps = PyroSample(dist.HalfNormal(noise_prior).to_event(0).to_event(1))

        self.update_mask = update_mask
        self.mask = True

    def _sample_top(self) -> list[torch.Tensor]:
        """Sample the top level parameter values"""
        return [getattr(self, n) for n in self.top]

    def _sample_bot(self) -> None:
        """Sample the lower level parameters and assign to the base module"""
        for mod, orig_name, name in self.bot:
            setattr(mod, orig_name, getattr(self, name))

    def forward(
        self,
        *args: torch.Tensor,
        results: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Call the base forward with the appropriate args.

        Args:
            *args: arguments forwarded to the underlying model. At least one must
                be a tensor so the batch shape can be inferred.

        Keyword Args:
            results (torch.tensor or None): results to condition on.
            weights (torch.tensor or None): weights on the results; defaults to ones.
        """
        if len(args) == 0:
            raise ValueError(
                "At least one tensor argument is required to infer the batch dimension."
            )

        shape = args[0].shape[:-1]
        if len(shape) != 2:
            raise ValueError("Input shape must be (ntime, nbatch).")

        if results is not None:
            if results.dim() != 3:
                raise ValueError("The results tensor must be 3-dimensional.")

        if weights is None:
            weights = torch.ones(shape[-1], device=self.eps.device)

        # Sampling top-level here tells Pyro these are not batched over samples.
        _ = self._sample_top()

        # `self.eps` is a PyroSample, so *where* it is read decides where the
        # site is sampled -- outside the plate gives one shared noise scale,
        # inside gives one per sample. The two reads below are exclusive, but
        # that was invisible both to a reader and to pylint (which reported
        # `eps` as possibly unbound); the sentinel makes it explicit.
        eps = self.eps if self.sample_noise_outside else None

        # Nested context managers required so pylint can resolve scale and mask.
        with pyro.plate(
            "samples", shape[-1]
        ), pyro.poutine.scale_messenger.ScaleMessenger(
            scale=weights
        ), pyro.poutine.mask_messenger.MaskMessenger(
            mask=self.mask
        ):
            self._sample_bot()
            res = self.base(*args, **kwargs)

            if self.update_mask:
                self.mask = self.mask & torch.logical_not(
                    torch.any(torch.isnan(res).squeeze(-1), dim=0)
                )
                res = torch.nan_to_num(res)

            if eps is None:
                eps = self.eps

            with pyro.plate("time", shape[0]):
                pyro.sample("obs", dist.Normal(res, eps).to_event(1), obs=results)

        return res


# ---------------------------------------------------------------------------
# Gauss-Newton preconditioning for SVI
# ---------------------------------------------------------------------------


def _site_residual(fn, value):
    """Whitened residual of one Gaussian-family site, shaped like ``value``.

    ``Normal`` gives ``(value - loc) / scale``; ``HalfNormal`` gives
    ``value / scale``. Together these cover everything :class:`MapNormal` and
    :class:`HierarchicalStatisticalModel` create.
    """
    base = fn
    while isinstance(base, dist.Independent):
        base = base.base_dist
    if isinstance(base, dist.HalfNormal):
        return value / base.scale
    if isinstance(base, dist.Normal):
        return (value - base.loc) / base.scale
    raise TypeError(
        f"Gauss-Newton preconditioning needs a Gaussian-family site, got "
        f"{type(base).__name__}. The MAP objective is only a least-squares "
        f"problem when every site is Normal or HalfNormal."
    )


def _site_weight(node, fn, residual):
    """Fold a site's poutine ``scale`` and ``mask`` into its residual rows.

    Both are declared against the site's ``batch_shape`` (they multiply the
    log-density), so they are broadcast there and then unsqueezed over the event
    dimensions to line up with the per-element residual. ``scale`` enters as
    ``sqrt(scale)`` because the residual is squared to form the log-density.
    """
    nevent = len(fn.event_shape)

    def _lift(x):
        x = torch.as_tensor(x, dtype=residual.dtype, device=residual.device)
        x = torch.broadcast_to(x, fn.batch_shape)
        return x.reshape(tuple(fn.batch_shape) + (1,) * nevent)

    out = residual
    scale = node.get("scale", 1.0)
    if not (isinstance(scale, (int, float)) and float(scale) == 1.0):
        out = out * torch.sqrt(_lift(scale))
    mask = node.get("mask")
    if mask is not None and mask is not True:
        out = out * _lift(mask)
    return out


class GaussianResidual:
    """The MAP residual of a Gaussian-family Pyro model, with its site structure.

    For a ``Delta`` guide the negative log-joint is
    ``0.5 * ||r||^2 + sum(log scale)``, so Gauss-Newton applies exactly to the
    ``0.5 * ||r||^2`` part. This object carries that ``r`` plus enough structure
    for a curvature estimator to exploit the model's plates.

    Attributes:
        flat (Tensor): the flat residual, differentiable w.r.t. the guide's
            parameters.
        blocks (dict): site name -> ``slice`` into ``flat``.
        plates (dict): site name -> tuple of enclosing plate names.
        obs_name (str or None): name of the observed site, if any.
        obs_shape (torch.Size): unflattened shape of the observed block.
        obs_scale (Tensor or None): the observed site's scale (the noise level).
    """

    def __init__(self, flat, blocks, plates, obs=None):
        self.flat = flat
        self.blocks = blocks
        self.plates = plates
        obs = obs or {}
        self.obs_name = obs.get("name")
        self.obs_shape = obs.get("shape")
        self.obs_scale = obs.get("scale")
        self.obs_batch_shape = obs.get("batch_shape")
        self.obs_plate_dims = obs.get("plate_dims", {})

    def obs_plate_axis(self, plate_name):
        """Positive axis of ``plate_name`` within the observed batch shape."""
        if plate_name not in self.obs_plate_dims:
            raise ValueError(
                f"the observed site is not inside a plate named {plate_name!r}; "
                f"it is inside {tuple(self.obs_plate_dims)}"
            )
        return len(self.obs_batch_shape) + self.obs_plate_dims[plate_name]

    @property
    def prior_names(self):
        """Names of the latent (unobserved) blocks, in trace order."""
        return [n for n in self.blocks if n != self.obs_name]

    def prior_flat(self):
        """The latent blocks concatenated, differentiably."""
        names = self.prior_names
        if not names:
            return self.flat.new_zeros(0)
        return torch.cat([self.flat[self.blocks[n]] for n in names])


def gaussian_map_residual(
    model, guide, *args, **kwargs
):  # pylint: disable=too-many-locals
    """Assemble the MAP residual of a Gaussian-family Pyro model from a trace.

    Runs the guide, replays the model under it, and turns every Gaussian-family
    sample site into whitened residual rows -- ``(value - loc) / scale`` for
    ``Normal``, ``value / scale`` for ``HalfNormal`` -- including the observed
    site. Any poutine ``scale`` / ``mask`` on a site is folded into its rows, so
    the resulting least-squares objective matches the weighting the ELBO uses.

    Gauss-Newton deliberately ignores the ``sum(log scale)`` log-normalizer terms
    of the log-joint: they are not of least-squares form, and dropping them is the
    standard Gauss-Newton approximation, not an oversight. The residual **must**
    include the prior rows, though -- the likelihood alone has exactly zero
    gradient w.r.t. hierarchical hyper-parameters, so a likelihood-only curvature
    would silently leave them unpreconditioned.

    Args:
        model: the Pyro model (e.g. a :class:`HierarchicalStatisticalModel`).
        guide: the guide, typically ``AutoDelta`` -- a point-mass guide is what
            makes the objective a deterministic least-squares problem.
        \\*args, \\*\\*kwargs: forwarded to both model and guide.

    Returns:
        GaussianResidual
    """
    guide_trace = pyro.poutine.trace(guide).get_trace(*args, **kwargs)
    trace = pyro.poutine.trace(pyro.poutine.replay(model, trace=guide_trace)).get_trace(
        *args, **kwargs
    )

    parts, blocks, plates, obs = [], {}, {}, None
    off = 0
    for name, node in trace.nodes.items():
        if node["type"] != "sample" or isinstance(node["fn"], _Subsample):
            continue
        fn, value = node["fn"], node["value"]
        r = _site_weight(node, fn, _site_residual(fn, value))
        r = torch.broadcast_to(r, value.shape).reshape(-1)
        parts.append(r)
        blocks[name] = slice(off, off + r.numel())
        plates[name] = tuple(f.name for f in node["cond_indep_stack"])
        off += r.numel()
        if node["is_observed"]:
            if obs is not None:
                raise ValueError(
                    f"multiple observed sites ({obs['name']!r} and {name!r}); "
                    "the hierarchical Gauss-Newton estimator assumes one"
                )
            base = fn
            while isinstance(base, dist.Independent):
                base = base.base_dist
            obs = {
                "name": name,
                "shape": value.shape,
                "scale": torch.broadcast_to(base.scale, value.shape),
                "batch_shape": tuple(fn.batch_shape),
                "plate_dims": {f.name: f.dim for f in node["cond_indep_stack"]},
            }

    if not parts:
        raise ValueError("the traced model has no Gaussian sample sites")
    return GaussianResidual(torch.cat(parts), blocks, plates, obs)


def _match_site(param_name, site_names):
    """Map a param-store name onto the model site it parameterizes.

    Guides prefix their parameters with their own name (``AutoDelta.foo`` for
    site ``foo``), so a suffix match recovers the site. Longest match wins;
    genuinely ambiguous names are an error rather than a silent guess.
    """
    cands = sorted(
        (s for s in site_names if param_name == s or param_name.endswith("." + s)),
        key=len,
        reverse=True,
    )
    if not cands:
        return None
    if len(cands) > 1 and len(cands[0]) == len(cands[1]):
        raise ValueError(
            f"parameter {param_name!r} matches several sites {cands[:2]}; cannot "
            "tell which plate it belongs to"
        )
    return cands[0]


def _plate_local_mask(curvature, gres, param_names, plate_name):
    """Boolean mask over the flat parameter vector: is this scalar's site inside
    ``plate_name``, i.e. private to one member of the plate?"""
    mask = torch.zeros(
        curvature.nparam, dtype=torch.bool, device=curvature.parameters[0].device
    )
    off = 0
    for pname, param in zip(param_names, curvature.parameters):
        site = _match_site(pname, gres.plates)
        if site is not None and plate_name in gres.plates[site]:
            mask[off : off + param.numel()] = True
        off += param.numel()
    return mask


def _obs_group_cotangents(
    gres, plate_axis, nsub, generator
):  # pylint: disable=too-many-locals
    """Cotangents that each select one *slice across the whole plate*.

    The Jacobian of the observed block is block diagonal over the plate: a row
    ``(t, b)`` touches only member ``b``'s parameters. So a cotangent that is 1
    at ``(t, ·)`` for every member at once returns, within each member's block,
    exactly that member's row -- the blocks cannot mix. One sweep therefore
    yields the whole plate's rows instead of one, which is the difference
    between covering every member and starving all but a handful of them.
    """
    bshape = tuple(gres.obs_batch_shape)
    nevent = len(gres.obs_shape) - len(bshape)
    group_axes = [i for i in range(len(bshape)) if i != plate_axis]
    counts = [bshape[i] for i in group_axes]
    ngroup = 1
    for c in counts:
        ngroup *= c

    take = min(int(nsub), ngroup)
    picked = torch.randperm(ngroup, generator=generator)[:take].tolist()

    cots = []
    for gi in picked:
        sel = [slice(None)] * len(bshape)
        rem = gi
        for ax, cnt in zip(reversed(group_axes), reversed(counts)):
            sel[ax] = rem % cnt
            rem //= cnt
        m = torch.zeros(bshape, dtype=gres.flat.dtype, device=gres.flat.device)
        m[tuple(sel)] = 1.0
        cots.append(m.reshape(bshape + (1,) * nevent).expand(gres.obs_shape))
    return cots, ngroup / take


def hierarchical_gn_diagonal(  # pylint: disable=too-many-locals
    curvature,
    gres,
    param_names,
    *,
    plate_name="samples",
    nsub=8,
    generator=None,
    validate=True,
):
    """Structure-aware ``diag(J^T J)`` for a hierarchical MAP problem.

    A hierarchical model makes the generic row-subsampling estimator useless:
    with one latent block per plate member, drawing ``nsub`` random rows out of
    ``ntime * nmember`` touches at most ``nsub`` members and leaves every other
    member with zero curvature, while the ``N / nsub`` rescaling badly distorts
    the ones it does hit. This estimator instead splits the residual and uses the
    method each part deserves:

    * **prior rows** carry no forward model, so their whole Jacobian comes back
      from a single batched reverse pass -- exact, and effectively free;
    * **observed rows, plate-local parameters** use one cotangent per slice
      across the plate, which is *exact* per member at ``nmember`` times fewer
      sweeps (see :func:`_obs_group_cotangents`);
    * **observed rows, the shared noise scale** are analytic: the residual is
      ``(y - f) / eps``, so ``d r_k / d eps = -r_k / eps`` and the whole column
      norm follows from ``r`` alone, with no sweep at all.

    Grouped cotangents would double-count a parameter shared across the plate,
    so shared parameters are excluded from the swept estimate and handled by the
    analytic term. Hierarchical hyper-parameters are shared but touch the
    likelihood only through the prior, so the prior block already covers them;
    ``validate`` checks that on the first call rather than assuming it.

    Args:
        curvature (GaussNewtonCurvature): supplies the parameters and the VJP.
        gres (GaussianResidual): from :func:`gaussian_map_residual`.
        param_names (list of str): param-store names, in ``curvature.parameters``
            order, used to map parameters onto model sites.

    Keyword Args:
        plate_name (str): the plate whose members own private latents.
        nsub (int): number of plate-slices to sample from the observed block.
        generator (torch.Generator, optional): RNG for that subsample.
        validate (bool): check the shared-parameter assumption (one extra sweep).

    Returns:
        tuple: ``(diag, nsweeps)`` -- the length-``p`` curvature diagonal and the
        number of reverse-mode sweeps it cost.
    """
    if curvature.mode != "diag":
        raise ValueError(
            f"hierarchical_gn_diagonal builds a diagonal, but the curvature is in "
            f"mode={curvature.mode!r}. A hierarchical model has one latent block "
            f"per plate member, so the full matrix is far too large to form."
        )
    flat = gres.flat
    sizes = [p.numel() for p in curvature.parameters]
    sweeps = 0

    # --- prior block: no forward model in the graph, so one batched pass does it
    r_prior = gres.prior_flat()
    diag = torch.zeros(
        curvature.nparam, dtype=flat.dtype, device=curvature.parameters[0].device
    )
    if r_prior.numel():
        eye = torch.eye(r_prior.numel(), dtype=flat.dtype, device=r_prior.device)
        grads = torch.autograd.grad(
            r_prior,
            curvature.parameters,
            grad_outputs=eye,
            retain_graph=True,
            allow_unused=True,
            is_grads_batched=True,
        )
        jac = torch.cat(
            [
                (
                    torch.zeros(
                        r_prior.numel(), n, dtype=flat.dtype, device=diag.device
                    )
                    if g is None
                    else g.reshape(r_prior.numel(), -1)
                )
                for g, n in zip(grads, sizes)
            ],
            dim=1,
        )
        diag = diag + (jac**2).sum(0)

    if gres.obs_name is None:
        return diag, sweeps

    local = _plate_local_mask(curvature, gres, param_names, plate_name)
    obs_sl = gres.blocks[gres.obs_name]
    axis = gres.obs_plate_axis(plate_name)

    # --- observed block, plate-local parameters: one sweep per plate slice
    cots, scale = _obs_group_cotangents(gres, axis, nsub, generator)
    acc = torch.zeros_like(diag)
    for cot in cots:
        full = torch.zeros_like(flat)
        full[obs_sl] = cot.reshape(-1)
        row = curvature.vjp(flat, full)
        acc = acc + row * row
        sweeps += 1
    diag = diag + torch.where(local, acc * scale, torch.zeros_like(acc))

    # --- observed block, shared noise scale: analytic, no sweep
    eps = gres.obs_scale.reshape(-1)
    deps = curvature.pack(
        torch.autograd.grad(
            eps[0], curvature.parameters, retain_graph=True, allow_unused=True
        )
    )
    r_obs = flat[obs_sl]
    shared_noise = deps**2 * ((r_obs / eps) ** 2).sum()
    diag = diag + torch.where(local, torch.zeros_like(shared_noise), shared_noise)

    if validate:
        _validate_shared(curvature, flat, obs_sl, local, deps)
    return diag, sweeps


def _validate_shared(curvature, flat, obs_sl, local, deps):
    """Warn if a shared parameter couples to the likelihood in a way the
    estimator does not model (one extra sweep).

    Shared parameters get no contribution from the swept estimate, on the
    grounds that in a hierarchical model they reach the data only through the
    plate-local latents. The noise scale is the one exception and is handled
    analytically. Anything else that moves the likelihood would be silently
    left without curvature, so probe for it instead of trusting the assumption.
    """
    probe_cot = torch.zeros_like(flat)
    probe_cot[obs_sl] = 1.0
    probe = curvature.vjp(flat, probe_cot).abs()
    tol = 1e-8 * float(torch.clamp(probe.max(), min=1.0))
    suspect = (~local) & (deps == 0) & (probe > tol)
    if bool(suspect.any()):
        warnings.warn(
            f"{int(suspect.sum())} shared parameter(s) affect the likelihood but are "
            "neither plate-local nor the noise scale; the hierarchical estimator "
            "gives them no curvature from the observed block. They will be "
            "preconditioned by their prior curvature alone.",
            stacklevel=3,
        )


class PyroGaussNewtonOptim(  # pylint: disable=too-many-instance-attributes,too-many-arguments
    pyro.optim.PyroOptim
):
    """Gauss-Newton preconditioning for Pyro SVI.

    A stock :class:`pyro.optim.PyroOptim` builds **one torch optimizer per
    parameter**, so it can never see the cross-parameter structure a
    preconditioner is made of. This subclass instead builds a single optimizer
    over every parameter at once, and rescales the gradient SVI just computed by
    the Gauss-Newton curvature before handing it over. ``SVI`` accepts it because
    it only checks ``isinstance(optim, PyroOptim)``.

    Two details of SVI shape the design. Parameters are created lazily, on the
    first step, so the optimizer and the curvature engine are built on first call
    rather than in ``__init__``. And ``SVI.step`` has already run ``backward`` and
    dropped the graph by the time the optimizer is invoked, so a refresh cannot
    reuse it -- ``residual_closure`` re-evaluates the model to get a
    differentiable residual. That cost is paid only on refresh steps.

    Because pyro hands over an unordered ``set`` of parameters, they are sorted
    by param-store name to give the packed vector a stable, reproducible layout.

    Args:
        optim_constructor: a torch optimizer class or factory, as for
            :class:`pyro.optim.PyroOptim`.
        optim_args (dict): its keyword arguments. Unlike the base class this must
            be a plain dict -- a per-parameter callable cannot describe a single
            optimizer covering all parameters.
        residual_closure (callable): returns the residual at the current
            parameters, differentiable w.r.t. them. Returning a
            :class:`GaussianResidual` (from :func:`gaussian_map_residual`) selects
            the structure-aware hierarchical estimator; a plain tensor falls back
            to generic row subsampling.

    Keyword Args:
        plate_name (str): plate whose members own private latents.
        nsub (int): plate-slices (or rows, generically) sampled per refresh.
        rho (float or None): gain-ratio refresh threshold; ``None`` computes the
            curvature once and reuses it.
        lam (float): initial Marquardt damping factor.
        lam_adapt (bool): Levenberg-Marquardt damping adaptation; see
            :meth:`pyzag.preconditioning.GaussNewtonCurvature.adapt_damping`.
        min_refresh_interval (int): minimum steps between refreshes.
        generator (torch.Generator, optional): RNG for subsampling.
        on_refresh (callable, optional): refresh callback, see
            :class:`pyzag.preconditioning.GaussNewtonCurvature`.
        curvature_fn (callable, optional): full override, called as
            ``curvature_fn(curvature, residual, param_names)`` and returning
            ``(H, nsweeps)``.
        check_optimizer (bool): reject a base optimizer that is invariant to
            gradient preconditioning. SVI's usual choice, ``ClippedAdam``, is
            one of those -- see
            :func:`pyzag.preconditioning.check_optimizer_respects_gradient_scale`.
    """

    def __init__(
        self,
        optim_constructor,
        optim_args,
        residual_closure,
        *,
        plate_name="samples",
        nsub=8,
        rho=0.25,
        lam=1e-2,
        lam_adapt=True,
        min_refresh_interval=1,
        generator=None,
        on_refresh=None,
        curvature_fn=None,
        check_optimizer=True,
    ):
        if callable(optim_args):
            raise ValueError(
                "optim_args must be a dict: this optimizer builds one torch "
                "optimizer covering every parameter, so per-parameter arguments "
                "have nothing to attach to."
            )
        super().__init__(optim_constructor, optim_args)
        self.check_optimizer = check_optimizer
        self.residual_closure = residual_closure
        self.plate_name = plate_name
        self.nsub = int(nsub)
        self.curvature_fn = curvature_fn
        self._engine_args = {
            "mode": "diag",
            "rho": rho,
            "lam": lam,
            "lam_adapt": lam_adapt,
            "min_refresh_interval": min_refresh_interval,
            "generator": generator,
            "on_refresh": on_refresh,
        }
        self.curvature = None
        self.inner = None
        self._names = None
        self._loss = None
        self._validated = False
        self._warned_no_loss = False
        self._pending_state = None

    # ---- loss plumbing ------------------------------------------------------
    def record_loss(self, loss):
        """Tell the optimizer the objective value at the **current** parameters.

        The gain-ratio staleness test needs the loss, and ``SVI.step`` does not
        pass it to the optimizer. :class:`PreconditionedSVI` calls this
        automatically; with a plain :class:`~pyro.infer.SVI` call it yourself
        before each ``svi.step``, or the curvature is computed once and never
        refreshed.
        """
        self._loss = None if loss is None else float(loss)

    def recover_from_failed_evaluation(self):
        """Back out the last step after the model failed to evaluate.

        Called by :class:`PreconditionedSVI` when the ELBO itself raises: with
        SVI the failure happens inside ``loss_and_grads``, before the optimizer
        is ever invoked, so recovery has to be driven from the training loop.
        """
        if self.curvature is None:
            return False
        return self.curvature.reject_failed_evaluation(self.inner)

    @property
    def recorded_loss(self):
        """The most recent loss passed to :meth:`record_loss`, or ``None``."""
        return self._loss

    # ---- lazy construction --------------------------------------------------
    def _build(self, params, names):
        if self._names is not None:
            warnings.warn(
                "the set of SVI parameters changed after training started; "
                "rebuilding the optimizer and discarding the cached curvature "
                "and optimizer state.",
                stacklevel=3,
            )
        self._names = names
        self.curvature = preconditioning.GaussNewtonCurvature(
            params, nsub=self.nsub, **self._engine_args
        )
        self.inner = self.pt_optim_constructor(params, **self.pt_optim_args)
        if self.check_optimizer:
            preconditioning.check_optimizer_respects_gradient_scale(self.inner)
        if self._pending_state is not None:
            self.inner.load_state_dict(self._pending_state)
            self._pending_state = None

    def _refresh(self):
        residual = self.residual_closure()
        if self.curvature_fn is not None:
            H, sweeps = self.curvature_fn(self.curvature, residual, self._names)
        elif isinstance(residual, GaussianResidual):
            H, sweeps = hierarchical_gn_diagonal(
                self.curvature,
                residual,
                self._names,
                plate_name=self.plate_name,
                nsub=self.nsub,
                generator=self._engine_args["generator"],
                validate=not self._validated,
            )
            self._validated = True
        else:
            self.curvature.refresh(residual)
            return
        self.curvature.set_H(H, sweeps)

    # ---- the optimizer call SVI makes ---------------------------------------
    def __call__(self, params, *args, **kwargs):
        store = pyro.get_param_store()
        params = sorted(params, key=store.param_name)
        names = [store.param_name(p) for p in params]
        if names != self._names:
            self._build(params, names)

        gn = self.curvature
        gn.begin_step()
        g = gn.pack_grads()

        loss = self._loss
        if loss is None and gn.rho is not None and not self._warned_no_loss:
            warnings.warn(
                "no loss was recorded, so the gain-ratio refresh trigger is inert "
                "and the curvature will be computed once and reused. Use "
                "PreconditionedSVI, or call optim.record_loss(loss) each step.",
                stacklevel=2,
            )
            self._warned_no_loss = True

        gain_ratio = gn.gain_ratio(loss) if loss is not None else None
        if loss is not None and gn.adapt_damping(loss, gain_ratio):
            # Previous step increased the loss: back it out and damp harder.
            gn.undo_last_step(self.inner)
            return

        refreshed = gn.should_refresh(gain_ratio)
        if refreshed:
            self._refresh()
        gn.note_refresh(gain_ratio, refreshed)

        preconditioning.apply_preconditioned_update(
            gn, g, self.inner, loss, theta_before=gn.theta()
        )

    # ---- state ---------------------------------------------------------------
    # The base class walks self.optim_objs, which stays empty here because there
    # is one global optimizer rather than one per parameter.
    def get_state(self):
        """Serializable state of the single inner optimizer."""
        if self.inner is None:
            return {}
        return {"names": list(self._names), "inner": self.inner.state_dict()}

    def set_state(self, state_dict):
        """Stage state to be loaded when the inner optimizer is built."""
        self._pending_state = state_dict.get("inner") if state_dict else None


class PreconditionedSVI(pyro.infer.SVI):
    """:class:`~pyro.infer.SVI` that reports each step's loss to its optimizer.

    Identical to the base class except that the loss is handed to the optimizer
    before the update. :class:`PyroGaussNewtonOptim` needs the objective value at
    the *current* parameters to run its gain-ratio staleness test, and the stock
    ``SVI.step`` computes exactly that but returns it to the caller instead of
    passing it down. Optimizers without a ``record_loss`` method are unaffected.
    """

    def step(self, *args, **kwargs):
        """Take one SVI step, recording the pre-update loss on the optimizer.

        A model driven outside its valid region raises here, inside
        ``loss_and_grads`` -- before the optimizer runs, so the optimizer cannot
        react on its own. If it can back the last step out and damp harder, do
        that and report the step as ``nan`` rather than letting the whole fit die
        on one bad point.
        """
        try:
            # The concrete messenger class rather than the `pyro.poutine.trace`
            # factory, for the same reason as the scale/mask handlers above:
            # pylint cannot resolve the factory's return type and reports the
            # `with` as a non-context-manager. Same object either way.
            with pyro.poutine.trace_messenger.TraceMessenger(
                param_only=True
            ) as param_capture:
                loss = self.loss_and_grads(self.model, self.guide, *args, **kwargs)
        except (ValueError, RuntimeError) as err:
            recover = getattr(self.optim, "recover_from_failed_evaluation", None)
            if recover is None or not recover():
                raise
            warnings.warn(
                f"the ELBO could not be evaluated ({type(err).__name__}: {err}); "
                "backing out the last step and increasing the damping.",
                stacklevel=2,
            )
            return float("nan")
        params = set(
            site["value"].unconstrained() for site in param_capture.trace.nodes.values()
        )
        record = getattr(self.optim, "record_loss", None)
        if record is not None:
            record(torch_item(loss))
        self.optim(params)
        zero_grads(params)
        return torch_item(loss)
