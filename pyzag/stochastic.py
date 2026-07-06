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

"""Tools for converting deterministc models implemented in pytorch to stochastic models"""

from __future__ import annotations

import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroSample


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

        if self.sample_noise_outside:
            eps = self.eps

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

            if not self.sample_noise_outside:
                eps = self.eps

            with pyro.plate("time", shape[0]):
                # eps is assigned in exactly one of the sample_noise_outside
                # branches above; pylint can't see they are exhaustive.
                # pylint: disable=possibly-used-before-assignment
                pyro.sample("obs", dist.Normal(res, eps).to_event(1), obs=results)

        return res
