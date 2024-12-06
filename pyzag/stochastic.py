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

import torch

import pyro
from pyro.nn import PyroSample
import pyro.distributions as dist


class MapNormal:
    """A map between a deterministic torch parameter and a two-scale normal distribution

    Args:
        cov: coefficient of variation used to define the scale priors

    Keyword Args:
        sep (str): seperator character in names
        loc_suffix: suffix to add to parameter name to give the upper-level distribution for the scale
        scale_suffix: suffix to add to the parameter name to give the lower-level distribution for the scale
    """

    def __init__(self, cov, loc_suffix="_loc", scale_suffix="_scale"):
        self.cov = cov

        self.loc_suffix = loc_suffix
        self.scale_suffix = scale_suffix

    def __call__(self, pyro_module, name, value, prefix):
        """Actually do the mapped conversion to a normal distribution

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
        mean = value.data.detach().clone()
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

    Eventually the plan is to let the user provide a dictionary instead of a single parameter_mapper

    Args:
        base (torch.nn.Module):     base torch module
        parameter_mapper (MapParameter): mapper class describing how to convert from Parameter to Distribution
        noise_prior (float): scale prior for white noise
    """

    def __init__(
        self,
        base,
        parameter_mapper,
        noise_prior,
    ):
        super().__init__()

        # Convert over and make a submodule
        self.base = base

        # Run through each parameter and apply the map to convert the parameter to a distribution
        # We also need to save what to sample at the top level
        self.top = []
        self.bot = []
        for nm, m in self.base.named_modules():
            converted_params = []
            for n, val in list(m.named_parameters(recurse=False)):
                upper_params, lower_param = parameter_mapper(self, n, val, nm + ".")
                # Isn't this fun
                delattr(m, n)
                setattr(m, n, val.data.detach().clone())
                self.top.extend(upper_params)
                self.bot.append((m, n, lower_param))
                converted_params.append(n)

            # This adds a new parameter to the module itself giving the *names* of the original parameters
            # This is required for the adjoint method to do introspection on what to track
            m.converted_params = converted_params

        # Setup noise
        if noise_prior.dim() == 0:
            self.sample_noise_outside = True
            self.eps = PyroSample(dist.HalfNormal(noise_prior))
        else:
            self.sample_noise_outside = False
            self.eps = PyroSample(dist.HalfNormal(noise_prior).to_event(0).to_event(1))

    def _sample_top(self):
        """Sample the top level parameter values"""
        return [getattr(self, n) for n in self.top]

    def _sample_bot(self):
        """Sample the lower level parameters and assign to the base module"""
        for mod, orig_name, name in self.bot:
            setattr(mod, orig_name, getattr(self, name))

    def forward(self, *args, results=None, **kwargs):
        """Class the base forward with the appropriate args

        Args:
            *args: whatever arguments the underlying model needs.  But at least one must be a tensor so we can infer the correct batch shapes!

        """
        if len(args) == 0:
            raise ValueError(
                "Model cannot use introspection to figure out the batch dimension!"
            )

        shape = args[0].shape[:-1]
        if len(shape) != 2:
            raise ValueError("For now we require shape of (ntime, nbatch)")

        if results is not None:
            if results.dim() != 3:
                raise ValueError(
                    "The results tensor should be a dim = 3 tensor, maybe unsqueeze your output?"
                )

        # Rather annoying that this is necessary, this is not a no-op as it tells pyro that these
        # are *not* batched over the number of samples
        _ = self._sample_top()

        # Same here
        if self.sample_noise_outside:
            eps = self.eps

        with pyro.plate("samples", shape[-1]):
            self._sample_bot()
            res = self.base(*args, **kwargs)

            if not self.sample_noise_outside:
                eps = self.eps

            with pyro.plate("time", shape[0]):
                pyro.sample("obs", dist.Normal(res, eps).to_event(1), obs=results)

        return res
