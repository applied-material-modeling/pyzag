import torch

import pyro
from pyro.nn import PyroSample
import pyro.distributions as dist
from pyro.distributions import constraints


class MapNormal:
    """A map between a deterministic torch parameter and a two-scale normal distribution

    Args:
        cov: coefficient of variation used to define the scale priors

    Keyword Args:
        loc_suffix: suffix to add to parameter name to give the upper-level distribution for the scale
        scale_suffix: suffix to add to the parameter name to give the lower-level distribution for the scale
    """

    def __init__(self, cov, loc_suffix="_loc", scale_suffix="_scale"):
        self.cov = cov

        self.loc_suffix = loc_suffix
        self.scale_suffix = scale_suffix

    def __call__(self, module, name, value):
        """Actually do the mapped conversion to a normal distribution

        Args:
            module (torch.nn.Module): object that owns this parameter
            name (str): named of parameter in module
            value (torch.nn.Parameter): value of the parameter

        Returns:
            list of names of the new top-level parameters
        """
        dim = value.dim()
        mean = value.data.detach().clone()
        scale = torch.abs(mean) * self.cov
        setattr(
            module,
            name + self.loc_suffix,
            PyroSample(dist.Normal(mean, scale).to_event(dim)),
        )
        setattr(
            module,
            name + self.scale_suffix,
            PyroSample(dist.HalfNormal(scale).to_event(dim)),
        )

        setattr(
            module,
            name,
            PyroSample(
                lambda m, name=name, dim=dim: dist.Normal(
                    getattr(m, name + self.loc_suffix),
                    getattr(m, name + self.scale_suffix),
                ).to_event(dim)
            ),
        )

        return [name + self.loc_suffix, name + self.scale_suffix]


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
        pyro.nn.module.to_pyro_module_(base)
        self.base = base

        # Run through each parameter and apply the map to convert the parameter to a distribution
        # We also need to save what to sample at the top level
        self.top = []
        for m in self.base.modules():
            for n, val in list(m.named_parameters(recurse=False)):
                new_names = parameter_mapper(m, n, val)
                self.top.extend([(m, n) for n in new_names])

        # Setup noise
        self.eps = PyroSample(dist.HalfNormal(noise_prior))

    def _sample_top(self):
        """Sample the top level parameter values"""
        return [getattr(m, n) for m, n in self.top]

    def forward(self, *args, results=None):
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

        # Rather annoying that this is necessary, this is not a no-op as it tells pyro that these
        # are *not* batched over the number of samples
        vals = self._sample_top()

        # Same here
        eps = self.eps

        with pyro.plate("samples", shape[-1]):
            res = self.base(*args)

            with pyro.plate("time", shape[0]):
                pyro.sample("obs", dist.Normal(res, eps).to_event(1), obs=results)

        return res
