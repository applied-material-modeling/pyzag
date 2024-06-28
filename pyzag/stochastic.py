import torch

import pyro
from pyro.nn import PyroSample
import pyro.distributions as dist
from pyro.distributions import constraints


class HierarchicalStatisticalModel(pyro.nn.module.PyroModule):
    """Converts a torch model over to being a Pyro-based hierarchical statistical model

    Args:
        base (torch.nn.Module):     base torch module
    """

    def __init__(
        self,
        base,
        noise_prior=0.1,
        std_prior=0.1,
        loc_suffix="_loc",
        scale_suffix="_scale",
    ):
        super().__init__()

        # Convert over and make a submodule
        pyro.nn.module.to_pyro_module_(base)
        self.base = base

        self.loc_suffix = loc_suffix
        self.scale_suffix = scale_suffix

        # Our approach is going to be to convert each parameter to a two level normal distribution
        # The mean prior will be the original parameter value and for now the standard deviation
        # will be the mean * prior_std_factor
        #
        # We insert the new pyro distributions at the same locations as the original parameters
        # We keep the bottom level distribution name the same as the original parameter
        # and append the _loc and _scale suffixes for the top level

        # We need to save what to sample at the top level
        self.top = []
        for m in self.base.modules():
            for n, val in list(m.named_parameters(recurse=False)):
                dim = val.dim()
                mean = val.data.detach().clone()
                scale = torch.abs(mean) * std_prior
                setattr(
                    m,
                    n + loc_suffix,
                    PyroSample(dist.Normal(mean, scale).to_event(dim)),
                )
                self.top.append((m, n + loc_suffix))
                setattr(
                    m,
                    n + scale_suffix,
                    PyroSample(dist.HalfNormal(scale).to_event(dim)),
                )

                setattr(
                    m,
                    n,
                    PyroSample(
                        lambda m, name=n, dim=dim: dist.Normal(
                            getattr(m, name + loc_suffix),
                            getattr(m, name + scale_suffix),
                        ).to_event(dim)
                    ),
                )
                self.top.append((m, n + scale_suffix))

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

        # Rather annoying that this is necessary
        vals = self._sample_top()

        # Same here
        eps = self.eps

        with pyro.plate("samples", shape[-1]):
            res = self.base(*args)

            with pyro.plate("time", shape[0]):
                pyro.sample("obs", dist.Normal(res, eps).to_event(1), obs=results)

        return res
