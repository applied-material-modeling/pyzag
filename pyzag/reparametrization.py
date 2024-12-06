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

"""Helper methods for reparameterizing modules, for example to scale parameter values and gradients"""

import torch
from torch.nn.utils import parametrize


class RangeRescale(torch.nn.Module):
    """Scale parameter within bounds"""

    def __init__(self, lb, ub, clamp=True):
        super().__init__()
        self.lb = lb
        self.ub = ub
        self.clamp = clamp

    def forward(self, X):
        """Go from scaled to natural parameters

        Args:
            X (torch.tensor): scaled parameter values
        """
        if self.clamp:
            X = torch.clamp(X, 0, 1)

        return X * (self.ub - self.lb) + self.lb

    def reverse(self, X):
        """Go from natural to scaled parameter values

        Args:
            X (torch.tensor): natural parameter values
        """
        Y = (X - self.lb) / (self.ub - self.lb)
        if self.clamp:
            return torch.clamp(Y, 0, 1)
        return Y

    def forward_std_dev(self, X):
        """Go from the standard deviation of a scaled normal to the actual standard deviation

        Args:
            X (torch.tensor): scaled standard deviation
        """
        return torch.abs(self.ub - self.lb) * X

    def reverse_std_dev(self, X):
        """Go from the standard deviation of the actual normal to the standard deviation of the scaled normal

        Args:
            X (torch.tensor): natural standard deviation
        """
        return X / torch.abs(self.ub - self.lb)


class Reparameterizer:
    """Reparameterize a torch Module by adding the appropriate rescale function to each parameter

    Args:
        map_dict (dict mapping str to rescaler): dictionary mapping the parameter name to the appropriate rescaler

    Keyword Args:
        error_not_provided (bool): if True, error out if a rescaler is missing

    """

    def __init__(self, map_dict, error_not_provided=False):
        self.map_dict = map_dict
        self.error_not_provided = error_not_provided

    def __call__(self, base):
        """Apply the reparameterization strategy to a model

        This function:
        1. Adds the parameterization
        2. Updates the original value of the parameter to reflect the scaling
        """
        queue = []
        for mname, module in base.named_modules():
            for pname, parameter in module.named_parameters(recurse=False):
                full_name = mname + "." + pname
                if full_name not in self.map_dict:
                    if self.error_not_provided:
                        raise ValueError(
                            "Parameter {pname} is not in the remapping dictionary"
                        )
                    continue

                queue.append(
                    (
                        module,
                        pname,
                        self.map_dict[full_name],
                        mname + ".parametrizations." + pname + ".original",
                        self.map_dict[full_name].reverse(parameter.data),
                    )
                )

        # You need this because the reparameterization changes the named_parameters dict
        for module, pname, reparam, new_name, new_value in queue:
            parametrize.register_parametrization(module, pname, reparam)
            p_scaled = base.get_parameter(new_name)
            p_scaled.data = new_value
