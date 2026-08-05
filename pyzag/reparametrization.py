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

from __future__ import annotations

from typing import Mapping

import torch
from torch.nn.utils import parametrize


class RangeRescale(torch.nn.Module):
    """Scale parameter within bounds"""

    def __init__(
        self,
        lb: torch.Tensor | float,
        ub: torch.Tensor | float,
        clamp: bool = True,
    ) -> None:
        super().__init__()
        self.lb = lb
        self.ub = ub
        self.clamp = clamp

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Go from scaled to natural parameters

        Args:
            X (torch.tensor): scaled parameter values
        """
        if self.clamp:
            X = torch.clamp(X, 0, 1)

        return X * (self.ub - self.lb) + self.lb

    def reverse(self, X: torch.Tensor) -> torch.Tensor:
        """Go from natural to scaled parameter values

        Args:
            X (torch.tensor): natural parameter values
        """
        Y = (X - self.lb) / (self.ub - self.lb)
        if self.clamp:
            return torch.clamp(Y, 0, 1)
        return Y

    def forward_std_dev(self, X: torch.Tensor) -> torch.Tensor:
        """Go from the standard deviation of a scaled normal to the actual standard deviation

        Args:
            X (torch.tensor): scaled standard deviation
        """
        return torch.abs(self.ub - self.lb) * X

    def reverse_std_dev(self, X: torch.Tensor) -> torch.Tensor:
        """Go from the standard deviation of the actual normal to the standard deviation of the scaled normal

        Args:
            X (torch.tensor): natural standard deviation
        """
        return X / torch.abs(self.ub - self.lb)


class CurvatureRescale(torch.nn.Module):
    """Scale a parameter by a curvature-derived step size, with optional bounds.

    The data-driven counterpart of :class:`RangeRescale`. Where that one takes
    the step scale from a hand-picked range width ``(ub - lb)``, this takes it
    from the Gauss-Newton curvature, ``scale = 1 / sqrt(diag H)`` -- build one
    per parameter with :func:`pyzag.preconditioning.gauss_newton_rescalers`.

    **This separates two jobs that a range rescale conflates.** A range width
    both bounds the parameter and sets its step size, and those pull in opposite
    directions: bounds you can trust are wide, and wide bounds condition the
    problem badly. Here the scale comes from the data and ``lb`` / ``ub`` are
    bounds and nothing else, so they can be as generous as honesty requires
    without costing anything.

    Being a reparametrization rather than a gradient preconditioner, it works
    with **any** optimizer -- including Adam, which is invariant to
    :class:`pyzag.preconditioning.GaussNewtonPreconditioner` and rejects it.
    The optimizer's own state (momentum, second moments) lives in the scaled
    coordinates along with the metric, which is what makes the pair coherent.

    It is **static**: the scale is fixed when it is built. If the curvature drifts
    materially over the fit, use ``GaussNewtonPreconditioner``, which can refresh;
    re-scaling mid-run would invalidate a stateful optimizer's moments.

    Args:
        scale (torch.tensor): per-element step scale, ``1 / sqrt(diag H)``.

    Keyword Args:
        offset (torch.tensor or float): natural value at scaled zero (default 0).
        lb, ub (torch.tensor, optional): bounds in **natural** units, clamped
            after scaling. Both must be given, or neither. Note that swapping a
            ``RangeRescale`` for this one *drops its clamp* unless you pass these.
    """

    def __init__(self, scale, offset=0.0, lb=None, ub=None):
        super().__init__()
        if (lb is None) != (ub is None):
            raise ValueError("pass both lb and ub, or neither")
        self.scale = scale
        self.offset = offset
        self.lb = lb
        self.ub = ub

    def _clamp(self, X):
        if self.lb is None:
            return X
        return torch.clamp(X, self.lb, self.ub)

    def forward(self, X):
        """Go from scaled to natural parameters

        Args:
            X (torch.tensor): scaled parameter values
        """
        return self._clamp(X * self.scale + self.offset)

    def reverse(self, X):
        """Go from natural to scaled parameter values

        Args:
            X (torch.tensor): natural parameter values
        """
        return (self._clamp(X) - self.offset) / self.scale

    def forward_std_dev(self, X):
        """Go from the standard deviation of a scaled normal to the actual standard deviation

        Args:
            X (torch.tensor): scaled standard deviation
        """
        return torch.abs(self.scale) * X

    def reverse_std_dev(self, X):
        """Go from the standard deviation of the actual normal to the standard deviation of the scaled normal

        Args:
            X (torch.tensor): natural standard deviation
        """
        return X / torch.abs(self.scale)


class Reparameterizer:
    """Reparameterize a torch Module by adding the appropriate rescale function to each parameter

    Args:
        map_dict (dict mapping str to rescaler): dictionary mapping the parameter name to the appropriate rescaler

    Keyword Args:
        error_not_provided (bool): if True, error out if a rescaler is missing

    """

    def __init__(
        self,
        map_dict: Mapping[str, RangeRescale],
        error_not_provided: bool = False,
    ) -> None:
        self.map_dict = map_dict
        self.error_not_provided = error_not_provided

    def __call__(self, base: torch.nn.Module) -> None:
        """Apply the reparameterization strategy to a model

        This function:
        1. Adds the parameterization
        2. Updates the original value of the parameter to reflect the scaling
        """
        queue = []
        for mname, module in base.named_modules():
            for pname, parameter in module.named_parameters(recurse=False):
                full_name = mname + "." + pname
                # A parameter on the root module has mname == "", so full_name
                # picks up a leading dot that `base.named_parameters()` does not
                # produce. Accept either spelling so a map_dict keyed straight off
                # named_parameters() works for top-level parameters too.
                key = full_name if full_name in self.map_dict else full_name.lstrip(".")
                if key not in self.map_dict:
                    if self.error_not_provided:
                        raise ValueError(
                            f"Parameter {pname} is not in the remapping dictionary"
                        )
                    continue

                queue.append(
                    (
                        module,
                        pname,
                        self.map_dict[key],
                        # lstrip for the same root-module reason as above: with
                        # mname == "" this would start with a dot, and
                        # get_parameter would try to resolve an empty attribute.
                        (mname + ".parametrizations." + pname + ".original").lstrip(
                            "."
                        ),
                        self.map_dict[key].reverse(parameter.detach()),
                    )
                )

        # Re-fetch via the new name: registration changed the named_parameters dict.
        for module, pname, reparam, new_name, new_value in queue:
            parametrize.register_parametrization(module, pname, reparam)
            p_scaled = base.get_parameter(new_name)
            with torch.no_grad():
                p_scaled.copy_(new_value)
