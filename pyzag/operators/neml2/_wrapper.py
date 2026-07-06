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

"""NEML2 <-> pyzag boundary wrapper."""

from __future__ import annotations

import torch

from ..base import BlockVector
from ._assembly import _av_to_flat, _split_flat_to_av
from ._vector import NEML2BlockVector
from ._jacobian import NEML2BlockJacobian


class NEML2Wrapper:
    """Concrete ODEWrapper for the NEML2 backend.

    Converts between pyzag's flat torch tensors (at the function boundary) and
    NEML2's structured AssembledVector / AssembledMatrix (used internally).

    Not strictly an ODE wrapper — NEML2 includes time discretization in the
    model itself — but implements the same interface as DenseODEWrapper so it
    can plug into pyzag's NonlinearFunctionOperatorFactory contract.
    """

    def __init__(self, layout: "AxisLayout") -> None:
        self.layout = layout

    def wrap_vector(self, raw: torch.Tensor) -> NEML2BlockVector:
        """Flat torch (..., nstate_flat) -> NEML2BlockVector with per-group tensors."""
        av = _split_flat_to_av(raw, self.layout)
        return NEML2BlockVector.from_av(av)

    def unwrap_vector(self, bv: BlockVector) -> torch.Tensor:
        """NEML2BlockVector -> flat torch (..., nstate_flat)."""
        if not isinstance(bv, NEML2BlockVector):
            raise TypeError("NEML2Wrapper.unwrap_vector requires NEML2BlockVector.")
        return _av_to_flat(bv.to_av())

    def wrap_jacobian(
        self, diag: "AssembledMatrix", sub: "AssembledMatrix"
    ) -> NEML2BlockJacobian:
        """Wrap diag/sub AssembledMatrix blocks into an NEML2BlockJacobian."""
        return NEML2BlockJacobian(diag, sub, self.layout)
