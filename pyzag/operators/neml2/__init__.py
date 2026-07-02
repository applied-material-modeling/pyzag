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

"""
NEML2-native backend for the pyzag block operator system.

Implements pyzag's ``BlockVector`` / ``SolvableBlockOperator`` / ``BlockJacobian``
/ ``ODEWrapper`` ABCs on top of NEML2's ``AssembledVector`` / ``AssembledMatrix``
/ ``AxisLayout``. The diagonal block solve uses a Schur complement when the
system has a 2-group split with at least one ``BLOCK`` group (the crystal
plasticity Taylor model is one canonical case), mirroring NEML2's own
``SchurComplement`` C++ solver pattern. Single-group systems fall back to
batched LU.

The user-facing factory that adapts a NEML2 ``NonlinearSystem`` to pyzag's
``NonlinearFunctionOperatorFactory`` lives in ``neml2.pyzag.interface``
(``NEML2PyzagFactory``).

Design notes (load-bearing for future maintenance):

* **Tensor metadata is authoritative; layouts can be stale.** ``AxisLayout``
  reports each variable's *declared* ``intmd_sizes``, but many models declare
  ``intmd=()`` even when the runtime tensors carry per-instance intmd structure
  (e.g. per-grain crystal-plasticity state with ``intmd=(5,)``). Read
  ``intmd_dim`` from each ``Tensor``'s own ``t.intmd.dim()`` — not the layout —
  when interpreting a tensor's shape.
* **Flat tensor packing is per-group, intmd-major.** ``_av_to_flat`` flattens
  each group's ``(B, *intmd, base_combined)`` to ``(B, intmd_numel *
  base_combined)``. The inverse split must reshape to reintroduce the intmd dim
  before slicing per-variable along the base axis, not split the flat slice by
  per-variable ``storage_sizes`` (which is wrong for multi-variable BLOCK
  groups).
* **Adjoint via materialize-transpose.** Wrap a transposed ``AssembledMatrix``
  (from ``_transpose_am``) in the same ``NEML2SolvableBlockOperator``, giving
  the full ABC (matvec, t_matvec, solve, PCR) on the adjoint path for free — no
  wrapper class.
"""

from ._containers import (
    IStructure,
    GroupSpec,
    BlockLayout,
    BlockTensor,
    BlockMatrix,
    BlockVectorAM,
    AxisLayout,
    AssembledMatrix,
    AssembledVector,
    Tensor,
)
from ._assembly import (
    _first_var_idx_in_group,
    _group_intmd_sizes,
    _group_intmd_dim,
    _layout_flat_size,
    _split_flat_to_av,
    _split_flat_per_var,
    _build_g_av_from_dict,
    _pack_per_var_to_av,
    _av_to_flat,
    _am_to_flat,
    _flat_to_sub_am,
    _transpose_am,
)
from ._vector import (
    NEML2BlockVector,
)
from ._pcr import (
    NEML2PCRState,
    NEML2SchurPCRState,
    MultiGroupPCRState,
    _FlatStructuredAinv,
    _FlatCarrier,
)
from ._operator import (
    NEML2SolvableBlockOperator,
)
from ._jacobian import (
    NEML2BlockJacobian,
)
from ._wrapper import (
    NEML2Wrapper,
)

__all__ = [
    "IStructure",
    "GroupSpec",
    "BlockLayout",
    "BlockTensor",
    "BlockMatrix",
    "BlockVectorAM",
    "AxisLayout",
    "AssembledMatrix",
    "AssembledVector",
    "Tensor",
    "_first_var_idx_in_group",
    "_group_intmd_sizes",
    "_group_intmd_dim",
    "_layout_flat_size",
    "_split_flat_to_av",
    "_split_flat_per_var",
    "_build_g_av_from_dict",
    "_pack_per_var_to_av",
    "_av_to_flat",
    "_am_to_flat",
    "_flat_to_sub_am",
    "_transpose_am",
    "NEML2BlockVector",
    "NEML2PCRState",
    "NEML2SchurPCRState",
    "MultiGroupPCRState",
    "_FlatStructuredAinv",
    "_FlatCarrier",
    "NEML2SolvableBlockOperator",
    "NEML2BlockJacobian",
    "NEML2Wrapper",
]
