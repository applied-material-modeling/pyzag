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

"""NEML2-backend block vector."""

from __future__ import annotations

from typing import Sequence

import torch

from ..base import BlockVector
from ._containers import AssembledVector, Tensor
from ._assembly import _group_intmd_dim, _pack_per_var_to_av


class NEML2BlockVector(BlockVector):
    """Block vector backed by per-group torch tensors with NEML2 layout metadata.

    Storage rationale: we deliberately do NOT store an AssembledVector directly
    because `.tensors[g].torch()` returns a copy/detached view (mutations don't
    propagate). Per-group torch tensors + explicit `intmd_dims` metadata mirror
    the dense backend pattern and let `__setitem__` / `clone` work in-place.

    AssembledVector is materialized only at NEML2 boundaries (via `to_av()`):
    when calling `matvec` (which delegates to `am * av`), `solve`, or NEML2's
    `sys.set_u` / `sys.set_g`.

    Args:
        raw_tensors: list of torch tensors, one per layout group.
            Shape per group: ``(nblk, batch, *intmd_sizes, base_flat)``.
        layout: AxisLayout describing the per-group structure.
        intmd_dims: number of intmd dims per group (consistent with `layout`;
            optional — derived from layout if not given).
    """

    def __init__(
        self,
        raw_tensors: list[torch.Tensor],
        layout: "AxisLayout",
        intmd_dims: list[int] | None = None,
    ) -> None:
        if intmd_dims is None:
            intmd_dims = [_group_intmd_dim(layout, g) for g in range(layout.ngroup())]
        if len(raw_tensors) != layout.ngroup():
            raise ValueError(
                f"NEML2BlockVector expects {layout.ngroup()} per-group tensors, "
                f"got {len(raw_tensors)}."
            )
        if len(intmd_dims) != layout.ngroup():
            raise ValueError(
                f"intmd_dims length ({len(intmd_dims)}) must match layout.ngroup() ({layout.ngroup()})."
            )
        self.raw_tensors = list(raw_tensors)
        self.layout = layout
        self.intmd_dims = list(intmd_dims)

    # ----- conversion to / from NEML2 AssembledVector -----

    def to_av(self) -> "AssembledVector":
        """Materialize as a neml2 AssembledVector (no-copy when possible)."""
        neml2_tensors = [
            Tensor(
                t, t.ndim - i - 1, i
            )  # dynamic = all dims except (intmd + base); base = 1
            for t, i in zip(self.raw_tensors, self.intmd_dims)
        ]
        return AssembledVector(self.layout, neml2_tensors)

    @classmethod
    def from_av(cls, av: "AssembledVector") -> "NEML2BlockVector":
        """Construct from a neml2 AssembledVector by extracting per-group raw tensors.

        Reads ``intmd_dim`` from each Tensor directly rather than from the
        layout — the Tensor's own metadata is authoritative even if the
        layout disagrees.
        """
        layout = av.layout
        intmd_dims = [t.intmd.dim() for t in av.tensors]
        raw_tensors = [t.torch() for t in av.tensors]
        return cls(raw_tensors, layout, intmd_dims)

    # ----- BlockVector abstract methods -----

    @property
    def device(self) -> torch.device:
        return self.raw_tensors[0].device

    @property
    def dtype(self) -> torch.dtype:
        return self.raw_tensors[0].dtype

    @property
    def nblk(self) -> int:
        return self.raw_tensors[0].shape[0]

    @property
    def batch_size(self) -> int:
        return self.raw_tensors[0].shape[1]

    @property
    def block_size(self) -> int:
        # Sum flat state over groups, read from the tensors not the layout: the
        # layout's intmd can be stale (see __init__.py), while the tensors carry
        # the true grain counts.
        total = 0
        for t in self.raw_tensors:
            g = 1
            for d in t.shape[2:]:  # everything after (nblk, batch) is intmd+base
                g *= d
            total += g
        return total

    def clone(self) -> "NEML2BlockVector":
        return NEML2BlockVector(
            [t.clone() for t in self.raw_tensors], self.layout, self.intmd_dims
        )

    def norm(self, dim: int = -1) -> torch.Tensor:
        """Per-block, per-batch L2 norm over the whole (multi-group) state.

        Combined L2 across groups (``sqrt(sum_g ||group_g||^2)``) to match the
        dense backend (:meth:`DenseBlockVector.norm`, a single L2 over the
        concatenated state) and NEML2's own Newton residual norm
        (``pergroup_norm_sq`` = sqrt of sum-across-groups of sum-of-squares). A
        per-group max would give different Newton/line-search stopping than the
        reference backend.
        """
        per_group_sq = []
        for t, i in zip(self.raw_tensors, self.intmd_dims):
            # Flatten the intmd + base dims (last 1 + i dims) into a single dim, then norm.
            flat = t.flatten(start_dim=-(1 + i))  # (nblk, B, dof_in_group)
            per_group_sq.append(torch.norm(flat, dim=dim) ** 2)
        # Sum squares across groups then sqrt -> combined L2, shape (nblk, B).
        return torch.stack(per_group_sq, dim=0).sum(dim=0).sqrt()

    def flat_norm(self) -> torch.Tensor:
        """Cross-block L2 norm per batch over the whole (multi-group) state.

        Combined L2 across groups, matching :meth:`DenseBlockVector.flat_norm`.
        """
        per_group_sq = []
        for t in self.raw_tensors:
            # Batch to front, flatten the rest: (nblk, B, *intmd, base) ->
            # (B, nblk * intmd * base).
            flat = t.transpose(0, 1).flatten(1)
            per_group_sq.append(torch.norm(flat, dim=-1) ** 2)
        return torch.stack(per_group_sq, dim=0).sum(dim=0).sqrt()

    def where(self, mask: torch.Tensor, other: BlockVector) -> "NEML2BlockVector":
        if not isinstance(other, NEML2BlockVector):
            raise TypeError("NEML2BlockVector.where expects NEML2BlockVector.")
        # Broadcast mask (B,) over (nblk, B, *intmd, base) for each group.
        out = []
        for t_self, t_other in zip(self.raw_tensors, other.raw_tensors):
            shape = (1, -1) + (1,) * (t_self.ndim - 2)
            out.append(torch.where(mask.reshape(shape), t_self, t_other))
        return NEML2BlockVector(out, self.layout, self.intmd_dims)

    def scale_batches(self, factor: torch.Tensor) -> "NEML2BlockVector":
        out = []
        for t in self.raw_tensors:
            shape = (1, -1) + (1,) * (t.ndim - 2)
            out.append(t * factor.reshape(shape))
        return NEML2BlockVector(out, self.layout, self.intmd_dims)

    def flip(self, dim: int) -> "NEML2BlockVector":
        return NEML2BlockVector(
            [t.flip(dim) for t in self.raw_tensors], self.layout, self.intmd_dims
        )

    def __neg__(self) -> "NEML2BlockVector":
        return NEML2BlockVector(
            [-t for t in self.raw_tensors], self.layout, self.intmd_dims
        )

    def __add__(self, other: BlockVector) -> "NEML2BlockVector":
        if not isinstance(other, NEML2BlockVector):
            raise TypeError("NEML2BlockVector can only add to NEML2BlockVector.")
        return NEML2BlockVector(
            [a + b for a, b in zip(self.raw_tensors, other.raw_tensors)],
            self.layout,
            self.intmd_dims,
        )

    def __sub__(self, other: BlockVector) -> "NEML2BlockVector":
        if not isinstance(other, NEML2BlockVector):
            raise TypeError("NEML2BlockVector can only subtract NEML2BlockVector.")
        return NEML2BlockVector(
            [a - b for a, b in zip(self.raw_tensors, other.raw_tensors)],
            self.layout,
            self.intmd_dims,
        )

    def __mul__(self, other: torch.Tensor | float | int) -> "NEML2BlockVector":
        return NEML2BlockVector(
            [t * other for t in self.raw_tensors], self.layout, self.intmd_dims
        )

    def __getitem__(self, idx: int | slice) -> "NEML2BlockVector":
        out = []
        for t in self.raw_tensors:
            sliced = t[idx]
            # Match dense-backend behavior: a scalar index returns nblk=1 (unsqueeze).
            if isinstance(idx, int) or (sliced.ndim < t.ndim):
                sliced = sliced.unsqueeze(0)
            out.append(sliced)
        return NEML2BlockVector(out, self.layout, self.intmd_dims)

    def __setitem__(self, idx: int | slice, value: BlockVector) -> None:
        if not isinstance(value, NEML2BlockVector):
            raise TypeError("NEML2BlockVector can only assign from NEML2BlockVector.")
        for t_self, t_val in zip(self.raw_tensors, value.raw_tensors):
            t_self[idx] = t_val

    @classmethod
    def cat(cls, vectors: Sequence[BlockVector], dim: int = 0) -> "NEML2BlockVector":
        if not vectors:
            raise ValueError("cat requires at least one vector")
        for v in vectors:
            if not isinstance(v, NEML2BlockVector):
                raise TypeError("All vectors must be NEML2BlockVector.")
        first = vectors[0]
        out = []
        for g in range(first.layout.ngroup()):
            out.append(torch.cat([v.raw_tensors[g] for v in vectors], dim=dim))
        return NEML2BlockVector(out, first.layout, first.intmd_dims)

    # NOTE: the shape-only ``zeros``/``empty`` constructors are no longer part of
    # the BlockVector interface (they cannot express the NEML2 multi-group
    # layout from a scalar ``block_size``). Use :meth:`zeros_like` for the
    # canonical path, or :meth:`zeros_with_layout` when a layout is available.

    @classmethod
    def zeros_like(cls, other: BlockVector) -> "NEML2BlockVector":
        if not isinstance(other, NEML2BlockVector):
            raise TypeError("NEML2BlockVector.zeros_like requires NEML2BlockVector.")
        return NEML2BlockVector(
            [torch.zeros_like(t) for t in other.raw_tensors],
            other.layout,
            other.intmd_dims,
        )

    @classmethod
    def zeros_with_layout(
        cls,
        nblk: int,
        batch_size: int,
        layout: "AxisLayout",
        dtype: torch.dtype,
        device: torch.device,
    ) -> "NEML2BlockVector":
        """Layout-aware zero factory (NEML2-specific, not in ABC).

        Builds zero tensors per variable, then uses SparseVector.assemble() to
        pack per-group AssembledVector tensors (correctly handles BLOCK/DENSE
        per-group assembly rules).
        """
        sparse_parts = []
        for i in range(layout.nvar()):
            intmd_sizes = list(layout.intmd_sizes(i))
            base_sizes = list(layout.base_sizes(i))
            shape = (nblk, batch_size, *intmd_sizes, *base_sizes)
            sparse_parts.append(
                Tensor(
                    torch.zeros(shape, dtype=dtype, device=device), 2, len(intmd_sizes)
                )
            )
        return cls.from_av(_pack_per_var_to_av(layout, sparse_parts))

    @classmethod
    def empty_with_layout(
        cls,
        nblk: int,
        batch_size: int,
        layout: "AxisLayout",
        dtype: torch.dtype,
        device: torch.device,
    ) -> "NEML2BlockVector":
        """Layout-aware uninitialized factory (NEML2-specific)."""
        sparse_parts = []
        for i in range(layout.nvar()):
            intmd_sizes = list(layout.intmd_sizes(i))
            base_sizes = list(layout.base_sizes(i))
            shape = (nblk, batch_size, *intmd_sizes, *base_sizes)
            sparse_parts.append(
                Tensor(
                    torch.empty(shape, dtype=dtype, device=device), 2, len(intmd_sizes)
                )
            )
        return cls.from_av(_pack_per_var_to_av(layout, sparse_parts))
