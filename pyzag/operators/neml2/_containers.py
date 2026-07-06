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

"""Pure-torch NEML2-mirror container types (layout, tensor, matrix, vector)."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from math import prod

import torch


class IStructure(enum.Enum):
    """Group-level structure tag. Mirrors NEML2's ``AxisLayout.IStructure``."""

    BLOCK = "BLOCK"
    DENSE = "DENSE"


@dataclass(frozen=True)
class GroupSpec:
    """Per-group metadata. All variables in a group share the same
    ``IStructure``. Per-variable ``intmd_sizes`` / ``base_sizes`` describe
    each variable's shape contribution."""

    names: tuple[str, ...]
    istructure: IStructure
    intmd_sizes: tuple[tuple[int, ...], ...]
    base_sizes: tuple[tuple[int, ...], ...]


class BlockLayout:
    """Multi-group axis layout. Mirrors NEML2's ``AxisLayout`` API surface
    (``ngroup``, ``nvar``, ``istr``, ``group_offsets``, ``intmd_sizes``,
    ``base_sizes``, ``storage_sizes``, ``var``, ``vars``, ``__eq__``).

    Equality is structural (group-wise comparison of names / intmd / base /
    istructure) — matches NEML2's ``AxisLayout.__eq__`` semantics so the
    boundary adapter at ``interface.py`` can use either side interchangeably
    for shape decisions.

    Two construction styles accepted:

    1. Native: ``BlockLayout(groups=(GroupSpec(...), GroupSpec(...)))``
    2. NEML2-compatible: ``BlockLayout(grouped_vars, intmd_shapes_flat,
       base_shapes_flat, istrs)`` — same positional signature as NEML2's
       ``AxisLayout(...)`` constructor (``intmd_shapes_flat`` / ``base_shapes_flat``
       are per-variable, flattened across groups in row-major order).
    """

    __slots__ = ("groups",)

    def __init__(self, *args, **kwargs) -> None:
        if "groups" in kwargs:
            groups = kwargs["groups"]
        elif len(args) == 1 and len(kwargs) == 0:
            (groups,) = args
        elif len(args) == 4:
            grouped_vars, intmd_shapes_flat, base_shapes_flat, istrs = args
            groups_list = []
            offset = 0
            for gi, names in enumerate(grouped_vars):
                n = len(names)
                groups_list.append(
                    GroupSpec(
                        names=tuple(names),
                        istructure=istrs[gi],
                        intmd_sizes=tuple(
                            tuple(intmd_shapes_flat[offset + i]) for i in range(n)
                        ),
                        base_sizes=tuple(
                            tuple(base_shapes_flat[offset + i]) for i in range(n)
                        ),
                    )
                )
                offset += n
            groups = tuple(groups_list)
        else:
            raise TypeError(
                "BlockLayout expects either `groups=...` or NEML2-style "
                f"4-arg signature; got args={args!r}, kwargs={kwargs!r}"
            )
        self.groups = tuple(groups)

    def __eq__(self, other) -> bool:
        if not isinstance(other, BlockLayout):
            return NotImplemented
        return self.groups == other.groups

    def __hash__(self) -> int:
        return hash(self.groups)

    def ngroup(self) -> int:
        """Number of groups in the layout."""
        return len(self.groups)

    def nvar(self) -> int:
        """Total number of variables across all groups."""
        return sum(len(g.names) for g in self.groups)

    def istr(self, g: int) -> IStructure:
        """Return the :class:`IStructure` tag of group ``g``."""
        return self.groups[g].istructure

    def group_offsets(self, g: int) -> tuple[int, int]:
        """Return the ``(start, end)`` global variable index range of group ``g``."""
        start = sum(len(self.groups[i].names) for i in range(g))
        end = start + len(self.groups[g].names)
        return (start, end)

    def _resolve(self, vi: int) -> tuple[GroupSpec, int]:
        """Translate a global variable index into (group, local index)."""
        offset = 0
        for g in self.groups:
            n = len(g.names)
            if vi < offset + n:
                return g, vi - offset
            offset += n
        raise IndexError(f"variable index {vi} out of range (nvar={self.nvar()})")

    def intmd_sizes(self, vi: int) -> list[int]:
        """Intermediate dim sizes of the global variable index ``vi``."""
        g, local = self._resolve(vi)
        return list(g.intmd_sizes[local])

    def base_sizes(self, vi: int) -> list[int]:
        """Base dim sizes of the global variable index ``vi``."""
        g, local = self._resolve(vi)
        return list(g.base_sizes[local])

    def var(self, vi: int) -> str:
        """Name of the global variable index ``vi``."""
        g, local = self._resolve(vi)
        return g.names[local]

    def vars(self) -> list[str]:
        """Flat list of all variable names across groups."""
        return [n for g in self.groups for n in g.names]

    def storage_sizes(self, include_intmd: bool) -> list[int]:
        """Per-variable storage size (in elements). With ``include_intmd``
        each entry is ``intmd_numel * base_numel``; without, just
        ``base_numel``."""
        result = []
        for g in self.groups:
            for vi in range(len(g.names)):
                base = int(prod(g.base_sizes[vi])) if g.base_sizes[vi] else 1
                if include_intmd:
                    intmd = int(prod(g.intmd_sizes[vi])) if g.intmd_sizes[vi] else 1
                    result.append(intmd * base)
                else:
                    result.append(base)
        return result


class _DimView:
    """Tiny view exposing ``.dim()`` — mirrors NEML2 Tensor's ``.dynamic``."""

    __slots__ = ("_d",)

    def __init__(self, d: int) -> None:
        self._d = d

    def dim(self) -> int:
        """Number of dynamic dims."""
        return self._d


class _IntmdView:
    """View exposing ``.dim()`` and ``.shape`` — mirrors NEML2 Tensor's
    ``.intmd``."""

    __slots__ = ("_d", "_shape")

    def __init__(self, d: int, shape: tuple[int, ...]) -> None:
        self._d = d
        self._shape = shape

    def dim(self) -> int:
        """Number of intermediate dims."""
        return self._d

    @property
    def shape(self) -> tuple[int, ...]:
        """Intermediate dim shape."""
        return self._shape


class BlockTensor:
    """Pyzag-native equivalent of ``neml2.tensors.Tensor``.

    Wraps a ``torch.Tensor`` with explicit ``(dynamic_dim, intmd_dim)``
    metadata; the remaining dims (after dynamic + intmd) are the variable's
    base dims. An undefined ``BlockTensor`` (``raw is None``) represents a
    zero / absent block.

    Mirrors the NEML2 Tensor API surface used throughout the backend:
    ``.torch()``, ``.defined()``, ``.dynamic.dim()``, ``.intmd.dim()``,
    ``.intmd.shape``, ``.ndim``, ``.shape``.
    """

    __slots__ = ("_raw", "_dyn", "_intmd")

    def __init__(
        self,
        raw: torch.Tensor | None = None,
        dynamic_dim: int = 0,
        intmd_dim: int = 0,
    ) -> None:
        self._raw = raw
        self._dyn = dynamic_dim
        self._intmd = intmd_dim

    def defined(self) -> bool:
        """True if this block wraps a real tensor (not a zero/absent block)."""
        return self._raw is not None

    def torch(self) -> torch.Tensor:
        """Return the wrapped :class:`torch.Tensor` (raises if undefined)."""
        if self._raw is None:
            raise RuntimeError("BlockTensor is undefined; cannot extract torch tensor.")
        return self._raw

    @property
    def dynamic(self) -> _DimView:
        """View exposing the number of dynamic dims via ``.dim()``."""
        return _DimView(self._dyn)

    @property
    def intmd(self) -> _IntmdView:
        """View exposing the intermediate dims via ``.dim()`` / ``.shape``."""
        if self._raw is None:
            return _IntmdView(self._intmd, ())
        shape = tuple(self._raw.shape[self._dyn : self._dyn + self._intmd])
        return _IntmdView(self._intmd, shape)

    @property
    def ndim(self) -> int:
        """Number of dims of the wrapped tensor (0 if undefined)."""
        return self._raw.ndim if self._raw is not None else 0

    @property
    def shape(self):
        """Shape of the wrapped tensor (empty :class:`torch.Size` if undefined)."""
        return self._raw.shape if self._raw is not None else torch.Size([])


class BlockMatrix:
    """Pyzag-native equivalent of ``neml2.es.AssembledMatrix``.

    Holds per-(i, j) block tensors with shape
    ``(nblk, sbat, *intmd, n_row_base_combined, n_col_base_combined)``.
    Undefined (i, j) entries are ``BlockTensor(None)`` (treated as zero).
    """

    __slots__ = ("row_layout", "col_layout", "tensors")

    def __init__(
        self,
        row_layout: BlockLayout,
        col_layout: BlockLayout,
        tensors: list[list[BlockTensor]],
    ) -> None:
        self.row_layout = row_layout
        self.col_layout = col_layout
        self.tensors = tensors


class BlockVectorAM:
    """Pyzag-native equivalent of ``neml2.es.AssembledVector``.

    Per-group ``BlockTensor`` entries, shape
    ``(nblk, sbat, *intmd, base_combined)`` per group. Named ``BlockVectorAM``
    (AssembledMatrix-style) to disambiguate from the pyzag ABC
    ``BlockVector`` in ``pyzag.operators.base``.
    """

    __slots__ = ("layout", "tensors")

    def __init__(
        self,
        layout: BlockLayout,
        tensors: list[BlockTensor],
    ) -> None:
        self.layout = layout
        self.tensors = tensors


# Public aliases mirroring the NEML2-side names.
AxisLayout = BlockLayout
AssembledMatrix = BlockMatrix
AssembledVector = BlockVectorAM
Tensor = BlockTensor


# Exposes ``AxisLayout.IStructure.BLOCK`` / ``DENSE`` at the class.
BlockLayout.IStructure = IStructure
