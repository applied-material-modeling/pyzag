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

"""Assembly / flatten helpers between block containers and flat tensors."""

from __future__ import annotations

import os
from math import prod

import torch

from ._containers import (
    AssembledMatrix,
    AssembledVector,
    AxisLayout,
    BlockLayout,
    BlockTensor,
    BlockVectorAM,
    IStructure,
    Tensor,
)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _first_var_idx_in_group(layout: "AxisLayout", g: int) -> int:
    """Global variable index of the first variable in group g.

    AxisLayout has per-variable accessors (intmd_sizes, base_sizes) but only
    per-group accessors for storage (group_offsets, istr). Since all variables
    in a group share the same IStructure, the first variable's metadata is
    representative; ``group_offsets(g)[0]`` is its global index.
    """
    return layout.group_offsets(g)[0]


def _pcr_tol() -> float:
    """Relative tolerance for the structured-PCR low-rank recompression.

    Default 1e-9 keeps the genuine rank-ns coupling while dropping numerical
    noise; tunable via ``PYZAG_PCR_TOL`` for stress tests.
    """
    try:
        return float(os.environ.get("PYZAG_PCR_TOL", "1e-9"))
    except ValueError:
        return 1e-9


def _group_intmd_sizes(layout: "AxisLayout", g: int) -> list[int]:
    """Intmd shape shared by variables in group g (empty list if DENSE / no intmd)."""
    if layout.istr(g) != AxisLayout.IStructure.BLOCK:
        return []
    return list(layout.intmd_sizes(_first_var_idx_in_group(layout, g)))


def _group_intmd_dim(layout: "AxisLayout", g: int) -> int:
    """Number of intmd dims for group g (0 for DENSE)."""
    return len(_group_intmd_sizes(layout, g))


def _layout_flat_size(layout: "AxisLayout") -> int:
    """Total flat DOF count across all variables (intmd × base summed)."""
    return sum(layout.storage_sizes(include_intmd=True))


def _group_flat_size(layout: "AxisLayout", g: int) -> int:
    """Total flat size (intmd × base) of all variables in group ``g``."""
    start, end = layout.group_offsets(g)
    storage = layout.storage_sizes(include_intmd=True)
    return sum(storage[vi] for vi in range(start, end))


def _split_flat_to_av(raw: torch.Tensor, layout: "AxisLayout") -> "AssembledVector":
    """Inverse of :func:`_av_to_flat`. Split a flat torch tensor
    ``(..., nstate_flat)`` back into an ``AssembledVector`` with one tensor
    per group.

    The flat packing produced by :func:`_av_to_flat` is **per-group,
    instance-major**:

    * **BLOCK group**: per-group tensor has shape
      ``(*dyn, *intmd, sum_of_base_numel)``. ``_av_to_flat`` flattens the
      last ``1 + intmd_ndim`` dims into one, producing
      ``(*dyn, intmd_numel * sum_of_base_numel)`` where the flat slice is
      *instance-major* — block 0's full base entries, then block 1's, etc.
      NOT per-variable contiguous. To invert, we reshape to
      ``(*dyn, *intmd, sum_of_base_numel)`` first, then slice each variable
      out along the base axis.
    * **DENSE group**: per-group tensor has shape ``(*dyn, sum_of_storage)``,
      with each variable contributing ``intmd_numel * base_numel`` contiguous
      entries. A per-variable split by ``storage_sizes`` recovers the parts.

    Builds the per-group tensors directly and constructs the
    ``AssembledVector`` via the ``(layout, tensors)`` ctor — no
    ``SparseVector.assemble`` round-trip.
    """
    dyn_dim = raw.ndim - 1
    group_sizes = [_group_flat_size(layout, g) for g in range(layout.ngroup())]
    group_flats = torch.split(raw, group_sizes, dim=-1)

    group_tensors = []
    for g, gflat in enumerate(group_flats):
        intmd_dim = _group_intmd_dim(layout, g)
        intmd_sz = _group_intmd_sizes(layout, g)  # shared intmd for BLOCK; [] for DENSE
        if intmd_dim > 0:
            # BLOCK: reshape (*dyn, group_size) -> (*dyn, *intmd, base_combined)
            base_combined = gflat.shape[-1] // int(prod(intmd_sz))
            new_shape = gflat.shape[:-1] + tuple(intmd_sz) + (base_combined,)
            gtensor = gflat.reshape(new_shape)
        else:
            gtensor = gflat
        group_tensors.append(Tensor(gtensor, dyn_dim, intmd_dim))
    return AssembledVector(layout, group_tensors)


def _split_flat_per_var(
    raw: torch.Tensor,
    layout: "AxisLayout",
    name_suffix: str = "",
) -> dict:
    """Per-variable companion of :func:`_split_flat_to_av`. Returns
    ``{var_name + suffix: torch.Tensor}`` with each value shaped
    ``(*dyn, *intmd_var, *base_var)``.

    Uses the same per-group / instance-major packing as
    :func:`_split_flat_to_av`, then slices each variable out of its group.
    For BLOCK groups this means slicing along the **combined-base** axis
    AFTER the intmd dim, not along the flat axis.
    """
    group_sizes = [_group_flat_size(layout, g) for g in range(layout.ngroup())]
    group_flats = torch.split(raw, group_sizes, dim=-1)

    out = {}
    for g, gflat in enumerate(group_flats):
        start, end = layout.group_offsets(g)
        intmd_dim = _group_intmd_dim(layout, g)
        intmd_sz = _group_intmd_sizes(layout, g)
        if intmd_dim > 0:
            # BLOCK: reshape and slice per-variable along the last (base) dim.
            base_combined = gflat.shape[-1] // int(prod(intmd_sz))
            shaped = gflat.reshape(
                gflat.shape[:-1] + tuple(intmd_sz) + (base_combined,)
            )
            offset = 0
            for vi in range(start, end):
                bsz = list(layout.base_sizes(vi))
                bnumel = int(prod(bsz)) if bsz else 1
                vslice = shaped[..., offset : offset + bnumel]
                if bsz:
                    vslice = vslice.reshape(vslice.shape[:-1] + tuple(bsz))
                else:
                    vslice = vslice.squeeze(-1)
                out[layout.var(vi) + name_suffix] = vslice
                offset += bnumel
        else:
            # DENSE: per-variable contiguous; split by storage_sizes is correct.
            storage = layout.storage_sizes(include_intmd=True)
            var_sizes = [storage[vi] for vi in range(start, end)]
            var_flats = torch.split(gflat, var_sizes, dim=-1)
            for idx, vi in enumerate(range(start, end)):
                vsz = list(layout.intmd_sizes(vi)) + list(layout.base_sizes(vi))
                if vsz:
                    out[layout.var(vi) + name_suffix] = var_flats[idx].reshape(
                        var_flats[idx].shape[:-1] + tuple(vsz)
                    )
                else:
                    out[layout.var(vi) + name_suffix] = var_flats[idx].squeeze(-1)
    return out


def _build_g_av_from_dict(
    eq_glayout: "AxisLayout",
    value_dict: dict,
    dyn_dim: int,
    fill_zero_for_missing: bool = True,
) -> "AssembledVector":
    """Build a single ``AssembledVector`` whose layout mirrors
    ``eq_glayout``'s variables / groups / istructure but whose intmd shapes
    come from the provided per-variable tensors (or fall back to
    ``eq_glayout``'s own intmd when a var is missing from ``value_dict``).

    This is the supported way to push given-variables into an
    :class:`neml2.es.NonlinearSystem`: one call, one AV, exactly the vars
    NEML2 expects in the right order. Multiple :meth:`set_g` calls are
    *replace*-semantics on the eq_sys's stored g vector, so partitioning g
    across several sublayouts and pushing one at a time silently loses
    earlier values.

    Args:
        eq_glayout:         The eq_sys's authoritative glayout
                            (``sys.glayout()``). We mirror its var names,
                            ordering, group partition, and istructures.
        value_dict:         ``{var_name → torch.Tensor}`` covering every var
                            in ``eq_glayout`` (or all but a few — see
                            ``fill_zero_for_missing``). Each tensor's shape
                            must be ``(*dyn, *intmd, *base)`` for that var.
        dyn_dim:            Number of leading dynamic (batch / time) dims.
        fill_zero_for_missing: If True, vars in ``eq_glayout`` but absent
                            from ``value_dict`` are zero-padded using
                            ``eq_glayout``'s base shape and current intmd.
                            If False, missing vars raise ``KeyError``.
    """
    # Zero-fill needs a reference tensor to source batch shape / dtype / device
    # from. If we'd fall back to zeros and have nothing to copy from, fail early
    # with a useful message rather than letting next(iter({})) raise a vague
    # RuntimeError deep in the loop.
    if fill_zero_for_missing and not value_dict:
        missing = [eq_glayout.var(i) for i in range(eq_glayout.nvar())]
        raise ValueError(
            "_build_g_av_from_dict: value_dict is empty but "
            "fill_zero_for_missing=True. At least one variable's tensor is "
            "required so the batch shape / dtype / device can be inferred. "
            f"eq_glayout expects vars: {missing}"
        )

    grouped_vars = []
    intmd_shapes_flat = []
    base_shapes_flat = []
    istrs = []
    parts = []
    for g in range(eq_glayout.ngroup()):
        start, end = eq_glayout.group_offsets(g)
        group_vars = []
        for vi in range(start, end):
            vname = eq_glayout.var(vi)
            group_vars.append(vname)
            base_shape = list(eq_glayout.base_sizes(vi))
            base_shapes_flat.append(base_shape)
            if vname in value_dict:
                t = value_dict[vname]
                intmd_shape = list(t.shape[dyn_dim : t.ndim - len(base_shape)])
            else:
                if not fill_zero_for_missing:
                    raise KeyError(
                        f"value_dict missing required variable {vname!r} "
                        "and fill_zero_for_missing=False"
                    )
                intmd_shape = list(eq_glayout.intmd_sizes(vi))
                # Source batch / dtype / device from any sibling tensor.
                # (The empty-dict case is guarded above.)
                sample = next(iter(value_dict.values()))
                batch = tuple(sample.shape[:dyn_dim])
                t = torch.zeros(
                    batch + tuple(intmd_shape) + tuple(base_shape),
                    dtype=sample.dtype,
                    device=sample.device,
                )
            intmd_shapes_flat.append(intmd_shape)
            parts.append(Tensor(t, dyn_dim, len(intmd_shape)))
        grouped_vars.append(group_vars)
        istrs.append(eq_glayout.istr(g))
    combined_layout = AxisLayout(
        grouped_vars, intmd_shapes_flat, base_shapes_flat, istrs
    )
    return _pack_per_var_to_av(combined_layout, parts)


def _pack_per_var_to_av(layout: BlockLayout, parts: list[BlockTensor]) -> BlockVectorAM:
    """Pack a per-variable list of :class:`BlockTensor` into a
    :class:`BlockVectorAM` (= AssembledVector). Pure-torch replacement for
    NEML2's ``SparseVector(layout, parts).assemble()``.

    Per-group packing:

    * **BLOCK**: all variables in the group share the same intmd shape.
      Flatten each variable's base dims to one dim, concat along that dim →
      group tensor shape ``(*dyn, *intmd_shared, sum_of_base_numel)``.
    * **DENSE**: flatten each variable's (intmd + base) dims to one, concat →
      group tensor shape ``(*dyn, sum_of_var_storage)``.

    The output ``BlockTensor`` per group carries the appropriate
    ``intmd_dim`` (== shared for BLOCK, 0 for DENSE).
    """
    group_tensors: list[BlockTensor] = []
    for g_idx, gspec in enumerate(layout.groups):
        start, end = layout.group_offsets(g_idx)
        group_parts = [parts[vi] for vi in range(start, end)]
        if gspec.istructure == IStructure.BLOCK:
            flattened = []
            for p in group_parts:
                base_ndim = p.ndim - p.dynamic.dim() - p.intmd.dim()
                t = p.torch()
                if base_ndim > 1:
                    new_shape = t.shape[:-base_ndim] + (
                        int(prod(t.shape[-base_ndim:])),
                    )
                    flattened.append(t.reshape(new_shape))
                elif base_ndim == 1:
                    flattened.append(t)
                else:  # base is scalar
                    flattened.append(t.unsqueeze(-1))
            # Empty group (no variables) -> undefined BlockTensor, mirroring the
            # DENSE branch below. ``torch.cat([])`` would otherwise raise
            # "expected a non-empty list of Tensors" (the cat analog of the
            # historical NEML2-boundary "stack expects a non-empty TensorList").
            group_tensor = torch.cat(flattened, dim=-1) if flattened else None
            intmd_dim = group_parts[0].intmd.dim() if group_parts else 0
            dyn_dim = group_parts[0].dynamic.dim() if group_parts else 0
            group_tensors.append(BlockTensor(group_tensor, dyn_dim, intmd_dim))
        else:  # DENSE
            flattened = []
            for p in group_parts:
                t = p.torch()
                dd = p.dynamic.dim()
                non_dyn = t.ndim - dd
                if non_dyn > 1:
                    new_shape = t.shape[:dd] + (int(prod(t.shape[dd:])),)
                    flattened.append(t.reshape(new_shape))
                elif non_dyn == 1:
                    flattened.append(t)
                else:  # scalar
                    flattened.append(t.unsqueeze(-1))
            group_tensor = torch.cat(flattened, dim=-1) if flattened else None
            dyn_dim = group_parts[0].dynamic.dim() if group_parts else 0
            group_tensors.append(BlockTensor(group_tensor, dyn_dim, 0))
    return BlockVectorAM(layout, group_tensors)


def _av_to_flat(av: "AssembledVector") -> torch.Tensor:
    """Flatten an AssembledVector back to a single flat torch tensor `(..., nstate_flat)`.

    Concatenates per-group tensors along the last dim, after flattening each
    group's intmd+base dims into one.

    Reads ``t.intmd.dim()`` from each Tensor directly rather than
    ``av.layout``'s declared intmd: the Tensor carries its own intmd
    metadata (set at construction or by NEML2's own collect_output), and
    this remains correct even if the AV's layout is somehow stale.
    """
    parts = []
    for t in av.tensors:
        raw = t.torch()
        intmd_ndim = t.intmd.dim()
        # Flatten intmd + base (the last 1+intmd_ndim dims) into one.
        flat_dims = 1 + intmd_ndim  # base is exactly 1 dim (we constructed it that way)
        if flat_dims > 1:
            raw = raw.flatten(start_dim=-flat_dims)
        parts.append(raw)
    return torch.cat(parts, dim=-1)


def _am_to_flat(am: "AssembledMatrix") -> torch.Tensor:
    """Materialize an AssembledMatrix as a single dense ``(nblk, B, n_flat, n_flat)``
    tensor by folding each group's intmd dims into the flat row / col dims.

    Embedding rules per ``(row_istr, col_istr)`` block (i, j):

      * **BLOCK×BLOCK**  : per-instance block-diagonal — for N grains of np×np
        each, builds an (N*np, N*np) block-diagonal matrix.
      * **BLOCK×DENSE**  : per-instance rows, global cols — reshape (N, np, ns)
        to (N*np, ns).
      * **DENSE×BLOCK**  : global rows, per-instance cols — permute and reshape
        (N, ns, np) to (ns, N*np). The flat x packs grain-major, so
        ``M_flat @ x_flat`` automatically sums over grains (matches the per-
        instance Σ_g A_sp[g] x_p[g] interpretation).
      * **DENSE×DENSE**  : passthrough.

    The resulting flat row / col ordering matches :func:`_av_to_flat`'s
    grain-major per-group packing, so ``M_flat @ flat_x`` is mathematically
    identical to the per-instance matvec interpretation used by Schur (and by
    :meth:`NEML2SolvableBlockOperator.matvec`) elsewhere in this module.

    **Grain-broadcast blocks (intmd_dim=0 on a BLOCK row or col)**: NEML2
    may return a Jacobian sub-block without an explicit grain (intmd) dim
    when the per-grain matrix is constant across grains — e.g.
    ``d(per-grain residual)/d(global drate)`` for the Taylor mix-mode
    diagonal. We detect this case (block ``ndim`` smaller than expected
    for the row/col istructure combination) and expand the missing grain
    dim using the row/col layout's declared intmd size before embedding.

    Restricted to **single-intmd** BLOCK groups (covers the Taylor crystal-
    plasticity case). Multi-intmd BLOCK groups would need every intmd dim
    folded; not currently exercised.
    """
    n_rows = am.row_layout.ngroup()
    n_cols = am.col_layout.ngroup()

    def _grains_for(layout, g_idx: int) -> int:
        """Number of grains for a BLOCK group; 1 (no broadcast) for DENSE."""
        if layout.istr(g_idx) != AxisLayout.IStructure.BLOCK:
            return 1
        sz = _group_intmd_sizes(layout, g_idx)
        return int(sz[0]) if sz else 1

    # Reference block for dtype / device / shape when zero-filling undefined blocks.
    ref_tensor = None
    for i in range(n_rows):
        for j in range(n_cols):
            if am.tensors[i][j].defined():
                ref_tensor = am.tensors[i][j].torch()
                break
        if ref_tensor is not None:
            break
    if ref_tensor is None:
        raise RuntimeError("_am_to_flat: AssembledMatrix has no defined block.")

    row_strips = []
    for i in range(n_rows):
        cols = []
        row_istr = am.row_layout.istr(i)
        for j in range(n_cols):
            blk_t = am.tensors[i][j]
            col_istr = am.col_layout.istr(j)
            if blk_t.defined():
                blk = blk_t.torch()
            else:
                # Undefined block — synthesize zero tensor with the right
                # (nblk, sbat, *intmd, n_g, n_h) shape so the embedding
                # branches below produce the correct zero contribution.
                row_grains_z = _grains_for(am.row_layout, i)
                col_grains_z = _grains_for(am.col_layout, j)
                # Per-instance base sizes (single-intmd assumption).
                r_start, r_end = am.row_layout.group_offsets(i)
                c_start, c_end = am.col_layout.group_offsets(j)
                r_base = sum(
                    (
                        int(prod(am.row_layout.base_sizes(vi)))
                        if am.row_layout.base_sizes(vi)
                        else 1
                    )
                    for vi in range(r_start, r_end)
                )
                c_base = sum(
                    (
                        int(prod(am.col_layout.base_sizes(vi)))
                        if am.col_layout.base_sizes(vi)
                        else 1
                    )
                    for vi in range(c_start, c_end)
                )
                nblk, sbat = ref_tensor.shape[0], ref_tensor.shape[1]
                if row_istr == AxisLayout.IStructure.BLOCK:
                    # BLOCK row (regardless of col structure): grains on the row.
                    blk = torch.zeros(
                        nblk,
                        sbat,
                        row_grains_z,
                        r_base,
                        c_base,
                        dtype=ref_tensor.dtype,
                        device=ref_tensor.device,
                    )
                elif col_istr == AxisLayout.IStructure.BLOCK:
                    blk = torch.zeros(
                        nblk,
                        sbat,
                        col_grains_z,
                        r_base,
                        c_base,
                        dtype=ref_tensor.dtype,
                        device=ref_tensor.device,
                    )
                else:
                    blk = torch.zeros(
                        nblk,
                        sbat,
                        r_base,
                        c_base,
                        dtype=ref_tensor.dtype,
                        device=ref_tensor.device,
                    )
            row_grains = _grains_for(am.row_layout, i)
            col_grains = _grains_for(am.col_layout, j)

            if (
                row_istr == AxisLayout.IStructure.BLOCK
                and col_istr == AxisLayout.IStructure.BLOCK
            ):
                # Expected (nblk, B, N, np, np). If intmd missing (ndim==4),
                # NEML2 broadcasted across grains — expand explicitly.
                if blk.ndim == 4:
                    blk = blk.unsqueeze(-3).expand(
                        blk.shape[0], blk.shape[1], row_grains, *blk.shape[-2:]
                    )
                assert blk.ndim == 5, (
                    "_am_to_flat: BLOCK×BLOCK block has unexpected ndim "
                    f"{blk.ndim} (shape {tuple(blk.shape)}); expected 4 or 5."
                )
                grains = blk.shape[-3]
                np_ = blk.shape[-2]
                flat = blk.new_zeros(
                    blk.shape[0], blk.shape[1], grains * np_, grains * np_
                )
                for g in range(grains):
                    flat[..., g * np_ : (g + 1) * np_, g * np_ : (g + 1) * np_] = blk[
                        ..., g, :, :
                    ]
                cols.append(flat)
            elif (
                row_istr == AxisLayout.IStructure.BLOCK
                and col_istr == AxisLayout.IStructure.DENSE
            ):
                # Expected (nblk, B, N, np, ns). Broadcast: (nblk, B, np, ns).
                if blk.ndim == 4:
                    blk = blk.unsqueeze(-3).expand(
                        blk.shape[0], blk.shape[1], row_grains, *blk.shape[-2:]
                    )
                assert blk.ndim == 5, (
                    "_am_to_flat: BLOCK×DENSE block has unexpected ndim "
                    f"{blk.ndim} (shape {tuple(blk.shape)}); expected 4 or 5."
                )
                grains = blk.shape[-3]
                np_ = blk.shape[-2]
                ns_ = blk.shape[-1]
                cols.append(blk.reshape(blk.shape[0], blk.shape[1], grains * np_, ns_))
            elif (
                row_istr == AxisLayout.IStructure.DENSE
                and col_istr == AxisLayout.IStructure.BLOCK
            ):
                # Expected (nblk, B, N, ns, np). Broadcast: (nblk, B, ns, np).
                if blk.ndim == 4:
                    blk = blk.unsqueeze(-3).expand(
                        blk.shape[0], blk.shape[1], col_grains, *blk.shape[-2:]
                    )
                assert blk.ndim == 5, (
                    "_am_to_flat: DENSE×BLOCK block has unexpected ndim "
                    f"{blk.ndim} (shape {tuple(blk.shape)}); expected 4 or 5."
                )
                grains = blk.shape[-3]
                ns_ = blk.shape[-2]
                np_ = blk.shape[-1]
                permuted = blk.permute(0, 1, 3, 2, 4)
                cols.append(
                    permuted.reshape(blk.shape[0], blk.shape[1], ns_, grains * np_)
                )
            else:  # DENSE+DENSE
                cols.append(blk)
        row_strips.append(torch.cat(cols, dim=-1))
    return torch.cat(row_strips, dim=-2)


def _flat_to_sub_am(flat: torch.Tensor, layout: "AxisLayout") -> "AssembledMatrix":
    """Inverse-embed a flat ``(nblk, B, nf, nf)`` operator into a 2-group
    BLOCK(p)+DENSE(s) AssembledMatrix (grain-major row/col order, matching
    :func:`_am_to_flat`).

    NOTE: the BLOCK pp storage can only hold the per-grain DIAGONAL blocks. If
    the flat operator has grain-OFF-diagonal pp coupling (as the reduced
    Schur-PCR subdiagonal does), that coupling is NOT representable here and is
    dropped. This is only ever called by :meth:`pcr_finalize_schur` to build the
    reduced subdiagonal ``B_red``, which is *harmless*: ``B_red`` is never
    consumed in a result-affecting way. In the overlapping power-of-two windowing
    (:meth:`BidiagonalPCRFactorization._pow2`) each window after the first starts
    on the single surviving block of the previous window; the only ``B_red``
    entry a later window reads is that surviving block's subdiagonal, which
    couples it to an already-eliminated in-window predecessor and therefore drops
    out of the solve. (Verified: zeroing the dense ``B_red`` leaves PCR results
    bit-identical to the sequential Thomas solve for non-power-of-two chunks.)
    The truncation here (and the zero ``B_red`` in :meth:`pcr_finalize_multigroup`)
    are correct for exactly this reason.
    """
    if layout.istr(0) == AxisLayout.IStructure.BLOCK:
        p, s = 0, 1
    elif layout.istr(1) == AxisLayout.IStructure.BLOCK:
        p, s = 1, 0
    else:
        raise NotImplementedError("_flat_to_sub_am requires one BLOCK group.")
    intmd = _group_intmd_sizes(layout, p)
    N = int(intmd[0]) if intmd else 1
    np_ = _group_flat_size(layout, p) // N
    ns = _group_flat_size(layout, s)
    P = N * np_
    nblk, B = flat.shape[0], flat.shape[1]
    pp_flat = flat[..., :P, :P]
    # Extract grain-diagonal pp blocks WITHOUT a Python per-grain loop:
    # reshape (..., P, P) -> (..., N, np, N, np), then pull the (g, g) grain
    # diagonal with torch.diagonal over the two grain axes (axes -4 and -2),
    # which lands a new trailing axis of length N. A final move puts it back
    # as (..., N, np, np). This is a single batched view + gather under
    # torch.compile.
    pp_g = pp_flat.reshape(nblk, B, N, np_, N, np_)
    # diagonal over grain axes (-4, -2): result (..., np, np, N)
    pp = torch.diagonal(pp_g, dim1=-4, dim2=-2)
    pp = pp.movedim(-1, -3).contiguous()  # (..., N, np, np)
    ps = flat[..., :P, P:].reshape(nblk, B, N, np_, ns)
    sp = (
        flat[..., P:, :P]
        .reshape(nblk, B, ns, N, np_)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    ss = flat[..., P:, P:]
    T = [[Tensor() for _ in range(2)] for _ in range(2)]
    T[p][p] = Tensor(pp, 2, 1)
    T[p][s] = Tensor(ps, 2, 1)
    T[s][p] = Tensor(sp, 2, 1)
    T[s][s] = Tensor(ss, 2, 0)
    return AssembledMatrix(layout, layout, T)


def _transpose_am(am: "AssembledMatrix") -> "AssembledMatrix":
    """Build A^T as a new AssembledMatrix.

    Swaps row/col layouts and transposes each block's base dims (last two).
    intmd_dim and intmd shapes carry over unchanged on each block.

    Mirrors how DenseBlockJacobian.adjoint_system uses .transpose(-1, -2) on
    the raw tensor + regular DenseBlockOperator. Here we apply the analog at
    the AssembledMatrix level, then wrap in a regular NEML2SolvableBlockOperator
    so the full SolvableBlockOperator API (PCR, solve, matvec) works on the
    adjoint path automatically.
    """
    n_rows = am.row_layout.ngroup()
    n_cols = am.col_layout.ngroup()
    T = [[Tensor() for _ in range(n_rows)] for _ in range(n_cols)]
    for i in range(n_rows):
        for j in range(n_cols):
            blk = am.tensors[i][j]
            # `defined()` exists on neml2.Tensor and is False for default-constructed.
            if blk.defined():
                raw_T = blk.torch().transpose(-1, -2)
                T[j][i] = Tensor(raw_T, blk.dynamic.dim(), blk.intmd.dim())
    return AssembledMatrix(am.col_layout, am.row_layout, T)
