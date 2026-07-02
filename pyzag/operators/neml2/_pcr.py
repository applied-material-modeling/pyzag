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

"""PCR state and flat/Schur carrier machinery for the NEML2 backend."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..base import PCRState
from ..dense import DensePCRState, _dense_pcr_cyclic_shift
from ._containers import AxisLayout
from ._assembly import _group_flat_size


# ---------------------------------------------------------------------------
# Structure-preserving O(N) multi-group Schur-PCR carrier.
#
# Mathematical basis (validated numerically on the Taylor regression model):
#
# The diagonal operator ``A`` (2-group BLOCK(p) + DENSE(s)) has an EXACT
# structured inverse
#
#     A^{-1} = blockdiag_pp(App[g]^{-1})  +  G,   rank(G) <= ns
#
# where G is the global low-rank Schur coupling ``L S^{-1} R`` (plus the ps/sp/ss
# blocks). The subdiagonal ``B`` (=Jn) is block-diagonal in groups (Taylor:
# Jn_pp = -I per grain, Jn_ss absent, only a small Jn_sp), so B itself is a
# carrier of the SAME form. The PCR reduction products
#
#     v[1:] -= B[1:] @ (A[:-1]^{-1} v[:-1])
#     b[2:]  = -B[2:] @ (A[1:-1]^{-1} B[1:-1])
#
# preserve this ``blockdiag_pp + global-low-rank`` form: the product of two
# carriers is again a carrier whose low-rank inner dimension grows by at most a
# constant per multiply (independent of the grain count N). On the real Taylor
# data the reduced low-rank stays at rank ~5 across all PCR levels and all N
# (5..200), with reconstruction error ~1e-12. This is what lets the structured
# path be O(N) where the flat-Dense PCR is O(N^3).
#
# Representation of a flat (nf x nf) per-time-step operator, batched over the
# PCR leading dims (W=tree-window, nblk=time, B=batch):
#
#   * ``Dg`` : (W, nblk, B, N, np, np)  per-grain pp block-diagonal blocks
#   * ``U``  : (W, nblk, B, nf, r)      global left low-rank factor (flat space)
#   * ``V``  : (W, nblk, B, nf, r)      global right low-rank factor (flat space)
#
# so the dense operator is  M = blockdiag_pp(Dg) + U @ V^T. The flat row/col
# ordering is grain-major p-rows (N*np) then s-rows (ns), matching ``_am_to_flat``
# / ``_av_to_flat``, so we can convert at the chunk boundary and validate against
# the flat-Dense oracle exactly.
# ---------------------------------------------------------------------------


@dataclass
class _FlatStructuredAinv:
    """Batched structured inverse of the diagonal operator in flat space.

    Holds, with arbitrary leading batch dims ``(...,)`` (the PCR carries
    ``(W, nblk, B)``):

      * ``App_inv`` : (..., N, np, np)  per-grain App^{-1}
      * ``L``       : (..., N, np, ns)  = App^{-1} A_ps
      * ``R``       : (..., N, ns, np)  = A_sp App^{-1}
      * ``S_inv``   : (..., ns, ns)     = (A_ss - sum_g A_sp App^{-1} A_ps)^{-1}

    Applies ``A^{-1} @ X`` for a flat matrix/vector ``X`` (grain-major rows)
    using the Schur identity; the cross-grain coupling enters only through the
    shared ``x_s`` channel (rank ns), so the apply is O(N).
    """

    App_inv: torch.Tensor
    L: torch.Tensor
    R: torch.Tensor
    S_inv: torch.Tensor
    np_: int
    ns: int
    ngrain: int

    @classmethod
    def from_diag_am(cls, am: "AssembledMatrix") -> "_FlatStructuredAinv":
        """Build from a 2-group BLOCK(p)+DENSE(s) diagonal AssembledMatrix.

        Cross blocks A_ps / A_sp may be undefined (treated as zero). Grain-
        broadcast App / A_sp (no explicit grain dim) are expanded to N grains.
        """
        row_l = am.row_layout
        if row_l.istr(0) == AxisLayout.IStructure.BLOCK:
            p, s = 0, 1
        elif row_l.istr(1) == AxisLayout.IStructure.BLOCK:
            p, s = 1, 0
        else:
            raise NotImplementedError(
                "_FlatStructuredAinv requires one BLOCK group (Taylor mix-mode)."
            )
        A_pp = am.tensors[p][p].torch()  # (..., N, np, np)
        ngrain = A_pp.shape[-3]
        A_ps_t = am.tensors[p][s]
        A_sp_t = am.tensors[s][p]
        A_ss = am.tensors[s][s].torch()  # (..., ns, ns)
        np_ = A_pp.shape[-1]
        ns = A_ss.shape[-1]

        App_inv = torch.linalg.inv(A_pp)
        if A_ps_t.defined():
            A_ps = A_ps_t.torch()  # (..., N, np, ns) or (..., np, ns) broadcast
            if A_ps.shape[-3] != ngrain:
                A_ps = A_ps.unsqueeze(-3).expand(*A_ps.shape[:-2], ngrain, np_, ns)
            L = torch.matmul(App_inv, A_ps)
        else:
            L = torch.zeros(
                *App_inv.shape[:-1], ns, dtype=A_pp.dtype, device=A_pp.device
            )
        if A_sp_t.defined():
            A_sp = A_sp_t.torch()  # (..., N, ns, np) or broadcast (..., ns, np)
            if A_sp.shape[-3] != ngrain:
                A_sp = A_sp.unsqueeze(-3).expand(*A_sp.shape[:-2], ngrain, ns, np_)
            R = torch.matmul(A_sp, App_inv)
            S = A_ss - torch.matmul(A_sp, L).sum(dim=-3)
        else:
            R = torch.zeros(
                *App_inv.shape[:-2], ns, np_, dtype=A_pp.dtype, device=A_pp.device
            )
            S = A_ss
        S_inv = torch.linalg.inv(S)
        return cls(App_inv, L, R, S_inv, np_, ns, ngrain)

    def apply_flat(self, X: torch.Tensor) -> torch.Tensor:
        """Compute ``A^{-1} @ X`` for X of shape ``(..., nf, k)`` in flat
        grain-major row order; returns the same shape. ``...`` must broadcast
        with the stored factors' leading dims.
        """
        N, np_, ns = self.ngrain, self.np_, self.ns
        P = N * np_
        k = X.shape[-1]
        bp = X[..., :P, :].reshape(*X.shape[:-2], N, np_, k)
        bs = X[..., P:, :]  # (..., ns, k)
        z_p = torch.matmul(self.App_inv, bp)  # (..., N, np, k)
        d_s = bs - torch.matmul(self.R, bp).sum(dim=-3)  # (..., ns, k)
        x_s = torch.matmul(self.S_inv, d_s)  # (..., ns, k)
        x_p = z_p - torch.matmul(self.L, x_s.unsqueeze(-3))  # broadcast x_s over grains
        out = X.new_empty(X.shape)
        out[..., :P, :] = x_p.reshape(*X.shape[:-2], P, k)
        out[..., P:, :] = x_s
        return out

    def to_carrier(self) -> "_FlatCarrier":
        """Express ``A^{-1}`` as a :class:`_FlatCarrier` with EXACT rank-ns
        global handle.

        From the 2x2 Schur inverse, ``A^{-1} = blockdiag_pp(App^{-1}) + G`` with

            G = W_L @ S_inv @ W_R,   W_L = [L; -I_ns],  W_R = [R | -I_ns]

        so ``U = W_L @ S_inv`` (nf x ns), ``V = W_R^T`` (nf x ns). Verified
        against the dense inverse on the Taylor regression data.
        """
        N, np_, ns = self.ngrain, self.np_, self.ns
        P = N * np_
        nf = P + ns
        lead = self.App_inv.shape[:-3]
        dev, dt = self.App_inv.device, self.App_inv.dtype
        eye_s = torch.eye(ns, dtype=dt, device=dev).expand(*lead, ns, ns)
        # W_L = [L; -I_ns] as (..., nf, ns); L's grain-major p-rows flatten to P.
        W_L = torch.zeros(*lead, nf, ns, dtype=dt, device=dev)
        W_L[..., :P, :] = self.L.reshape(*lead, P, ns)
        W_L[..., P:, :] = -eye_s
        # W_R = [R | -I_ns] as (..., ns, nf); R's grain-major p-cols flatten to P.
        W_R = torch.zeros(*lead, ns, nf, dtype=dt, device=dev)
        # R is (..., N, ns, np): move grain to the col side -> (..., ns, N, np) -> (..., ns, P).
        R_flat = self.R.transpose(-2, -3).reshape(*lead, ns, P)
        W_R[..., :, :P] = R_flat
        W_R[..., :, P:] = -eye_s
        U = torch.matmul(W_L, self.S_inv)  # (..., nf, ns)
        V = W_R.transpose(-1, -2)  # (..., nf, ns)
        Dg = self.App_inv
        return _FlatCarrier(Dg, U, V, np_, ns, N)


@dataclass
class _FlatCarrier:
    """A flat operator ``M = blockdiag_pp(Dg) + U @ V^T`` with batch leading dims.

    ``Dg``: (..., N, np, np); ``U``,``V``: (..., nf, r). Operations keep the
    representation structured (never materialize the dense (nf x nf) matrix)
    except for validation, so all ops are O(N * r^2) per time step.
    """

    Dg: torch.Tensor
    U: torch.Tensor
    V: torch.Tensor
    np_: int
    ns: int
    ngrain: int

    @property
    def nf(self) -> int:
        return self.ngrain * self.np_ + self.ns

    def dense(self) -> torch.Tensor:
        """Materialize the dense ``(..., nf, nf)`` matrix.

        Vectorized (no Python per-grain loop): the per-grain ``Dg`` blocks
        ``(*lead, N, np, np)`` are scattered onto the ``P=N*np`` block diagonal
        by building a ``(*lead, N, np, N, np)`` tensor whose only nonzero
        grain-pair is the (g, g) diagonal — achieved with an ``(N, N)`` identity
        mask broadcast over the ``(np, np)`` block axes — then reshaping to
        ``(*lead, P, P)``. The global rank-r handle ``U @ V^T`` is added on the
        full ``nf`` grid. This keeps the op a single batched graph under
        ``torch.compile``.
        """
        lead = self.Dg.shape[:-3]
        N, np_ = self.ngrain, self.np_
        P = N * np_
        nf = self.nf
        dt, dev = self.U.dtype, self.U.device
        # (N, N) grain identity broadcast over (np, np): keeps Dg on the
        # block diagonal only. Dg: (*lead, N, 1, np, np) ; eye: (N, N, 1, 1).
        eyeN = torch.eye(N, dtype=dt, device=dev).reshape(N, N, 1, 1)
        pp = self.Dg.unsqueeze(-3) * eyeN  # (*lead, N, N, np, np)
        # (*lead, N, np, N, np) -> (*lead, P, P)
        pp = pp.transpose(-3, -2).reshape(*lead, P, P)
        M = self.U.new_zeros(*lead, nf, nf)
        M[..., :P, :P] = pp
        M = M + torch.matmul(self.U, self.V.transpose(-1, -2))
        return M

    def matmul_flat(self, X: torch.Tensor) -> torch.Tensor:
        """Compute ``M @ X`` for flat ``X`` of shape ``(..., nf, k)``."""
        N, np_ = self.ngrain, self.np_
        P = N * np_
        k = X.shape[-1]
        out = X.new_zeros(X.shape)
        xp = X[..., :P, :].reshape(*X.shape[:-2], N, np_, k)
        dp = torch.matmul(self.Dg, xp)  # (..., N, np, k)
        out[..., :P, :] = dp.reshape(*X.shape[:-2], P, k)
        # global low-rank: U @ (V^T @ X)
        VtX = torch.matmul(self.V.transpose(-1, -2), X)  # (..., r, k)
        out = out + torch.matmul(self.U, VtX)
        return out


def _carrier_recompress(
    U: torch.Tensor, V: torch.Tensor, tol: float = 1e-9, fixed_rank: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompress a batched low-rank product ``U @ V^T`` to a smaller common
    inner rank via QR + SVD on the small R factors (never forms U@V^T).

    ``U``,``V``: (..., nf, k). Returns (..., nf, r) with a SINGLE r shared
    across the whole batch (so the carry stays a dense tensor). When
    ``fixed_rank`` is given, the result has EXACTLY that inner dim (truncated or
    zero-padded) — required when the carrier is mutated in place on a strided
    view, which needs a constant trailing shape. Otherwise r is chosen as the
    max numerical rank over the batch at relative tolerance ``tol``.
    """
    if U.shape[-1] == 0:
        # Empty time slice (e.g. nblk==2 -> b[:, 1:-1] is empty): nothing to do.
        # This is a STATIC shape test (Python int), so it is compile-safe and
        # does not graph-break.
        return U, V
    # Non-finite handling, BRANCHLESS (compile-safe): a non-finite carrier means
    # the current Newton iterate's A is bad (e.g. stiff cold start). We must not
    # raise (aborts the whole solve) and must propagate NaN so the line-search
    # rejects the step. A data-dependent ``if torch.isfinite(...).all()`` branch
    # graph-breaks under ``torch.compile``, so instead: (1) sanitize inputs to
    # finite values so qr/svd cannot error, recompress normally, then (2) re-stamp
    # NaN wherever any input element was non-finite via a ``torch.where`` mask.
    # Numerically identical on the finite path; all-NaN U/V on the bad one.
    finite = torch.isfinite(U).all(dim=(-2, -1)) & torch.isfinite(V).all(dim=(-2, -1))
    bad = ~finite  # (..., ) per-batch flag
    U = torch.where(bad.unsqueeze(-1).unsqueeze(-1), torch.zeros_like(U), U)
    V = torch.where(bad.unsqueeze(-1).unsqueeze(-1), torch.zeros_like(V), V)
    Qu, Ru = torch.linalg.qr(U, mode="reduced")  # Qu (..,nf,k), Ru (..,k,k)
    Qv, Rv = torch.linalg.qr(V, mode="reduced")
    mid = torch.matmul(Ru, Rv.transpose(-1, -2))  # (..., k, k)
    Um, Sm, Vmh = torch.linalg.svd(mid, full_matrices=False)  # Sm (..., k)
    if fixed_rank is not None:
        keep = min(fixed_rank, Sm.shape[-1])
    else:
        # Choose a single kept rank across the batch: largest count of singular
        # values above the per-instance relative threshold.
        smax = Sm[..., :1].clamp_min(torch.finfo(Sm.dtype).tiny)
        keep_mask = Sm > (tol * smax)
        per_inst = keep_mask.reshape(-1, Sm.shape[-1]).sum(dim=-1)
        keep = int(per_inst.max().item()) if per_inst.numel() else 1
        keep = max(keep, 1)
    s_sqrt = Sm[..., :keep].clamp_min(0.0).sqrt().unsqueeze(-2)  # (..., 1, keep)
    Un = torch.matmul(Qu, Um[..., :, :keep]) * s_sqrt
    Vn = torch.matmul(Qv, Vmh[..., :keep, :].transpose(-1, -2)) * s_sqrt
    if fixed_rank is not None and keep < fixed_rank:
        padU = Un.new_zeros(*Un.shape[:-1], fixed_rank - keep)
        padV = Vn.new_zeros(*Vn.shape[:-1], fixed_rank - keep)
        Un = torch.cat([Un, padU], dim=-1)
        Vn = torch.cat([Vn, padV], dim=-1)
    # Re-stamp NaN onto outputs for batch entries whose inputs were non-finite
    # (branchless; mirrors the old all-NaN early return for those entries).
    nan = torch.full((), float("nan"), dtype=Un.dtype, device=Un.device)
    Un = torch.where(bad.unsqueeze(-1).unsqueeze(-1), nan, Un)
    Vn = torch.where(bad.unsqueeze(-1).unsqueeze(-1), nan, Vn)
    return Un, Vn


def _carrier_mul(
    M1: _FlatCarrier, M2: _FlatCarrier, tol: float = 1e-9, fixed_rank: int | None = None
) -> _FlatCarrier:
    """Structured product ``M1 @ M2`` of two carriers (batched, structure-
    preserving). See module-level derivation: the product is

        blockdiag_pp(Dg1 @ Dg2)
        + [bd_left(Dg1, U2) | U1 | U1 @ (V1^T U2)] @ [V2 | bd_right(Dg2, V1) | V2]^T

    where bd_left applies the pp block-diagonal of M1 to U2's p-rows (zeroing
    s-rows) and bd_right applies Dg2^T to V1's p-rows.
    """
    N, np_, ns = M1.ngrain, M1.np_, M1.ns
    P = N * np_
    Dg = torch.matmul(M1.Dg, M2.Dg)

    def bd_left(Dg_blocks: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        # X: (..., nf, k). Apply per-grain pp blocks to p-rows, zero s-rows.
        out = X.new_zeros(X.shape)
        xp = X[..., :P, :].reshape(*X.shape[:-2], N, np_, X.shape[-1])
        out[..., :P, :] = torch.matmul(Dg_blocks, xp).reshape(
            *X.shape[:-2], P, X.shape[-1]
        )
        return out

    def bd_right(Dg_blocks: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        # (U V^T) @ BD = U (BD^T V)^T ; here X plays the role of V (..., nf, k).
        out = X.new_zeros(X.shape)
        xp = X[..., :P, :].reshape(*X.shape[:-2], N, np_, X.shape[-1])
        out[..., :P, :] = torch.matmul(Dg_blocks.transpose(-1, -2), xp).reshape(
            *X.shape[:-2], P, X.shape[-1]
        )
        return out

    mid = torch.matmul(M1.V.transpose(-1, -2), M2.U)  # (..., r1, r2)
    U = torch.cat([bd_left(M1.Dg, M2.U), M1.U, torch.matmul(M1.U, mid)], dim=-1)
    V = torch.cat([M2.V, bd_right(M2.Dg, M1.V), M2.V], dim=-1)
    Uc, Vc = _carrier_recompress(U, V, tol=tol, fixed_rank=fixed_rank)
    return _FlatCarrier(Dg, Uc, Vc, np_, ns, N)


def _carrier_neg(M: _FlatCarrier) -> _FlatCarrier:
    return _FlatCarrier(-M.Dg, -M.U, M.V, M.np_, M.ns, M.ngrain)


def _carrier_time_slice(M: _FlatCarrier, sl: slice) -> _FlatCarrier:
    """Slice a carrier on the time axis (axis 1 under the ``(W, nblk, B, ...)``
    leading dims)."""
    return _FlatCarrier(M.Dg[:, sl], M.U[:, sl], M.V[:, sl], M.np_, M.ns, M.ngrain)


def _carrier_pad_rank(M: _FlatCarrier, r: int) -> _FlatCarrier:
    """Zero-pad a carrier's low-rank inner dim up to ``r`` so two carriers with
    different ranks can be stored in one tensor after a scatter assignment."""
    cur = M.U.shape[-1]
    if cur >= r:
        return M
    padU = M.U.new_zeros(*M.U.shape[:-1], r - cur)
    padV = M.V.new_zeros(*M.V.shape[:-1], r - cur)
    return _FlatCarrier(
        M.Dg,
        torch.cat([M.U, padU], -1),
        torch.cat([M.V, padV], -1),
        M.np_,
        M.ns,
        M.ngrain,
    )


def _carrier_cyclic_shift(M: _FlatCarrier, level: int) -> _FlatCarrier:
    """Apply the Dense PCR cyclic shift to each carrier tensor (Dg, U, V)."""
    return _FlatCarrier(
        _dense_pcr_cyclic_shift(M.Dg, level),
        _dense_pcr_cyclic_shift(M.U, level),
        _dense_pcr_cyclic_shift(M.V, level),
        M.np_,
        M.ns,
        M.ngrain,
    )


def _carrier_from_sub_am(am: "AssembledMatrix") -> _FlatCarrier:
    """Build a carrier from a block-diagonal-in-groups subdiagonal AM (Jn).

    Jn_pp is the per-grain pp block-diagonal; Jn_ss / Jn_ps / Jn_sp (when
    present) are placed in the global low-rank handle. The handle starts as the
    exact dense remainder restricted to s-rows/cols and cross-grain, then is
    compressed; on Taylor data this is rank ~1.
    """
    row_l = am.row_layout
    if row_l.istr(0) == AxisLayout.IStructure.BLOCK:
        p, s = 0, 1
    elif row_l.istr(1) == AxisLayout.IStructure.BLOCK:
        p, s = 1, 0
    else:
        raise NotImplementedError("_carrier_from_sub_am requires one BLOCK group.")
    Jn_pp_t = am.tensors[p][p]
    Jn_pp = Jn_pp_t.torch()  # (..., N, np, np)
    lead = Jn_pp.shape[:-3]
    N = Jn_pp.shape[-3]
    np_ = Jn_pp.shape[-1]
    # The DENSE group's flat size:
    ns = _group_flat_size(am.row_layout, s)
    P = N * np_
    nf = P + ns
    Dg = Jn_pp.clone()
    # Build the global handle G = full(am) - blockdiag_pp(Dg) on the relevant
    # s-rows/cols and cross blocks. We materialize only the NON-pp-blockdiag
    # part directly from the AM blocks (no N^2 dense of the pp diagonal).
    dev, dt = Jn_pp.device, Jn_pp.dtype
    G = torch.zeros(*lead, nf, nf, dtype=dt, device=dev)
    # ps block (p-rows, s-cols)
    ps_t = am.tensors[p][s]
    if ps_t.defined():
        ps = ps_t.torch()  # (..., N, np, ns) or broadcast
        if ps.shape[-3] != N:
            ps = ps.unsqueeze(-3).expand(*ps.shape[:-2], N, np_, ns)
        G[..., :P, P:] = ps.reshape(*lead, P, ns)
    sp_t = am.tensors[s][p]
    if sp_t.defined():
        sp = sp_t.torch()  # (..., N, ns, np) or broadcast
        if sp.shape[-3] != N:
            sp = sp.unsqueeze(-3).expand(*sp.shape[:-2], N, ns, np_)
        G[..., P:, :P] = sp.permute(*range(len(lead)), -2, -3, -1).reshape(*lead, ns, P)
    ss_t = am.tensors[s][s]
    if ss_t.defined():
        G[..., P:, P:] = ss_t.torch()
    # Low-rank factorization of G via SVD (G already has bounded rank).
    Ug, Sg, Vgh = torch.linalg.svd(G, full_matrices=False)
    if torch.compiler.is_compiling():
        # Avoid the value-dependent ``.item()`` rank pick under torch.compile
        # (it would graph-break). The handle G is nonzero ONLY on the ns s-rows
        # and ns s-cols (ps adds s-cols, sp adds s-rows, ss the corner), so its
        # rank is bounded by 2*ns regardless of values. Keep exactly that many
        # singular triplets — the surplus (zero) ones pad harmlessly and are
        # dropped by the next recompression. Structural -> a Python int.
        keep = min(2 * ns, nf)
    else:
        smax = Sg[..., :1].clamp_min(torch.finfo(Sg.dtype).tiny)
        keep = int((Sg > 1e-12 * smax).reshape(-1, Sg.shape[-1]).sum(-1).max().item())
        keep = max(keep, 1)
    s_sqrt = Sg[..., :keep].sqrt().unsqueeze(-2)
    U = Ug[..., :, :keep] * s_sqrt
    V = Vgh[..., :keep, :].transpose(-1, -2) * s_sqrt
    return _FlatCarrier(Dg, U, V, np_, ns, N)


class NEML2SchurPCRState(PCRState):
    """Structure-preserving O(N) multi-group Schur-PCR working state.

    Mirrors the Dense ``DensePCRState`` shape contract (a leading ``W`` tree-
    window dim prepended ahead of the ``nblk`` time dim so the cyclic shift can
    double ``W`` and halve ``nblk`` each level), but instead of LU-factored
    dense blocks it carries everything in structure-preserving carrier form:

      * ``ainv``    : :class:`_FlatCarrier` for the per-step diagonal A^{-1}
                      (``blockdiag_pp(App^{-1}) + rank-ns`` global handle; A is
                      never modified by PCR, only reindexed by the cyclic shift).
      * ``b``       : :class:`_FlatCarrier` for the per-step subdiagonal Jn,
                      updated each level by ``b' = -B A^{-1} B`` in carrier form.
      * ``v``       : flat RHS ``(W, nblk, B, nf, 1)``.

    The carrier ``Dg`` / ``U`` / ``V`` tensors and ``v`` all carry the
    ``(W, nblk, B)`` leading dims and are sliced ``[:, :-1]`` / ``[:, 1:]`` /
    ``[:, 1:-1]`` on the time axis during reduction, then cyclic-shifted
    (axis 0/1) exactly like the Dense path.
    """

    def __init__(
        self,
        ainv: "_FlatCarrier",
        b: "_FlatCarrier",
        v: torch.Tensor,
        layout: "AxisLayout",
        tol: float,
        fixed_rank: int = 0,
    ) -> None:
        self.ainv = ainv
        self.b = b
        self.v = v
        self.layout = layout
        self.tol = tol
        self.fixed_rank = fixed_rank


class MultiGroupPCRState(PCRState):
    """Multi-group PCR state for systems with cross-block A.

    **Design choice (load-bearing):** when A has cross-group coupling (e.g.
    Taylor mix-mode's ``A_ps`` / ``A_sp`` from ``MixedControlSetup``), the
    per-group PCR shortcut is mathematically invalid, AND building a
    Schur-aware matrix solve that's compatible with PCR's reduction
    ``B A^{-1} B`` is a deeper algebra problem (NEML2's ``mm`` semantics
    apply ``intmd_sum`` on BLOCK contractions, which doesn't compose
    cleanly with a per-instance Schur solve for matrix RHS — the resulting
    X does NOT satisfy ``NEML2.mm(A, X) == B``).

    So for PCR on multi-group cross-block systems, we **delegate to
    Dense PCR on the flat-materialized operators**. Per time step, we
    materialize A_t and B_t via :func:`_am_to_flat` into a single
    ``(nblk, B, n_flat, n_flat)`` torch tensor, then run pyzag's
    ``DenseBlockOperator`` PCR routines. Correctness is anchored on
    Dense PCR (already validated against Dense Thomas on the flat). The
    final per-step solve in ``BidiagonalPCRFactorization.matvec`` still
    goes through ``NEML2SolvableBlockOperator.solve`` (i.e. Schur per
    step), which doesn't use this state.

    The state wraps a ``DensePCRState`` together with the metadata needed
    to convert results back to NEML2BlockVector / Operator at finalize.
    """

    def __init__(
        self,
        dense_state: DensePCRState,
        layout: "AxisLayout",
        intmd_dims_per_group: list[int],
        intmd_sizes_per_group: list[list[int]],
        B_template: "AssembledMatrix | None" = None,
    ) -> None:
        self.dense_state = dense_state
        self.layout = layout
        self.intmd_dims_per_group = intmd_dims_per_group
        self.intmd_sizes_per_group = intmd_sizes_per_group
        # Template AM of the original input B, used by ``pcr_finalize`` to
        # build a shape-compatible (but value-trivial) B_red so the caller's
        # ``B[s+1:e] = B_red`` assignment works.
        self.B_template = B_template


class NEML2PCRState(PCRState):
    """PCR working state: a list of DensePCRState, one per diagonal group.

    Each diagonal group's bidiagonal-in-time system runs Dense PCR independently
    (per-group PCR is only valid when the subdiagonal Jn is block-diagonal in
    groups — see runtime check in NEML2SolvableBlockOperator.pcr_init).

    The ``intmd_dims_per_group`` field records the runtime intmd-dim count
    per group at the moment ``pcr_init`` folded those dims into the batch
    axis. ``pcr_finalize`` reads it back to restore the original layout —
    we cannot rely on ``layout.intmd_sizes`` because NEML2-native layouts
    often report intmd=() even when the runtime tensors carry per-grain
    (or other) intmd structure.
    """

    def __init__(
        self,
        per_group: list[DensePCRState],
        layout: "AxisLayout",
        intmd_dims_per_group: list[int],
        intmd_sizes_per_group: list[list[int]],
    ) -> None:
        self.per_group = per_group
        self.layout = layout
        self.intmd_dims_per_group = intmd_dims_per_group
        self.intmd_sizes_per_group = intmd_sizes_per_group
