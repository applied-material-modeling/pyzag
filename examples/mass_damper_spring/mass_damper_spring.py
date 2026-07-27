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

"""Standalone pyzag example: a mass-damper-spring chain.

Defines a stiff mass-damper-spring ODE as a plain ``torch.nn.Module``, runs a
batched forward solve and plots it, then compares parameter gradients from
pyzag's adjoint method, full-graph autograd, and finite differences (wall time
and accuracy). Edit the config constants below and re-run; no CLI arguments.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from pyzag.nonlinear import (
    RecursiveNonlinearEquationSolver,
    StepGenerator,
    solve,
    solve_adjoint,
)
from pyzag.ode import BackwardEulerODE

NCHUNK = 50
NBATCH = 4
NTIME = 4000
HALF_SIZE = 3
FD_EPS = 1e-5
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


class MassDamperSpring(torch.nn.Module):
    """Mass-spring-damper chain reduced to a first-order system.

    State is ``[displacements, velocities]`` so ``size = 2 * half_size``, with
    stiffness ``K``, damping ``C`` and mass ``M`` as the learnable parameters.
    """

    def __init__(
        self,
        half_size,
        K_range=(1.0, 5.0),
        C_range=(0.1, 0.5),
        M_range=(0.5, 1.0),
        t_max=10.0,
        force_mag=1.0,
        force_period_range=(0.5, 2.0),
    ):
        super().__init__()
        self.half_size = half_size
        self.size = half_size * 2
        self.K = torch.nn.Parameter(torch.linspace(*K_range, half_size))
        self.C = torch.nn.Parameter(torch.linspace(*C_range, half_size))
        self.M = torch.nn.Parameter(torch.linspace(*M_range, half_size))
        self.t_max = t_max
        self.force_mag = force_mag
        self.force_period_range = force_period_range

    def setup(self, nbatch, ntime):
        time_t = (
            torch.linspace(0, self.t_max, ntime).unsqueeze(-1).expand(ntime, nbatch)
        )
        y0 = torch.zeros(nbatch, self.size)
        return time_t, y0

    def force(self, t):
        return self.force_mag * torch.sin(
            2.0
            * torch.pi
            / torch.linspace(*self.force_period_range, t.shape[-1], device=t.device)
            * t
        )

    def forward(self, t, y):
        if t.dim() == y.dim():
            t = t.squeeze(-1)
        f = self.force(t)
        return self.rate(t, y, f), self.jacobian(t, y, f)

    def rate(self, t, y, force):
        ydot = torch.zeros_like(y)
        hs = self.half_size
        u = y[..., :hs]
        v = y[..., hs:]
        du = torch.diff(u, dim=-1)
        dv = torch.diff(v, dim=-1)
        ydot[..., :hs] = v
        ydot[..., hs:-1] += self.K[..., :-1] * du / self.M[..., :-1]
        ydot[..., hs + 1 :] += -self.K[..., :-1] * du / self.M[..., 1:]
        ydot[..., -1] += -self.K[..., -1] * u[..., -1] / self.M[..., -1]
        ydot[..., hs] += force / self.M[..., 0]
        ydot[..., hs:-1] += self.C[..., :-1] * dv / self.M[..., :-1]
        ydot[..., hs + 1 :] += -self.C[..., :-1] * dv / self.M[..., 1:]
        ydot[..., -1] += -self.C[..., -1] * v[..., -1] / self.M[..., -1]
        return ydot

    def jacobian(self, t, y, force):
        J = torch.zeros(y.shape + (self.size,), device=t.device)
        hs = self.half_size
        J[..., :hs, hs:] += torch.eye(hs, device=t.device)
        J[..., hs:, :hs] += torch.diag_embed(-self.K / self.M)
        J[..., hs + 1 :, 1:hs] += torch.diag_embed(-self.K[..., :-1] / self.M[..., 1:])
        J[..., hs:, :hs] += torch.diag_embed(
            self.K[..., :-1] / self.M[..., :-1], offset=1
        )
        J[..., hs:, :hs] += torch.diag_embed(
            self.K[..., :-1] / self.M[..., 1:], offset=-1
        )
        J[..., hs:, hs:] += torch.diag_embed(-self.C / self.M)
        J[..., hs + 1 :, hs + 1 :] += torch.diag_embed(
            -self.C[..., :-1] / self.M[..., 1:]
        )
        J[..., hs:, hs:] += torch.diag_embed(
            self.C[..., :-1] / self.M[..., :-1], offset=1
        )
        J[..., hs:, hs:] += torch.diag_embed(
            self.C[..., :-1] / self.M[..., 1:], offset=-1
        )
        return J


def build():
    problem = MassDamperSpring(half_size=HALF_SIZE).to(device=DEVICE, dtype=DTYPE)
    solver = RecursiveNonlinearEquationSolver(
        BackwardEulerODE(problem), step_generator=StepGenerator(block_size=NCHUNK)
    ).to(device=DEVICE, dtype=DTYPE)
    time_t, y0 = problem.setup(NBATCH, NTIME)
    time_t = time_t.unsqueeze(-1).to(device=DEVICE, dtype=DTYPE)
    y0 = y0.to(device=DEVICE, dtype=DTYPE)
    return problem, solver, time_t, y0


def _sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def timed_gradient(method, solver, y0, time_t, params):
    """Return ``(wall_seconds, grads)`` for ``method`` in {"adjoint", "ad"}."""
    for p in params:
        p.grad = None
    _sync()
    t0 = time.perf_counter()
    driver = solve_adjoint if method == "adjoint" else solve
    driver(solver, y0, NTIME, time_t).pow(2).sum().backward()
    _sync()
    return time.perf_counter() - t0, [p.grad.detach().clone() for p in params]


def fd_compare(solver, y0, time_t, params):
    """Central-difference gradient wrt the first entry of each parameter.

    Returns ``(wall_seconds, {param_index: (fd_value, entry_index)})``. Only one
    entry per parameter is probed, because a full finite-difference gradient
    costs O(n_params) forward solves -- exactly what the adjoint avoids.
    """
    out = {}
    _sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        for pi, p in enumerate(params):
            flat = p.view(-1)
            orig = flat[0].item()
            flat[0] = orig + FD_EPS
            lp = solve(solver, y0, NTIME, time_t).pow(2).sum().item()
            flat[0] = orig - FD_EPS
            lm = solve(solver, y0, NTIME, time_t).pow(2).sum().item()
            flat[0] = orig
            out[pi] = ((lp - lm) / (2.0 * FD_EPS), 0)
    _sync()
    return time.perf_counter() - t0, out


def plot_trajectory(y, time_t, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    t_np = time_t[:, 0, 0].cpu().numpy()
    for b in range(min(NBATCH, 3)):
        ax.plot(t_np, y[:, b, 0].detach().cpu().numpy(), label=f"param set {b}")
    ax.set_xlabel("time")
    ax.set_ylabel("displacement[0]")
    ax.set_title("Mass-damper-spring forward solve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    torch.manual_seed(SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    torch.set_default_dtype(DTYPE)

    problem, solver, time_t, y0 = build()
    params = list(problem.parameters())
    print(
        f"device={DEVICE}, dtype={DTYPE}, ntime={NTIME}, nchunk={NCHUNK}, nbatch={NBATCH}"
    )

    out_png = Path(__file__).resolve().parent / "mass_damper_spring.png"
    with torch.no_grad():
        y_traj = solve(solver, y0, NTIME, time_t)
    plot_trajectory(y_traj, time_t, out_png)
    print(f"forward trajectory saved to {out_png}")

    timed_gradient("adjoint", solver, y0, time_t, params)
    timed_gradient("ad", solver, y0, time_t, params)

    t_adj, g_adj = timed_gradient("adjoint", solver, y0, time_t, params)
    t_ad, g_ad = timed_gradient("ad", solver, y0, time_t, params)
    t_fd, fd = fd_compare(solver, y0, time_t, params)

    scale = max(a.abs().max().item() for a in g_adj) or 1.0
    max_adj_ad = max((a - b).abs().max().item() for a, b in zip(g_adj, g_ad))
    max_adj_fd = max(
        abs(g_adj[pi].view(-1)[idx].item() - val) for pi, (val, idx) in fd.items()
    )

    print()
    print(f"{'method':<12}{'time (s)':>12}")
    print("-" * 24)
    print(f"{'adjoint':<12}{t_adj:>12.4f}")
    print(f"{'autograd':<12}{t_ad:>12.4f}")
    print(f"{'finite-diff':<12}{t_fd:>12.4f}")
    print()
    print(
        f"max |adjoint - autograd|    = {max_adj_ad:.3e}  (relative {max_adj_ad / scale:.3e})"
    )
    print(
        f"max |adjoint - finite-diff| = {max_adj_fd:.3e}  (relative {max_adj_fd / scale:.3e})"
    )


if __name__ == "__main__":
    main()
