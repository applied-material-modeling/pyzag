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

"""Standalone pyzag example: a neural-network ODE.

Defines an ODE whose right-hand side is a small linear+tanh network with a
hand-derived analytical Jacobian, runs a batched forward solve and plots it,
then compares parameter gradients from pyzag's adjoint method, full-graph
autograd, and finite differences (wall time and accuracy). Edit the config
constants below and re-run; no CLI arguments.
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
N = 4
NLAYERS = 2
FD_EPS = 1e-5
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def _mbmm(A1, A2):
    return torch.einsum("...ik,...kj->...ij", A1, A2)


def _dtanh(x):
    return 1.0 - torch.tanh(x) ** 2.0


class LinearNetwork(torch.nn.Module):
    """ODE whose rate is a linear+tanh network with an analytical Jacobian.

    State size is ``n``; the network consumes ``[state, force]`` and the learnable
    parameters are the layer weights and biases.
    """

    def __init__(
        self, n, nlayers=3, t_max=1.0, force_mag=1.0, force_period_range=(1.0e-2, 1.0)
    ):
        super().__init__()
        self.n = n
        self.size = n + 1
        self.nlayers = nlayers
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(self.size, self.size) for _ in range(nlayers)]
            + [torch.nn.Linear(self.size, n)]
        )
        self.activation = torch.nn.Tanh()
        self.dactivation = _dtanh
        self.t_max = t_max
        self.force_mag = force_mag
        self.force_period_range = force_period_range

    def setup(self, nbatch, ntime):
        time_t = (
            torch.linspace(0, self.t_max, ntime).unsqueeze(-1).expand(ntime, nbatch)
        )
        y0 = torch.zeros(nbatch, self.n)
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
        x = torch.cat([y, force.unsqueeze(-1)], dim=-1)
        for l in self.layers:
            x = self.activation(l(x))
        return x

    def jacobian(self, t, y, force):
        x = torch.cat([y, force.unsqueeze(-1)], dim=-1)
        full_shape = y.shape[:-1] + (self.size,)
        J = torch.diag_embed(
            torch.ones(full_shape, device=y.device, dtype=y.dtype), dim1=-1, dim2=-2
        )
        for l in self.layers:
            x = _mbmm(x.unsqueeze(-2), l.weight.transpose(-1, -2)).squeeze(-2) + l.bias
            J = _mbmm(
                _mbmm(J, l.weight.transpose(-1, -2)),
                torch.diag_embed(self.dactivation(x), dim1=-1, dim2=-2),
            )
            x = self.activation(x)
        return J[..., :-1, :].transpose(-1, -2)


def build():
    problem = LinearNetwork(n=N, nlayers=NLAYERS).to(device=DEVICE, dtype=DTYPE)
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
    ax.set_ylabel("state[0]")
    ax.set_title("Neural-network ODE forward solve")
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

    out_png = Path(__file__).resolve().parent / "linear_network.png"
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
