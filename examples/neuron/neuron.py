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

"""Standalone pyzag example: a network of coupled neurons.

Defines a Hodgkin-Huxley-style coupled-neuron ODE as a plain
``torch.nn.Module`` returning ``(rate, jacobian)``, runs a batched forward solve
and plots the membrane voltages, then compares parameter gradients from pyzag's
adjoint method, full-graph autograd, and finite differences (wall time and
accuracy). Edit the config constants below and re-run; no CLI arguments.
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
NTIME = 8000
NNEURONS = 2
FD_EPS = 1e-5
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


class Neuron(torch.nn.Module):
    """Coupled-neuron oscillator model (Schwemmer and Lewis, 2012).

    Each neuron carries four states (V, m, h, n); the full system size is
    ``4 * nneurons``. All conductances, reversal potentials and gating
    parameters are learnable.
    """

    def __init__(
        self,
        nneurons,
        C_range=(0.1, 1.0),
        g_Na_range=(0.1, 1.0),
        E_Na_range=(0.1, 1.0),
        g_K_range=(0.1, 1.0),
        E_K_range=(0.1, 1.0),
        g_L_range=(0.1, 1.0),
        E_L_range=(0.1, 1.0),
        m_inf_range=(0.1, 1.0),
        tau_m_range=(0.5, 5.0),
        h_inf_range=(0.1, 1.0),
        tau_h_range=(1.5, 15.0),
        n_inf_range=(0.1, 1.0),
        tau_n_range=(1.0, 10.0),
        g_C_range=(0.001, 0.01),
        t_max=10.0,
        I_max_range=(0.1, 1.0),
        I_period_range=(0.5, 2.0),
    ):
        super().__init__()
        self.nneurons = nneurons
        self.size = 4 * nneurons
        self.C = torch.nn.Parameter(torch.linspace(*C_range, nneurons))
        self.g_Na = torch.nn.Parameter(torch.linspace(*g_Na_range, nneurons))
        self.E_Na = torch.nn.Parameter(torch.linspace(*E_Na_range, nneurons))
        self.g_K = torch.nn.Parameter(torch.linspace(*g_K_range, nneurons))
        self.E_K = torch.nn.Parameter(torch.linspace(*E_K_range, nneurons))
        self.g_L = torch.nn.Parameter(torch.linspace(*g_L_range, nneurons))
        self.E_L = torch.nn.Parameter(torch.linspace(*E_L_range, nneurons))
        self.m_inf = torch.nn.Parameter(torch.linspace(*m_inf_range, nneurons))
        self.tau_m = torch.nn.Parameter(torch.linspace(*tau_m_range, nneurons))
        self.h_inf = torch.nn.Parameter(torch.linspace(*h_inf_range, nneurons))
        self.tau_h = torch.nn.Parameter(torch.linspace(*tau_h_range, nneurons))
        self.n_inf = torch.nn.Parameter(torch.linspace(*n_inf_range, nneurons))
        self.tau_n = torch.nn.Parameter(torch.linspace(*tau_n_range, nneurons))
        self.g_C = torch.nn.Parameter(torch.linspace(*g_C_range, nneurons))
        self.t_max = t_max
        self.I_max_range = I_max_range
        self.I_period_range = I_period_range

    def setup(self, nbatch, ntime):
        time_t = (
            torch.linspace(0, self.t_max, ntime).unsqueeze(-1).expand(ntime, nbatch)
        )
        y0 = torch.zeros(nbatch, self.size)
        return time_t, y0

    def force(self, t):
        return torch.linspace(
            *self.I_max_range, t.shape[-1], device=t.device
        ) * torch.sin(
            2.0
            * torch.pi
            / torch.linspace(*self.I_period_range, t.shape[-1], device=t.device)
            * t
        )

    def forward(self, t, y):
        if t.dim() == y.dim():
            t = t.squeeze(-1)
        f = self.force(t)
        return self.rate(t, y, f), self.jacobian(t, y, f)

    def rate(self, t, y, force):
        V = y[..., 0::4]
        m = y[..., 1::4]
        h = y[..., 2::4]
        n = y[..., 3::4]
        ydot = torch.zeros_like(y)
        ydot[..., 0::4] = (
            1.0
            / self.C[None, ...]
            * (
                -self.g_Na[None, ...] * m**3.0 * h * (V - self.E_Na[None, ...])
                - self.g_K[None, ...] * n**4.0 * (V - self.E_K[None, ...])
                - self.g_L[None, ...] * (V - self.E_L[None, ...])
                + force[..., None]
            )
        )
        dV = torch.sum(
            self.g_C[None, ...]
            * (V[..., :, None] - V[..., None, :])
            / self.C[None, ...],
            dim=-1,
        )
        ydot[..., 0::4] += dV
        ydot[..., 1::4] = (self.m_inf - m) / self.tau_m
        ydot[..., 2::4] = (self.h_inf - h) / self.tau_h
        ydot[..., 3::4] = (self.n_inf - n) / self.tau_n
        return ydot

    def jacobian(self, t, y, force):
        J = torch.zeros(y.shape + y.shape[-1:], device=y.device)
        V = y[..., 0::4]
        m = y[..., 1::4]
        h = y[..., 2::4]
        n = y[..., 3::4]
        J[..., 0::4, 0::4] = torch.diag_embed(
            1.0
            / self.C[None, ...]
            * (
                -self.g_L[None, ...]
                - self.g_Na[None, ...] * h * m**3.0
                - self.g_K[None, ...] * n**4.0
            )
        )
        J[..., 0::4, 0::4] -= self.g_C[None, ...] / self.C[None, ...]
        J[..., 0::4, 0::4] += torch.eye(self.nneurons, device=y.device).expand(
            *y.shape[:-1], -1, -1
        ) * torch.sum(self.g_C / self.C)
        J[..., 0::4, 1::4] = torch.diag_embed(
            -1.0
            / self.C[None, ...]
            * (3.0 * self.g_Na[None, ...] * h * m**2.0 * (-self.E_Na[None, ...] + V))
        )
        J[..., 0::4, 2::4] = torch.diag_embed(
            -1.0
            / self.C[None, ...]
            * (self.g_Na[None, ...] * m**3.0 * (-self.E_Na[None, ...] + V))
        )
        J[..., 0::4, 3::4] = torch.diag_embed(
            -1.0
            / self.C[None, ...]
            * (4.0 * self.g_K[None, ...] * n**3.0 * (-self.E_K[None, ...] + V))
        )
        J[..., 1::4, 1::4] = torch.diag(-1.0 / self.tau_m)
        J[..., 2::4, 2::4] = torch.diag(-1.0 / self.tau_h)
        J[..., 3::4, 3::4] = torch.diag(-1.0 / self.tau_n)
        return J


def build():
    problem = Neuron(nneurons=NNEURONS).to(device=DEVICE, dtype=DTYPE)
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
    ax.set_ylabel("V (neuron 0)")
    ax.set_title("Coupled-neuron forward solve")
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

    out_png = Path(__file__).resolve().parent / "neuron.png"
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
