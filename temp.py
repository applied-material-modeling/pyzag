#!/usr/bin/env python3

from pyzag import ode, nonlinear

from pyoptmat import ode as pode

import torch

# Ensure test consistency
torch.manual_seed(42)

import unittest

torch.set_default_dtype(torch.float64)


class LinearSystem(torch.nn.Module):
    """Linear system of equations"""

    def __init__(self, n):
        super().__init__()
        self.n = n
        Ap = torch.rand((n, n))
        self.A = torch.nn.Parameter(Ap.transpose(0, 1) * Ap)

        self.vals, self.vecs = torch.linalg.eigh(self.A)

    def forward(self, t, y):
        return torch.matmul(self.A.unsqueeze(0).unsqueeze(0), y.unsqueeze(-1)).squeeze(
            -1
        ), self.A.unsqueeze(0).unsqueeze(0).expand(
            t.shape[0], t.shape[1], self.n, self.n
        )

    def y0(self, nbatch):
        return torch.rand((nbatch, self.n))


if __name__ == "__main__":
    n = 4
    nbatch = 1
    ntime = 100
    nchunk = 1

    sec = LinearSystem(n)
    model = ode.ForwardEulerODE(sec)
    y0 = sec.y0(nbatch)

    times = torch.linspace(0, 1, ntime).unsqueeze(-1).expand(-1, nbatch).unsqueeze(-1)

    arg = pode.odeint_adjoint(
        sec, y0, times.squeeze(-1), block_size=nchunk, method="forward-euler"
    )
    sigh = torch.linalg.norm(arg)
    sigh.backward()
    print("ODE")
    print(sigh)
    print(sec.A.grad)

    model.zero_grad()

    solver = nonlinear.RecursiveNonlinearEquationSolver(
        model,
        y0,
        step_generator=nonlinear.StepGenerator(nchunk),
    )

    res = solver.solve(ntime, times)
    val = torch.linalg.norm(res)
    print("AD")
    print(val)
    val.backward()

    first = solver.func.ode.A.grad
    print(first)

    solver.zero_grad()
    res2 = nonlinear.solve_adjoint(solver, ntime, times)
    val2 = torch.linalg.norm(res2)
    print("Adjoint")
    print(val2)
    val2.backward()

    second = solver.func.ode.A.grad
    print(second)
