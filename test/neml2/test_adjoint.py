"""Test adjoint correctness"""

import unittest
import os.path

import torch

torch.set_default_dtype(torch.double)

# Selectively enable
try:
    from pyzag.neml2 import model
    import neml2
except ImportError:
    raise unittest.SkipTest("NEML 2 not installed")

from pyzag import nonlinear


class DerivativeCheck:
    def adjoint_grads(self):
        solver = nonlinear.RecursiveNonlinearEquationSolver(
            self.pmodel,
            step_generator=nonlinear.StepGenerator(self.nchunk),
            predictor=nonlinear.PreviousStepsPredictor(),
        )
        solver.zero_grad()
        res = nonlinear.solve_adjoint(
            solver, self.initial_state.detach().clone(), len(self.forces), self.forces
        )
        val = torch.norm(res)
        val.backward()

        return {n: p.grad.detach().clone() for n, p in solver.named_parameters()}

    def fd_grads(self, eps=1.0e-6):
        solver = nonlinear.RecursiveNonlinearEquationSolver(
            self.pmodel,
            step_generator=nonlinear.StepGenerator(self.nchunk),
            predictor=nonlinear.PreviousStepsPredictor(),
        )
        res = {}
        with torch.no_grad():
            val0 = torch.norm(
                nonlinear.solve(
                    solver,
                    self.initial_state.detach().clone(),
                    len(self.forces),
                    self.forces,
                )
            )
            for n, p in solver.named_parameters():
                p0 = p.clone()
                dx = torch.abs(p0) * eps
                p.data = p0 + dx
                val1 = torch.norm(
                    nonlinear.solve(
                        solver,
                        self.initial_state.detach().clone(),
                        len(self.forces),
                        self.forces,
                    )
                )
                res[n] = (val1 - val0) / dx
                p.data = p0

        return res

    def test_adjoint_vs_fd(self):
        grads_adjoint = self.adjoint_grads()
        grads_fd = self.fd_grads()

        self.assertEqual(grads_adjoint.keys(), grads_fd.keys())

        for n in grads_adjoint.keys():
            self.assertTrue(torch.allclose(grads_adjoint[n], grads_fd[n]))


class TestElasticModel(unittest.TestCase, DerivativeCheck):
    def setUp(self):
        self.nmodel = neml2.load_model(
            os.path.join(os.path.dirname(__file__), "elastic_model.i"), "implicit_rate"
        )
        self.pmodel = model.NEML2Model(self.nmodel)

        self.nbatch = 20
        self.ntime = 100

        end_time = torch.logspace(-1, -5, self.nbatch)
        time = torch.stack(
            [torch.linspace(0, et, self.ntime) for et in end_time]
        ).T.unsqueeze(-1)
        strain = (
            torch.stack(
                [
                    torch.linspace(0, 0.1, self.ntime),
                    torch.linspace(0, -0.05, self.ntime),
                    torch.linspace(0, -0.05, self.ntime),
                    torch.zeros(self.ntime),
                    torch.zeros(self.ntime),
                    torch.zeros(self.ntime),
                ]
            )
            .T[:, None]
            .expand(-1, self.nbatch, -1)
        )

        self.initial_state = torch.zeros((self.nbatch, 6))
        self.forces = self.pmodel.collect_forces({"t": time, "E": strain})

        self.nchunk = 10


class TestViscoplasticModel(unittest.TestCase, DerivativeCheck):
    def setUp(self):
        self.nmodel = neml2.load_model(
            os.path.join(os.path.dirname(__file__), "viscoplastic_model.i"),
            "implicit_rate",
        )
        self.pmodel = model.NEML2Model(self.nmodel)

        self.nbatch = 20
        self.ntime = 100

        end_time = torch.logspace(-1, -5, self.nbatch)
        time = torch.stack(
            [torch.linspace(0, et, self.ntime) for et in end_time]
        ).T.unsqueeze(-1)
        strain = (
            torch.stack(
                [
                    torch.linspace(0, 0.1, self.ntime),
                    torch.linspace(0, -0.05, self.ntime),
                    torch.linspace(0, -0.05, self.ntime),
                    torch.zeros(self.ntime),
                    torch.zeros(self.ntime),
                    torch.zeros(self.ntime),
                ]
            )
            .T[:, None]
            .expand(-1, self.nbatch, -1)
        )

        self.initial_state = torch.zeros((self.nbatch, 7))
        self.forces = self.pmodel.collect_forces({"t": time, "E": strain})

        self.nchunk = 10


class TestComplexModel(unittest.TestCase, DerivativeCheck):
    def setUp(self):
        self.nmodel = neml2.load_model(
            os.path.join(os.path.dirname(__file__), "complex_model.i"),
            "model",
        )
        self.pmodel = model.NEML2Model(
            self.nmodel, exclude_parameters=["yield_zero.sy"]
        )

        self.nbatch = 20
        self.ntime = 100

        end_time = torch.logspace(-1, -5, self.nbatch)
        time = torch.stack(
            [torch.linspace(0, et, self.ntime) for et in end_time]
        ).T.unsqueeze(-1)
        conditions = (
            torch.stack(
                [
                    torch.linspace(0, 0.1, self.ntime),
                    torch.linspace(0, -50, self.ntime),
                    torch.linspace(0, -0.025, self.ntime),
                    torch.linspace(0, 0.15, self.ntime),
                    torch.linspace(0, 75.0, self.ntime),
                    torch.linspace(0, 0.05, self.ntime),
                ]
            )
            .T[:, None]
            .expand(-1, self.nbatch, -1)
        )

        control = torch.zeros((self.ntime, self.nbatch, 6))
        control[..., 1] = 1.0
        control[..., 4] = 1.0

        temperatures = torch.stack(
            [
                torch.linspace(T1, T2, self.ntime)
                for T1, T2 in zip(
                    torch.linspace(300, 500, self.nbatch),
                    torch.linspace(600, 1200, self.nbatch),
                )
            ]
        ).T.unsqueeze(-1)

        self.initial_state = torch.zeros((self.nbatch, 8))
        self.forces = self.pmodel.collect_forces(
            {
                "t": time,
                "control": control,
                "fixed_values": conditions,
                "T": temperatures,
            }
        )

        self.nchunk = 10
