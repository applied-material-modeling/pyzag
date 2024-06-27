import sys

sys.path.append("../..")

import torch
import matplotlib.pyplot as plt

from pyzag import nonlinear
from pyzag.neml2 import model
import neml2

torch.set_default_dtype(torch.double)

if __name__ == "__main__":
    # Name to load from the input file
    fname = "complex_model.i"
    mname = "model"

    # Problem dimensions
    nbatch = 20
    ntime = 100

    # I will need to double check these match the actual NEML2 data
    end_time = torch.logspace(-1, -5, nbatch)
    time = torch.stack([torch.linspace(0, et, ntime) for et in end_time]).T.unsqueeze(
        -1
    )
    conditions = (
        torch.stack(
            [
                torch.linspace(0, 0.1, ntime),
                torch.linspace(0, -50, ntime),
                torch.linspace(0, -0.025, ntime),
                torch.linspace(0, 0.15, ntime),
                torch.linspace(0, 75.0, ntime),
                torch.linspace(0, 0.05, ntime),
            ]
        )
        .T[:, None]
        .expand(-1, nbatch, -1)
    )

    control = torch.zeros((ntime, nbatch, 6))
    control[..., 1] = 1.0
    control[..., 4] = 1.0

    temperatures = torch.stack(
        [
            torch.linspace(T1, T2, ntime)
            for T1, T2 in zip(
                torch.linspace(300, 500, nbatch),
                torch.linspace(600, 1200, nbatch),
            )
        ]
    ).T.unsqueeze(-1)

    nmodel = neml2.load_model(fname, mname)
    pmodel = model.NEML2Model(nmodel, exclude_parameters=["yield_zero.sy"])

    # There is NEML2Model.collect_state, but come on...
    initial_state = torch.zeros((nbatch, 8))
    forces = pmodel.collect_forces(
        {"t": time, "control": control, "fixed_values": conditions, "T": temperatures}
    )

    solver = nonlinear.RecursiveNonlinearEquationSolver(
        pmodel,
        initial_state,
        step_generator=nonlinear.StepGenerator(10),
        predictor=nonlinear.PreviousStepsPredictor(),
    )
    # Uncomment this line to use non-adjoint
    res = nonlinear.solve_adjoint(solver, ntime, forces)

    whatever = torch.norm(res)
    whatever.backward()

    print([(n, p.grad) for n, p in pmodel.named_parameters()])

    output = pmodel.extract_state(res)

    plt.plot(conditions[..., 0], output["mixed_state"].detach().numpy()[..., 0])
    plt.show()
