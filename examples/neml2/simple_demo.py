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
    fname = "model.i"
    mname = "implicit_rate"

    # Problem dimensions
    nbatch = 20
    ntime = 100

    # I will need to double check these match the actual NEML2 data
    end_time = torch.logspace(-1, -5, nbatch)
    time = torch.stack([torch.linspace(0, et, ntime) for et in end_time]).T.unsqueeze(
        -1
    )
    strain = (
        torch.stack(
            [
                torch.linspace(0, 0.1, ntime),
                torch.linspace(0, -0.05, ntime),
                torch.linspace(0, -0.05, ntime),
                torch.zeros(ntime),
                torch.zeros(ntime),
                torch.zeros(ntime),
            ]
        )
        .T[:, None]
        .expand(-1, nbatch, -1)
    )

    nmodel = neml2.load_model(fname, mname)
    pmodel = model.NEML2Model(nmodel)

    pmodel.yield_sy.data = torch.tensor(100.0)

    # There is NEML2Model.collect_state, but come on...
    initial_state = torch.zeros((nbatch, 7))
    forces = pmodel.collect_forces({"t": time, "E": strain})

    solver = nonlinear.RecursiveNonlinearEquationSolver(
        pmodel,
        initial_state,
        step_generator=nonlinear.StepGenerator(10),
        predictor=nonlinear.PreviousStepsPredictor(),
    )
    # Uncomment this line to use non-adjoint
    # res = solver.solve(ntime, forces)
    res = nonlinear.solve_adjoint(solver, ntime, forces)

    whatever = torch.norm(res)
    whatever.backward()

    print([(n, p.grad) for n, p in pmodel.named_parameters()])

    output = pmodel.extract_state(res)

    plt.plot(strain[:, :, 0], output["S"][:, :, 0].detach().numpy())
    plt.xlabel("Strain")
    plt.ylabel("Stress")
    plt.show()

    plt.plot(strain[:, :, 0], output["internal_ep"][:, :, 0].detach().numpy())
    plt.xlabel("Strain")
    plt.ylabel("Plastic strain")
    plt.show()
