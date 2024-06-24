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
    pmodel = model.NEML2Model(
        nmodel, exclude_parameters=["elasticity.E", "elasticity.nu"]
    )

    pmodel.yieldaabbsy.data = torch.tensor(100.0)

    # There is NEML2Model.collect_state, but come on...
    initial_state = torch.zeros((nbatch, 7))
    forces = pmodel.collect_forces({"t": time, "E": strain})

    F = forces[:10]
    S = torch.rand_like(F)

    print("BEFORE")
    print(nmodel.named_parameters()["flow_rate.eta"].requires_grad)
    print(pmodel.flow_rateaabbeta.requires_grad)

    pmodel.flow_rateaabbeta.data = torch.tensor(150.0)

    print("BEFORE SET")
    print(pmodel.flow_rateaabbeta)
    print(nmodel.named_parameters()["flow_rate.eta"].tensor().tensor())

    y, J = pmodel(S, F)

    print("AFTER")
    print(pmodel.flow_rateaabbeta)
    print(nmodel.named_parameters()["flow_rate.eta"].tensor().tensor())

    print(pmodel.flow_rateaabbeta.requires_grad)
    print(nmodel.named_parameters()["flow_rate.eta"].requires_grad)
    print(y.requires_grad)
    print(J.requires_grad)

    test = torch.norm(y)
    test.backward()

    print(nmodel.named_parameters()["flow_rate.eta"].grad)
    print(pmodel.flow_rateaabbeta.grad)

    sys.exit()

    solver = nonlinear.RecursiveNonlinearEquationSolver(
        pmodel,
        initial_state,
        step_generator=nonlinear.StepGenerator(10),
        predictor=nonlinear.PreviousStepsPredictor(),
    )
    res = solver.solve(ntime, forces)

    print(pmodel.yieldaabbsy)
    print(nmodel.named_parameters()["yield.sy"].tensor().tensor())

    whatever = torch.norm(res)
    print(whatever)
    whatever.backward()

    print(pmodel.yieldaabbsy)

    plt.plot(strain[:, 0, 0], res[:, 0, 0].detach().numpy())
    plt.plot(strain[:, -1, 0], res[:, -1, 0].detach().numpy())
    plt.show()
