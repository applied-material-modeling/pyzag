import sys

sys.path.append("../..")

import torch

from pyzag.neml2 import model
import neml2

if __name__ == "__main__":
    fname = "model.i"
    mname = "implicit_rate"

    nbatch = 20
    ntime = 100

    end_time = torch.logspace(-1, 5, nbatch)
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

    stress = (
        torch.stack(
            [
                torch.linspace(0, 300, ntime),
                torch.linspace(0, -10, ntime),
                torch.linspace(0, -5, ntime),
                torch.zeros(ntime),
                torch.zeros(ntime),
                torch.zeros(ntime),
            ]
        )
        .T[:, None]
        .expand(-1, nbatch, -1)
    )

    plastic_strain = torch.full((ntime, nbatch), 0.1).unsqueeze(-1)

    nmodel = neml2.load_model(fname, mname)
    pmodel = model.NEML2Model(nmodel)

    state = pmodel.collect_state({"S": stress, "internal/ep": plastic_strain})
    forces = pmodel.collect_forces({"t": time, "E": strain})

    print(pmodel.forward(state, forces))
