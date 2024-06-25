import torch
import neml2
from neml2.tensors import BatchTensor, LabeledVector

torch.set_default_dtype(torch.double)

if __name__ == "__main__":
    # Name to load from the input file
    fname = "model.i"
    mname = "implicit_rate"

    # Problem dimensions
    nbatch = 20

    with torch.autograd.set_detect_anomaly(True):
        # Name to load from the input file
        fname = "model.i"
        mname = "implicit_rate"

        # Problem dimensions
        nbatch = 20
        ntime = 100

        # I will need to double check these match the actual NEML2 data
        end_time = torch.logspace(-1, -5, nbatch)
        time = torch.stack(
            [torch.linspace(0, et, ntime) for et in end_time]
        ).T.unsqueeze(-1)
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

        print(nmodel.input_axis())
        print(nmodel.output_axis())

        p = nmodel.named_parameters()["yield.sy"]
        p.requires_grad_(True)

        old_state = torch.zeros((nbatch, 7))

        nmodel.reinit((nbatch,), 1)

        for i in range(1, ntime):
            state = old_state
            for j in range(2):
                input_state = torch.cat(
                    [
                        strain[i],
                        time[i],
                        strain[i - 1],
                        time[i - 1],
                        state,
                        torch.zeros_like(old_state),
                    ],
                    dim=-1,
                )

                test = BatchTensor(input_state, 1)

                R, J = nmodel.value_and_dvalue(
                    LabeledVector(BatchTensor(input_state, 1), [nmodel.input_axis()])
                )

                Rt = R.tensor().tensor()
                Jt = J.tensor().tensor()[:, :, 21:28]

                state = state - torch.linalg.solve(Jt, Rt)

            old_state = state

        test = torch.norm(state)
        test.backward()
