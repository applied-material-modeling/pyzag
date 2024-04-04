"""Random helper functions"""

import torch
import numpy as np
import numpy.random as ra


class ArbitraryBatchTimeSeriesInterpolator(torch.nn.Module):
    """
    Interpolate :code:`data` located at discrete :code:`times`
    linearly to point :code:`t`.

    This version handles batched of arbitrary size -- only the rightmost
    batch dimension must agree with the input data.  All other dimensions are
    broadcast.

    Args:
      times (torch.tensor):     input time series as a code:`(ntime,nbatch)`
                                array
      values (torch.tensor):    input values series as a code:`(ntime,nbatch)`
                                array
    """

    def __init__(self, times, data):
        super().__init__()
        # Transpose to put the shared dimension first
        self.times = times.t()
        self.values = data.t()

        self.slopes = torch.diff(self.values, dim=-1) / torch.diff(self.times, dim=-1)

    def forward(self, t):
        """
        Calculate the linearly-interpolated current values

        Args:
          t (torch.tensor):   batched times as :code:`(...,nbatch,)` array

        Returns:
          torch.tensor:       batched values at :code:`t`
        """
        squeeze = t.dim() == 1

        t = t.t()  # Transpose so the common dimension is first
        if squeeze:
            t = t.unsqueeze(-1)

        # Which dimensions to offset -- likely there is some efficiency
        # to be gained by flipping these, but probably depends on the
        # input shape
        d1 = -1
        d2 = -2

        offsets = t.unsqueeze(d1) - self.times[..., :-1].unsqueeze(d2)

        poss = self.slopes.unsqueeze(d2) * offsets + self.values[..., :-1].unsqueeze(d2)

        locs = torch.logical_and(
            t.unsqueeze(d1) <= self.times[..., 1:].unsqueeze(d2),
            t.unsqueeze(d1) > self.times[..., :-1].unsqueeze(d2),
        )

        # Now we need to fix up the stupid left side
        locs[..., 0] = torch.logical_or(
            locs[..., 0], t == self.times[..., 0].unsqueeze(-1)
        )

        if squeeze:
            return poss[locs].squeeze(-1).t()

        return poss[locs].reshape(t.shape).t()


def generate_random_cycle(
    max_strain=[0, 0.02],
    R=[-1, 1],
    strain_rate=[1.0e-3, 1.0e-5],
    tension_hold=[0, 1 * 3600.0],
    compression_hold=[0, 600],
):
    """
    Generate a random cycle in the provided ranges

    Keyword Args:
      max_strain (list):        range of the maximum strains
      R (list):                 range of R ratios
      strain_rate (list):       range of loading strain rates
      tension_hold (list):      range of tension hold times
      compression_hold (list):  range of compression hold times

    Returns:
      dict:                     dictionary describing cycle, described below

    * :code:`"max_strain"` -- maximum strain value
    * :code:`"R"` -- R ratio :math:`\\frac{max}{min}`
    * :code:`"strain_rate"` -- strain rate during load/unload
    * :code:`"tension_hold"` -- hold on the tension end of the cycle
    * :code:`"compression_hold"` -- hold on the compressive end of the cycle
    """
    return {
        "max_strain": ra.uniform(*max_strain),
        "R": ra.uniform(*R),
        "strain_rate": 10.0
        ** ra.uniform(np.log10(strain_rate[0]), np.log10(strain_rate[1])),
        "tension_hold": ra.uniform(*tension_hold),
        "compression_hold": ra.uniform(*compression_hold),
    }


def sample_cycle_normalized_times(cycle, N, nload=10, nhold=10):
    # pylint: disable=too-many-locals
    """
    Sample a cyclic test at a normalized series of times

    Take a random cycle dictionary and expand into discrete
    times, strains samples where times are the actual, physical
    times, given over the fixed phases

      * :math:`0 \\rightarrow  t_{phase}` -- tension load
      * :math:`t_{phase} \\rightarrow 2 t_{phase}` -- tension hold
      * :math:`2 t_{phase} \\rightarrow 3 t_{phase}` --   unload
      * :math:`3 t_{phase} \\rightarrow 4 t_{phase}` -- compression load
      * :math:`4 t_{phase} \\rightarrow 5 t_{phase}` -- compression hold
      * :math:`5 t_{phase} \\rightarrow 6 t_{phase}` -- unload

    This pattern repeats for N cycles

    Args:
      cycle (dict): dictionary defining the load cycle
      N (int):      number of repeats to include in the history

    Keyword Args:
      nload (int):  number of steps to use for the load time
      nhold (int):  number of steps to use for the hold time
    """
    emax = cycle["max_strain"]
    emin = cycle["R"] * cycle["max_strain"]
    erate = cycle["strain_rate"]

    # Segments:
    t1 = np.abs(emax) / erate
    t2 = cycle["tension_hold"]
    t3 = np.abs(emax - emin) / erate
    t4 = cycle["compression_hold"]
    t5 = np.abs(emin) / erate
    divisions = [t1, t2, t3, t4, t5]
    timesteps = [nload, nhold, 2 * nload, nhold, nload]
    cdivisions = np.cumsum(divisions)
    period = cdivisions[-1]

    Ntotal = nload * 4 + nhold * 2

    times = np.zeros((1 + Ntotal * N,))
    cycles = np.zeros(times.shape, dtype=int)

    n = 1
    tc = 0
    for k in range(N):
        for ti, ni in zip(divisions, timesteps):
            times[n : n + ni] = np.linspace(tc, tc + ti, ni + 1)[1:]
            cycles[n : n + ni] = k
            n += ni
            tc += ti

    tp = times % period
    strains = np.piecewise(
        tp,
        [
            np.logical_and(tp >= 0, tp < cdivisions[0]),
            np.logical_and(tp >= cdivisions[0], tp < cdivisions[1]),
            np.logical_and(tp >= cdivisions[1], tp < cdivisions[2]),
            np.logical_and(tp >= cdivisions[2], tp < cdivisions[3]),
            np.logical_and(tp >= cdivisions[3], tp < cdivisions[4]),
        ],
        [
            lambda tt: tt / t1 * emax,
            lambda tt: tt * 0 + emax,
            lambda tt: emax - (tt - cdivisions[1]) / t3 * (emax - emin),
            lambda tt: tt * 0 + emin,
            lambda tt: emin - (tt - cdivisions[3]) / t5 * emin,
        ],
    )

    return times, strains, cycles
