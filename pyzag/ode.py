"""Module for solving ordinary differential equations using numerical integration"""

import torch

from pyzag import nonlinear


class IntegrateODE(nonlinear.NonlinearRecursiveFunction):
    """Maps an ODE to a nonlinear function with some numerical integration scheme

    The input is a torch Module which defines the rate form of the ODE.  The forward function must
    return both the time rate of change of the state and the derivative of the rate of change with
    respect to the state as a function of time and the current state.

    The input and output for the underlying ODE must meet our global batch convention, where the first
    dimension of the input is used to vectorize time/step.

    Args:
        ode (torch.nn.Module): module defining the system of ODEs
    """

    def __init__(self, ode, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ode = ode


class BackwardEulerODE(IntegrateODE):
    """Applies a backward Euler time integration scheme to an ODE

    The input is a torch Module which defines the rate form of the ODE.  The forward function must
    return both the time rate of change of the state and the derivative of the rate of change with
    respect to the state as a function of time and the current state.

    The input and output for the underlying ODE must meet our global batch convention, where the first
    dimension of the input is used to vectorize time/step.

    Args:
        ode (torch.nn.Module): module defining the system of ODEs
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Lookback of 1!
        self.lookback = 1

    def forward(self, x, t):
        """Provide the blocked residual and Jacobian for the time integration scheme

        Args:
            x (torch.tensor):   (nchunk+self.n,...,nstate) tensor.  As always, the first batch dimension
                is used to vectorize over steps (for an ODE, time steps).  Then the operator
                must accept arbitrary batch dimensions.  The final dimension is the problem
                state size.
            t (torch.tensor):   (nchunk+self.n,...,) tensor of times as driving forces

        Returns:
            R (torch.tensor):   (nchunk,...,nstate ) tensor giving the nonlinear residual
            J (torch.tensor):   (self.n, nchunk,...,nstate,nstate) tensor giving the Jacobians

        """
        dt = torch.diff(t, dim=0)

        x_dot, J_dot = self.ode(t, x)

        R = x[1:] - x[:-1] - x_dot[1:] * dt
        J = torch.stack(
            [
                -torch.eye(x.shape[-1], dtype=x.dtype, device=x.device).expand_as(
                    J_dot[1:]
                ),
                torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
                - J_dot[1:] * dt.unsqueeze(-1),
            ]
        )

        return R, J


class ForwardEulerODE(IntegrateODE):
    """Applies a forward Euler time integration scheme to an ODE

    The input is a torch Module which defines the rate form of the ODE.  The forward function must
    return both the time rate of change of the state and the derivative of the rate of change with
    respect to the state as a function of time and the current state.

    The input and output for the underlying ODE must meet our global batch convention, where the first
    dimension of the input is used to vectorize time/step.

    Args:
        ode (torch.nn.Module): module defining the system of ODEs
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Lookback of 1!
        self.lookback = 1

    def forward(self, x, t):
        """Provide the blocked residual and Jacobian for the time integration scheme

        Args:
            x (torch.tensor):   (nchunk+self.n,...,nstate) tensor.  As always, the first batch dimension
                is used to vectorize over steps (for an ODE, time steps).  Then the operator
                must accept arbitrary batch dimensions.  The final dimension is the problem
                state size.
            t (torch.tensor):   (nchunk+self.n,...,) tensor of times as driving forces

        Returns:
            R (torch.tensor):   (nchunk,...,nstate ) tensor giving the nonlinear residual
            J (torch.tensor):   (self.n, nchunk,...,nstate,nstate) tensor giving the Jacobians

        """
        dt = torch.diff(t, dim=0)

        x_dot, J_dot = self.ode(t, x)

        R = x[1:] - x[:-1] - x_dot[:-1] * dt
        J = torch.stack(
            [
                -torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
                - J_dot[:-1] * dt.unsqueeze(-1),
                torch.eye(x.shape[-1], dtype=x.dtype, device=x.device).expand_as(
                    J_dot[:-1]
                ),
            ]
        )

        return R, J
