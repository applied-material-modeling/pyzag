from pyzag import nonlinear

import torch

import neml2
from neml2.tensors import LabeledAxisAccessor as AA


def assemble_tensor(order, items):
    """Assemble individual items into a tensor

    Args:
        order (list of str): the order in which to stack the tensor
        items (dict): tensors to assemble
    """
    return torch.cat([items[name] for name in order], dim=-1)


def expand_name(name, sep="/"):
    """Expand a list of name parts into a full NEML2 name

    Args:
        name (list of str): name as string

    Keyword Args:
        sep (str): separator character
    """
    return sep.join(name)


class NEML2Model(nonlinear.NonlinearRecursiveFunction):
    """Wraps a NEML2 model into a `nonlinear.NonlinearRecursiveFunction`

    Args:
        model (NEML2 model):    the model to wrap

    Keyword Args:
        state_axis (str): name of the state axis
        forces_axis (str): name of the forces axis
        residual_axis (str): name of the residual axis
        old_prefix (str): prefix on the axis name to get the old values
    """

    def __init__(
        self,
        model,
        state_axis="state",
        forces_axis="forces",
        residual_axis="residual",
        old_prefix="old_",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.model = model
        self.lookback = 1  # Really there isn't any other option in NEML2 right now

        self.state_axis = state_axis
        self.forces_axis = forces_axis
        self.residual_axis = residual_axis
        self.old_prefix = old_prefix

        # Identify the names of the state variables and establish a fixed order
        self.state_names = self._gather_vars(self.state_axis)

        # Establish the names of the forces
        self.force_names = self._gather_vars(self.forces_axis)

    def collect_state(self, state_dict):
        """Assemble a dict of state tensors into one tensor"""
        return assemble_tensor(self.state_names, state_dict)

    def collect_forces(self, forces_dict):
        """Assemble a dict of force tensors into one tensor"""
        return assemble_tensor(self.force_names, forces_dict)

    def _gather_vars(self, axis):
        """Gather variable names from an axis

        Args:
            axis (str): axis name
        """
        assert self.model.input_axis().has_subaxis(AA(axis))
        return [
            expand_name(n)
            for n in self.model.input_axis().subaxis(axis).all_variable_names()
        ]

    def forward(self, state, forces):
        """Actually call the NEML model and return the residual and Jacobian

        Args:
            state (torch.tensor): tensor of state
            forces (torch.tensor): tensor of forces
        """
        batch_dim = state.dim() - 1

        # Need to assemble into one big tensor

        # Call the model to get the results and derivatives

        # Rearrange output
