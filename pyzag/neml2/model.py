from pyzag import nonlinear

import torch

import neml2
from neml2.tensors import LabeledAxisAccessor as AA
from neml2.tensors import BatchTensor, LabeledVector


def assemble_tensor(order, items):
    """Assemble individual items into a tensor

    Args:
        order (list of str): the order in which to stack the tensor
        items (dict): tensors to assemble
    """
    return torch.cat([items[expand_name(name)] for name in order], dim=-1)


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
        exclude_parameters (list of str): exclude parameters in the list from becoming torch parameters
        neml2_sep_char (str): parameter seprator character used in NEML2
        our_set_char (str): how to convert that seperator into something python can use
    """

    def __init__(
        self,
        model,
        state_axis="state",
        forces_axis="forces",
        residual_axis="residual",
        old_prefix="old_",
        exclude_parameters=[],
        neml2_sep_char=".",
        our_sep_char="_",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.model = model
        self.lookback = 1  # Really there isn't any other option in NEML2 right now

        # Setup parameters
        self.neml2_sep_char = neml2_sep_char
        self.our_sep_char = our_sep_char
        self._setup_parameters(exclude_parameters)

        # Store basic information about the model
        self.state_axis = state_axis
        self.forces_axis = forces_axis
        self.residual_axis = residual_axis
        self.old_prefix = old_prefix

        # Identify the names of the state variables and establish a fixed order
        self.state_names = self._gather_vars(self.state_axis)

        # Establish the names of the forces
        self.force_names = self._gather_vars(self.forces_axis)

        # Figure out the order of the state, old_state, forces, and old_forces
        self.input_order = self.model.input_axis().subaxis_names()

        # Though the cost isn't high, figure out the indices needed to extract the state and old_state from the input axis
        self.output_slices = self._make_output_slices()

        # Do some basic consistency checking
        self._check_model()

    def _setup_parameters(self, exclude_parameters):
        """Initialize the torch parameters for this object

        Args:
            exclude_parameters (list of str): parameters to not include
        """
        self.parameter_name_map = {}
        for k in self.model.named_parameters().keys():
            if k in exclude_parameters:
                continue
            val = self.model.named_parameters()[k]
            val.requires_grad_(True)
            rename = k.replace(self.neml2_sep_char, self.our_sep_char)
            self.parameter_name_map[rename] = k
            self.register_parameter(rename, torch.nn.Parameter(val.tensor().tensor()))

    def _make_output_slices(self):
        """Just figure out where the "state" and "old_state" blocks live"""
        sz = [
            self.model.input_axis().subaxis(n).storage_size() for n in self.input_order
        ]
        # Ugh or pull in numpy, or make a tensor just for cumsum
        offsets = [0]
        for i in sz:
            offsets.append(offsets[-1] + i)

        state_index = self.input_order.index(self.state_axis)
        old_state_index = self.input_order.index(self.old_prefix + self.state_axis)

        return [
            [offsets[state_index], offsets[state_index + 1]],
            [offsets[old_state_index], offsets[old_state_index + 1]],
        ]

    def _check_model(self):
        """Check that the NEML2 model is properly setup

        These could all be debug only checks, but really they are only called once so may not be worth it
        """
        # Input should have forces, old_forces, state, and old_state
        assert self.model.input_axis().nsubaxis == 4
        assert self.model.input_axis().has_subaxis(AA(self.state_axis))
        assert self.model.input_axis().has_subaxis(AA(self.forces_axis))
        assert self.model.input_axis().has_subaxis(
            AA(self.old_prefix + self.state_axis)
        )
        assert self.model.input_axis().has_subaxis(
            AA(self.old_prefix + self.forces_axis)
        )

        # There should be a common set of state variables and a common set of forces variables
        for var in self.state_names:
            for prefix in ["", self.old_prefix]:
                assert (
                    self.model.input_axis()
                    .subaxis(prefix + self.state_axis)
                    .has_variable(AA(*var))
                )
        for var in self.force_names:
            for prefix in ["", self.old_prefix]:
                assert (
                    self.model.input_axis()
                    .subaxis(prefix + self.forces_axis)
                    .has_variable(AA(*var))
                )

        # Output axis should just have residual
        assert self.model.output_axis().nsubaxis == 1

        # And the variables should be the same as state
        for var in self.state_names:
            assert (
                self.model.output_axis()
                .subaxis(self.residual_axis)
                .has_variable(AA(*var))
            )

    def collect_state(self, state_dict):
        """Assemble a dict of state tensors into one tensor

        Args:
            state_dict: a dictionary  mapping names to tensors
        """
        return assemble_tensor(self.state_names, state_dict)

    def collect_forces(self, forces_dict):
        """Assemble a dict of force tensors into one tensor

        Args:
            forces_dict: a dictionary mapping names to tensors
        """
        return assemble_tensor(self.force_names, forces_dict)

    def _gather_vars(self, axis):
        """Gather variable names from an axis

        Args:
            axis (str): axis name
        """
        assert self.model.input_axis().has_subaxis(AA(axis))
        return self.model.input_axis().subaxis(axis).variable_names()

    def forward(self, state, forces):
        """Actually call the NEML model and return the residual and Jacobian

        Args:
            state (torch.tensor): tensor of state
            forces (torch.tensor): tensor of forces
        """
        # Need to assemble into one big tensor
        x = self._assemble_input(state, forces)

        # Update the "true" model parameters
        self._update_parameter_values()

        # Call the model to get the results and derivatives
        y, J = self.model.value_and_dvalue(x)

        # Gary promises the output will be in the same order as the input, so only thing to do is stack the derivatives
        J = self._extract_jacobian(J)

        return y.tensor().tensor(), J

    def _update_parameter_values(self):
        """Copy over the torch parameters to NEML2 prior to run"""
        for n, p in self.named_parameters():
            nparam = self.model.named_parameters()[self.parameter_name_map[n]]
            nparam.set(BatchTensor(p, nparam.batch_dim()))

    def _assemble_input(self, state, forces):
        """Assemble the big BatchTensor of model input.

        This has to account for the lookback and then map the entries into the right order

        It also updates the model batch shape

        Args:
            state (torch.tensor): assembled state tensor
            forces (torch.tensor): assembled forces tensor
        """
        batch_dim = state.dim() - 1  # Should always be 2
        assert batch_dim == 2

        data = {
            "state": state[self.lookback :],
            "old_state": state[: -self.lookback],
            "forces": forces[self.lookback :],
            "old_forces": forces[: -self.lookback],
        }

        x = BatchTensor(
            torch.cat([data[entry] for entry in self.input_order], dim=-1), batch_dim
        )
        self.model.reinit(x.batch.shape, 1)
        return LabeledVector(x, [self.model.input_axis()])

    def _extract_jacobian(self, J):
        """Extract the "state" and "old_state" parts of the Jacobian

        Args:
            J: output from NEML2 model
        """
        Jt = J.tensor().tensor()
        return torch.stack(
            [
                Jt[..., self.output_slices[1][0] : self.output_slices[1][1]],
                Jt[..., self.output_slices[0][0] : self.output_slices[0][1]],
            ]
        )