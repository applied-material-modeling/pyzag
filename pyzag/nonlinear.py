# Copyright 2024, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: pyzag
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Basic functionality for solving recursive nonlinear equations and calculating senstivities using the adjoint method"""

import torch

from pyzag import chunktime


class NonlinearRecursiveFunction(torch.nn.Module):
    # pylint: disable=W0223,W0246
    """Basic structure of a nonlinear recursive function

    This class has two basic responsibilities, to define the residual and Jacobian of the function itself and second
    to define the lookback.

    The function and Jacobian are defined through `forward` so that this class is callable.

    The lookback is defined as a property

    The function here defines a series through the recursive relation

    .. math::
        f(x_{n+1}, x_{n}, x_{n-1}, \\ldots) = 0

    We denote the number of past entries in the time series used in defining this residual function as
    the *lookback*.  A function with a lookback of 1 has the definition

    .. math::
        f(x_{n+1}, x_{n}) = 0

    A lookback of 2 is

    .. math::
        f(x_{n+1}, x_{n}, x_{n-1}) = 0

    etc.

    *For the moment this class only supports functions with a lookback of 1!*.

    The function also must provide the Jacobian of the residual with respect to each input.  So a function
    with lookback 1 must provide both

    .. math::
        J_{n+1} = \\frac{\\partial f}{\\partial x_{n+1}}

    and

    .. math::
        J_{n} = \\frac{\\partial f}{\\partial x_n}

    The input and output shapes of the function as dictated by the lookback and the desired amount of time-vectorization.
    Let's call the number of time steps to be solved for at once :math:`n_{block}` and the lookback as
    :math:`n_{lookback}`.  Let's say our function has a state size :math:`\\left| x \\right| = n_{state}`.
    Our convention is the first dimension of input and output is the batch time axis and the last
    dimension is the true state size.  The input shape of the state :math:`x` must be
    :math:`(n_{block} + n_{lookback}, \\ldots, n_{state})` where :math:`\\ldots` indicates some arbitrary
    number of batch dimensions.  The output shape of the nonlinear residual will be
    :math:`(n_{block}, \\ldots, n_{state})`.  The output shape of the Jacobian will be
    :math:`(n_{lookback} + 1, n_{block}, \\ldots, n_{state}, n_{state})`.

    Additionally, we allow the function to take some driving forces as input.  Mathematically we could
    envision these as an additional input tensor :math:`u` with shape :math:`(n_{block} + n_{lookback}, \\ldots, n_{force})` and
    we expand the residual function definition to be

    .. math::
        f(x_{n+1}, x_{n}, x_{n-1}, \\ldots, u_{n+1}, u_{n}, u_{n-1}, \\ldots) = 0

    However, for convience we instead take these driving forces as python `*args` and `**kwargs`.  Each entry in `*args` and `**kwargs` must have a
    shape of :math:`(n_{block} + n_{lookback}, \\ldots)` and we leave it to the user to use each entry as they see fit.

    To put it another way, the only hard requirement for driving forces is that the first dimension of the
    tensor must be slicable in the same way as the state :math:`x`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FullTrajectoryPredictor:
    """Predict steps using a complete user-provided trajectory

    This is often used during optimization runs, where the provided trajectory could be the results from the previous step in the
    optimization routine

    Args:
        history (torch.tensor):     tensor of shape :math:`(n_{time},...,n_{state})` giving a complete previous trajectory
    """

    def __init__(self, history):
        self.history = history

    def predict(self, results, k, kinc):
        # pylint: disable=W0613
        """Predict the next steps

        Args:
            results (torch.tensor): current results tensor, filled up to step k.
            k (int): start of current chunk
            kinc (int): next number of steps to predict
        """
        return self.history[k : k + kinc]


class ZeroPredictor:
    """Predict steps just using zeros"""

    def predict(self, results, k, kinc):
        """Predict the next steps

        Args:
            results (torch.tensor): current results tensor, filled up to step k.
            k (int): start of current chunk
            kinc (int): next number of steps to predict
        """
        return torch.zeros_like(results[k : k + kinc])


class PreviousStepsPredictor:
    """Predict by providing the values from the previous chunk of steps steps"""

    def predict(self, results, k, kinc):
        """Predict the next steps

        Args:
            results (torch.tensor): current results tensor, filled up to step k.
            k (int): start of current chunk
            kinc (int): next number of steps to predict
        """
        if k - kinc - 1 < 0:
            return torch.zeros_like(results[k : k + kinc])
        return results[(k - kinc) : k]


class LastStepPredictor:
    """Predict by providing the values from the previous single step"""

    def predict(self, results, k, kinc):
        """Predict the next steps

        Args:
            results (torch.tensor): current results tensor, filled up to step k.
            k (int): start of current chunk
            kinc (int): next number of steps to predict
        """
        if k < 1:
            return torch.zeros_like(results[k : k + kinc])

        return results[k - 1].unsqueeze(0).expand((kinc,) + results.shape[1:])


class StepExtrapolatingPredictor:
    """Predict by extrapolating using the previous *chunks* of steps"""

    def predict(self, results, k, kinc):
        """Predict the next steps

        Args:
            results (torch.tensor): current results tensor, filled up to step k.
            k (int): start of current chunk
            kinc (int): next number of steps to predict
        """
        if k < 1:
            return torch.zeros_like(results[k : k + kinc])

        results[k - 1].unsqueeze(0).expand((kinc,) + results.shape[1:])

        dinc = (results[k - 1] - results[k - 2]) + results[k - 1]

        return dinc.unsqueeze(0).expand((kinc,) + results.shape[1:])


class ExtrapolatingPredictor:
    """Predict by extrapolating the values from the previous *single* steps"""

    def predict(self, results, k, kinc):
        """Predict the next steps

        Args:
            results (torch.tensor): current results tensor, filled up to step k.
            k (int): start of current chunk
            kinc (int): next number of steps to predict
        """
        if k - kinc - 1 < 0:
            return torch.zeros_like(results[k : k + kinc])
        if k - 2 * kinc - 1 < 0:
            return results[(k - kinc) : k]

        inc = results[(k - kinc) : k] - results[(k - 2 * kinc) : k - kinc]

        return results[(k - kinc) : k] + inc


class StepGenerator:
    """Generate chunks of recursive steps to produce at once

    Args:
        block_size (int):   regular chunk size
        first_block_size (int): if > 0 then use a special first chunk size, after that use block_size
    """

    def __init__(self, block_size=1, first_block_size=0):
        self.block_size = block_size
        self.offset_step = first_block_size
        self.back = False

        self.n = 0
        self.steps = []
        self.pairs = []
        self.i = 0

    def __call__(self, n):
        """Generate n steps of chunks

        Args:
            n (int): number of steps
        """
        self.back = False
        self.n = n
        self.steps = [1]
        if self.offset_step > 0:
            self.steps += [self.offset_step + 1]
        self.steps += list(range(self.steps[-1], n, self.block_size))[1:] + [n]

        self.pairs = list(zip(self.steps[:-1], self.steps[1:]))

        self.i = 0

        return self

    def __iter__(self):
        return self

    def reverse(self):
        """Reverse the iterator to yield chunks starting from the end"""
        self.back = True
        rev = [
            (self.n - k2, self.n - k1)
            for k1, k2 in zip(self.steps[:-1], self.steps[1:])
        ][:-1]
        if rev[-1][0] != 1:
            rev += [(1, rev[-1][0])]
        self.pairs = rev

        self.i = 0

        return self

    def __next__(self):
        """Iterate forward through the steps"""
        self.i += 1
        if self.i <= len(self.pairs):
            return self.pairs[self.i - 1]
        raise StopIteration


class InitialOffsetStepGenerator(StepGenerator):
    """Generate chunks of recursive steps to produce at once

    The user can provide a list of initial chunks to use, after which the object
    goes back to chunks of block_size

    Args:
        block_size (int):   regular chunk size
        initial_steps (list of int): initial steps
    """

    def __init__(self, *args, initial_steps=None, **kwargs):
        super().__init__(*args, **kwargs)

        if initial_steps is None:
            self.initial_steps = []
        else:
            self.initial_steps = initial_steps

    def __call__(self, n):
        """Generate n steps of chunks

        Args:
            n (int): number of steps
        """
        self.back = False
        self.n = n
        self.steps = [1]
        if len(self.initial_steps) > 0:
            self.steps += [i + 1 for i in self.initial_steps]
        self.steps += list(range(self.steps[-1], n, self.block_size))[1:] + [n]

        self.pairs = list(zip(self.steps[:-1], self.steps[1:]))

        self.i = 0

        return self


class RecursiveNonlinearEquationSolver(torch.nn.Module):
    """Generates a time series from a recursive nonlinear equation and (optionally) uses the adjoint method to provide derivatives

    The series is generated in a batched manner, generating ``block_size`` steps at a time.

    Args:
        func (:py:class:`pyzag.nonlinear.NonlinearRecursiveFunction`):   defines the nonlinear system

    Keyword Args:
        step_generator (:py:class:`pyzag.nonlinear.StepGenerator`): iterator to generate the blocks to integrate at once, default has a block size of 1 and no special fist step
        predictor (:py:class:`pyzag.nonlinear.Predictor`): how to generate guesses for the nonlinear solve.  Default uses all zeros
        direct_solve_operator (:py:class:`pyzag.chunktime.LUFactorization`):  how to solve the batched, blocked system of equations.  Default is to use Thomas's method
    """

    def __init__(
        self,
        func,
        step_generator=StepGenerator(1),
        predictor=ZeroPredictor(),
        direct_solve_operator=chunktime.BidiagonalThomasFactorization,
        nonlinear_solver=chunktime.ChunkNewtonRaphson(),
    ):
        super().__init__()
        # Store basic information
        self.func = func

        self.direct_solve_operator = direct_solve_operator
        self.step_generator = step_generator
        self.predictor = predictor
        self.nonlinear_solver = nonlinear_solver

        # Backward cache
        self.n = 0
        self.forces = []
        self.result = None
        self.adjoint_params = []

        # For the moment we only accept lookback = 1
        if self.func.lookback != 1:
            raise ValueError(
                f"The RecursiveNonlinearFunction has lookback = {self.func.lookback}, but the current solver only handles lookback = 1!"
            )

    def forward(self, *args, **kwargs):
        """Alias for solve

        Args:
            y0 (torch.tensor):  initial state values with shape ``(..., nstate)``
            n (int):    number of recursive time steps to solve, step 1 is y0
            ``*args``:      driving forces to pass to the model

        Keyword Args:
            adjoint_params (None or list of parameters): if provided, cache the information needed to run an adjoint pass over the parameters in the list
        """
        return self.solve(*args, **kwargs)

    def solve(self, y0, n, *args, adjoint_params=None):
        """Solve the recursive equations for n steps

        Args:
            y0 (torch.tensor):  initial state values with shape ``(..., nstate)``
            n (int):    number of recursive time steps to solve, step 1 is y0
            ``*args``:      driving forces to pass to the model

        Keyword Args:
            adjoint_params (None or list of parameters): if provided, cache the information needed to run an adjoint pass over the parameters in the list
        """
        # Make sure our shapes are okay
        self._check_shapes(y0, n, args)

        # Setup results and store y0
        result = torch.empty(n, *y0.shape, dtype=y0.dtype, device=y0.device)
        result[0] = y0

        # Actually solve
        for k1, k2 in self.step_generator(n):
            result[k1:k2] = self.block_update(
                result[k1 - self.func.lookback : k1].clone(),
                self.predictor.predict(result, k1, k2 - k1).clone(),
                [arg[k1 - self.func.lookback : k2].clone() for arg in args],
            )

        # Cache result and driving forces if needed for adjoint pass
        if adjoint_params:
            self.n = n
            self.forces = [arg.clone() for arg in args]
            self.result = result.clone()
            self.adjoint_params = adjoint_params

        return result

    def block_update(self, prev_solution, solution, forces):
        """Actually update the recursive system

        Args:
            prev_solution (tensor): previous lookback steps of solution
            solution (tensor): guess at nchunk steps of solution
            forces (list of tensors): driving forces for next chunk plus lookback, to be passed as ``*args``
        """

        def RJ(y):
            # Batch update the rate and jacobian
            R, J = self.func(
                torch.cat([prev_solution, y]),
                *forces,
            )
            return R, chunktime.BidiagonalForwardOperator(
                J[1], J[0, 1:], inverse_operator=self.direct_solve_operator
            )

        return self.nonlinear_solver.solve(RJ, solution)

    def rewind(self, output_grad):
        """Rewind through an adjoint pass to provide the dot product for each quantity in output_grad

        Args:
            output_grad (torch.tensor): thing to dot product with
        """
        # Setup storage for result
        grad_result = tuple(
            torch.zeros(p.shape, device=output_grad.device) for p in self.adjoint_params
        )

        # Loop backwards through time
        for k1, k2 in self.step_generator(len(self.result)).reverse():
            # Get our block of the results
            with torch.enable_grad():
                R, J = self.func(
                    self.result[k1 - 1 : k2 + 1],
                    *[f[k1 - 1 : k2 + 1] for f in self.forces],
                )
                # We want these in reverse order for the chunked update
                R = R.flip(0)
                J = J.flip(1)

            # Setup initial condition if this is our first time through
            if k2 + 1 == len(self.result):
                adjoint = -torch.linalg.solve(
                    J[1, 0].transpose(-1, -2), output_grad[-1]
                ).unsqueeze(0)
                # And count the accumulation
                with torch.enable_grad():
                    grad_result = self.accumulate(
                        grad_result, adjoint, R[:1], retain=True
                    )

            # Do the block adjoint update
            adjoint = self.block_update_adjoint(
                J, output_grad[k1:k2].flip(0), adjoint[-1]
            )

            # And accumulate
            with torch.enable_grad():
                grad_result = self.accumulate(grad_result, adjoint, R[1:])

        return grad_result, adjoint[-1]

    def accumulate(self, grad_result, full_adjoint, R, retain=False):
        """Accumulate the updated gradient values

        Args:
            grad_result (tuple of tensor): current gradient results
            full_adjoint (torch.tensor): adjoint values
            R (torch.tensor): function values, for AD

        Keyword Args:
            retain (bool): if True, retain the AD graph for a second pass
        """
        # This was a design choice.  Right now we don't know if parameters affect the initial conditions *only*
        # or if they come into the recursive function somehow.  If they only affect the recursive function then
        # grad will raise an error if you don't set allowed_unused.  If you do set it then you need to
        # check for Nones in the output.
        g = torch.autograd.grad(
            R, self.adjoint_params, full_adjoint, retain_graph=retain, allow_unused=True
        )
        return tuple(
            pi + gi if gi is not None else pi for pi, gi in zip(grad_result, g)
        )

    def block_update_adjoint(self, J, grads, a_prev):
        """Do the blocked adjoint solve

        Args:
            J (torch.tensor):   block of jacobians
            grads (torch.tensor): block of gradient values
            a_prev (torch.tensor): previous adjoint value

        Returns:
            adjoint_block (torch.tensor): next block of updated adjoint values
        """
        # Remember to transpose
        operator = self.direct_solve_operator(
            J[1, 1:].transpose(-1, -2), J[0, 1:-1].transpose(-1, -2)
        )
        rhs = -grads
        rhs[0] -= torch.matmul(J[0, 0].transpose(-1, -2), a_prev.unsqueeze(-1)).squeeze(
            -1
        )

        return operator.matvec(rhs)

    def _check_shapes(self, y0, n, forces):
        """Check the shapes of everything before starting the calculation

        Args:
            y0 (torch.tensor):  initial state values with shape ``(..., nstate)``
            n (int):        number of recursive time steps
            forces (list):  list of driving forces
        """
        correct_force_batch_shape = (n,) + y0.shape[:-1]
        for f in forces:
            if f.shape[:-1] != correct_force_batch_shape:
                raise ValueError(
                    "One of the provided driving forces does not have the correct shape.  The batch shape should be "
                    + str(correct_force_batch_shape)
                    + " but is instead "
                    + str(f.shape[:-1])
                )


class AdjointWrapper(torch.autograd.Function):
    # pylint: disable=all
    """Defines the backward pass for pytorch, allowing us to mix the adjoint calculation with AD"""

    @staticmethod
    def forward(ctx, solver, y0, n, forces, *params):
        with torch.no_grad():
            y = solver.solve(y0, n, *forces, adjoint_params=params)
            ctx.solver = solver
            return y

    @staticmethod
    def backward(ctx, output_grad):
        with torch.no_grad():
            grad_res, adj_last = ctx.solver.rewind(output_grad)
            if ctx.needs_input_grad[1]:
                return (None, -adj_last, None, None, *grad_res)
            return (None, None, None, None, *grad_res)


def solve(solver, y0, n, *forces):
    """Solve a :py:class:`pyzag.nonlinear.RecursiveNonlinearEquationSolver` for a time history without the adjoint method

    Args:
        solver (py:class:`pyzag.nonlinear.RecursiveNonlinearEquationSolver`): solve to apply
        n (int): number of recursive steps
        ``*forces`` (``*args`` of tensors): driving forces
    """
    return solver.solve(y0, n, *forces)


def solve_adjoint(solver, y0, n, *forces):
    """Apply a :py:class:`pyzag.nonlinear.RecursiveNonlinearEquationSolver` to solve for a time history in an adjoint differentiable way

    Args:
        solver (:py:class:`pyzag.nonlinear.RecursiveNonlinearEquationSolver`): solve to apply
        n (int): number of recursive steps
        ``*forces`` (``*args`` of tensors): driving forces
    """
    # This is very fragile code to accomodate pyro
    # 1) We no longer can use the list of torch parameters b/c we converted them to pyro distributions
    # 2) We can't rely on a list of values (i.e. tensors) stored in converted_params because we need to
    # .  grab these *after* pyro samples them
    # 3) We can't rely on this object having a converted_params method because it might be a basic "unconverted"
    #    pytorch module.
    additional_params = []
    for m in solver.modules():
        if hasattr(m, "converted_params"):
            additional_params.extend(getattr(m, p) for p in m.converted_params)

    all_params = list(solver.parameters()) + additional_params

    return AdjointWrapper.apply(solver, y0, n, forces, *all_params)
