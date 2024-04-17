import torch

from pyzag import chunktime

# TODO: 1) separate out step selector to object 2) separate out initial guess selector to object,
# 3) linear solver to object, 4) nonlinear solver to class


class NonlinearRecursiveFunction(torch.nn.Module):
    """Basic structure of a nonlinear recursive function

    This class has two basic responsibilities, to define the residual and Jacobian of the function itself and second
    to define the lookback.

    The function and Jacobian are defined through `forward` so that this class is callable.

    The lookback is defined as a property

    The function here defines a time series through the recursive relation

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

    However, for convience we instead take these driving forces as python `*args`.  Each entry in `*args` must have a
    shape of :math:`(n_{block} + n_{lookback}, \\ldots)` and we leave it to the user to use each entry as they see fit.

    Put it another way, the only hard requirement for driving forces provided as `*args` is that the first dimension of the
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
    """Predict by providing the values from the previous steps"""

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


class StepGenerator:
    """Generate chunks of recursive steps to produce at once

    Args:
        block_size (int):   regular chunk size
        first_block_size (int): if > 0 then use a special first chunk size, after that use block_size
    """

    def __init__(self, block_size=1, first_block_size=0):
        self.block_size = block_size
        self.offset_step = first_block_size

    def __call__(self, n):
        self.steps = [1]
        if self.offset_step > 0:
            self.steps += [self.offset_step + 1]
        self.steps += list(range(self.steps[-1], n, self.block_size))[1:] + [n]

        self.i = 0

        return self

    def __iter__(self):
        return self

    def __next__(self):
        """Return two at a time"""
        self.i += 1
        if self.i < len(self.steps):
            return self.steps[self.i - 1], self.steps[self.i]
        raise StopIteration


class RecursiveNonlinearEquationSolver:
    """Generates a time series from a recursive nonlinear equation and (optionally) uses the adjoint method to provide derivatives

    The time series is generated in a batched manner, generating `block_size` steps at a time.

    Args:
        func (nonlinear.NonlinearRecursiveFunction):   defines the nonlinear system
        y0 (torch.tensor):  initial state values with shape (..., nstate)

    Keyword Args:
        step_generator (nonlinear.StepGenerator): iterator to generate the blocks to integrate at once, default has a block size of 1 and no special fist step
        predictor (nonlinear.Predictor): how to generate guesses for the nonlinear solve.  Default uses all zeros
        direct_solve_operator (chunktime.LUFactorization):  how to solve the batched, blocked system of equations.  Default is to use Thomas's method
    """

    def __init__(
        self,
        func,
        y0,
        step_generator=StepGenerator(1),
        predictor=ZeroPredictor(),
        direct_solve_operator=chunktime.BidiagonalThomasFactorization,
        nonlinear_solver=chunktime.ChunkNewtonRaphson(),
    ):
        # Store basic information
        self.func = func
        self.y0 = y0

        self.direct_solve_operator = direct_solve_operator
        self.step_generator = step_generator
        self.predictor = predictor
        self.nonlinear_solver = nonlinear_solver

        # For the moment we only accept lookback = 1
        if self.func.lookback != 1:
            raise ValueError(
                "The RecursiveNonlinearFunction has lookback = %i, but the current solver only handles lookback = 1!"
                % self.func.lookback
            )

    def solve(self, n, *args, cache_adjoint=False):
        """Solve the recursive equations for n steps

        Args:
            n (int):    number of recursive time steps to solve, step 1 is y0
            *args:      driving forces to pass to the model

        Keyword Args:
            cache_adjoint (bool): if true store results for adjoint pass
        """
        # Make sure our shapes are okay
        self._check_shapes(n, args)

        # Setup results and store y0
        result = torch.empty(
            n, *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device
        )
        result[0] = self.y0

        # Actually solve
        for k1, k2 in self.step_generator(n):
            result[k1:k2] = self.block_update(
                result[k1 - self.func.lookback : k1],
                self.predictor.predict(result, k1, k2 - k1),
                [arg[k1 - self.func.lookback : k2] for arg in args],
            )

        # Cache result and driving forces if needed for adjoint pass
        if cache_adjoint:
            self.forces = [arg.flip(0) for arg in args]
            self.result = result.flip(0)

        return result

    def block_update(self, prev_solution, solution, forces):
        """Actually update the recursive system

        Args:
            prev_solution (tensor): previous lookback steps of solution
            solution (tensor): guess at nchunk steps of solution
            forces (list of tensors): driving forces for next chunk plus lookback, to be passed as *args
        """

        def RJ(y):
            # Batch update the rate and jacobian
            R, J = self.func(
                torch.cat([prev_solution, y]),
                *forces,
            )
            return R, chunktime.BidiagonalForwardOperator(
                J[1], J[0], inverse_operator=self.direct_solve_operator
            )

        return self.nonlinear_solver.solve(RJ, solution)

    def _check_shapes(self, n, forces):
        """Check the shapes of everything before starting the calculation

        Args:
            n (int):        number of recursive time steps
            forces (list):  list of driving forces
        """
        correct_force_batch_shape = (n,) + self.y0.shape[:-1]
        for f in forces:
            if f.shape[:-1] != correct_force_batch_shape:
                raise ValueError(
                    "One of the provided driving forces does not have the correct shape.  The batch shape should be "
                    + str(correct_force_batch_shape)
                    + " but is instead "
                    + str(f.shape[:-1])
                )
