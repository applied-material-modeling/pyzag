class RecursiveNonlinearEquationSolver:
    """Generates a time series from a recursive nonlinear equation and (optionally) uses the adjoint method to provide derivatives

    The time series is generated in a batched manner, generating `block_size` steps at a time.

    The key input to this class is a pytorch callable defining the recursive system of nonlinear equations
    defining a time series through the relation:

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


    Args:
        func (callable):    callable returning the nonlinear residual and Jacobian.
            The signature of this function is R, J = f(X, *args) where the shape of the state X and the
            driving forces provided by args must meet the definition described above.  Similarly, the output
            shapes of R and J relate to the input shapes of X and *args as described above.
        y0 (torch.tensor):  initial state values with shape (..., nstate)

    Keyword Args:
        block_size (int):   maximum block size for time vectorization.  This might be decreased at the end of the time series to
            hit the number of time steps requested by the `solve` method.
    """

    def __init__(self, func, y0, block_size=1):
        # Store basic information
        self.func = func
        self.y0 = y0
        self.block_size = block_size

    def solve(self, n, *args):
        """Solve the recursive equations for n steps

        Args:

        """
        pass
