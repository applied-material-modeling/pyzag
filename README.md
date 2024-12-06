# pyzag


[![Documentation](https://github.com/applied-material-modeling/pyzag/actions/workflows/build_docs.yml/badge.svg?branch=main)](https://applied-material-modeling.github.io/pyzag/) [![Testing](https://github.com/applied-material-modeling/pyzag/actions/workflows/run_tests.yml/badge.svg?branch=main)](https://github.com/applied-material-modeling/pyzag/actions/workflows/run_tests.yml) [![Code formatting](https://github.com/applied-material-modeling/pyzag/actions/workflows/formatting_check.yml/badge.svg?branch=main)](https://github.com/applied-material-modeling/pyzag/actions/workflows/formatting_check.yml)

pyzag is a library for efficiently training generic models defined with a recursive nonlinear function.  Full documentation is available [here](https://applied-material-modeling.github.io/pyzag/).

The library is available as open source code with an [MIT license](https://raw.githubusercontent.com/applied-material-modeling/pyzag/main/LICENSE).

## Nonlinear recursive functions

A nonlinear recursive function has the form

$$f\left(x_{i-1}, x_i; p\right) =0 \, \forall i \in \left(1,2,\ldots,n \right)$$

with $x$ the model *state* (the unknowns to solve for) and $p$ the model *parameters*.  Given the model and an *initial condition* $x_0$ we can define a sequence $\mathcal{X} = \left(x_0, x_1, \ldots, x_n \right)$ by recursively solving the nonlinear equation for $x_n$.

While this form seems abstract, it actually describes a large number of interesting and useful models.  For example, consider the ordinary differential equation defined by

$$\dot{x} = g\left(x; p \right)$$

$$x(0) = x_0$$

We can convert this into a nonlinear recursive equation by applying a numerical time integration scheme, for example the [backward Euler method](https://en.wikipedia.org/wiki/Backward_Euler_method):

$$x_{i} = x_{i-1} + g(x_i; p) \Delta t_i $$

This algebraic equation has our standard form for a nonlinear recursive model:

$$f\left(x_{i-1}, x_i; p \right) = x_i - x_{i-1} - g(x_i; p) \Delta t_i $$

However, defining our time series with an algebraic equation, rather than a differential equation, provides access to a range of models that cannot be expressed as ODEs, for example difference equations.

## Training building blocks

The goal of training is basically to find the parameters $p$ for a nonlinear recursive function $f$ and initial condition $x_0$ such that the resulting sequence $\mathcal{X}$ best matches a target series $\hat{\mathcal{X}}$.  At a minimum to train a model we need to efficiently generate the time series $\mathcal{X}$ for different parameter values and, often, for multiple targets.  Additionally, we often need the derivative of the sequence $\mathcal{X}$ with respect to the model parameters $p$.

pyzag provides a few building block methods for efficiently generating sequences and their derivatives:

1. pyzag can vectorize simulating the sequences both for independent instantiations of the same model (i.e. batch vectorization) but also by vectorizing over some number of steps $i$.  [This paper](https://arxiv.org/abs/2310.08649) describes the basic idea, but pyzag extends the concept to general nonlinear recursive models.  The advantage of the approach is that it can increase the calculation bandwith if batch parallelism alone is not enough to fully utilize the compute device.
2. pyzag implements the parameter gradient calculation with the adjoint method.  For long sequences this approach is much more memory efficient compared to automatic differentiation and is also generally more computationally efficient.
3. pyzag also provides several methods for solving the resulting batched, time-chunked nonlinear and linear equations and predictors for starting the nonlinear solves based on previously simulated pieces of the sequence.

## Deterministic and stochastic models

pyzag is built on top of [PyTorch](https://pytorch.org/), integrating the adjoint calculation into PyTorch AD.  Users can seemlessly define and train deterministic models using PyTorch primitives.

The library also provides helper classes to convert a deterministic model, defined as a nonlinear recursive relation implemented with a PyTorch model, into a statistical model using the [pyro](https://pyro.ai/) library.  Specifically, pyzag provides methods for automatically converting the deterministic model to a stochastic model by replacing determinsitc parameters with prior distributions as well as methods for converting models into a hierarchical statistical format to provide dependence across multiple sequences.
