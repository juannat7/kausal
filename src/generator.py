import abc
import numpy as np
import math
import random
import seaborn as sns
import matplotlib.pyplot as plt
from typing import *

import jax
import jax.numpy as jnp
import jax.random as rng
import jax_cfd.base as cfd

import torch
from torch import Tensor, Size
from torch.distributions import Normal, MultivariateNormal, Uniform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

r"""Markov chains"""
class MarkovChain(abc.ABC):
    r"""Abstract first-order time-invariant Markov chain class

    Wikipedia:
        https://wikipedia.org/wiki/Markov_chain
        https://wikipedia.org/wiki/Time-invariant_system
    """

    @abc.abstractmethod
    def prior(self, shape: Size = ()) -> Tensor:
        r""" x_0 ~ p(x_0) """

        pass

    @abc.abstractmethod
    def transition(self, x: Tensor) -> Tensor:
        r""" x_i ~ p(x_i | x_{i-1}) """

        pass

    def trajectory(self, x: Tensor, length: int, last: bool = False) -> Tensor:
        r""" (x_1, ..., x_n) ~ \prod_i p(x_i | x_{i-1}) """

        if last:
            for _ in range(length):
                x = self.transition(x)

            return x
        else:
            X = []

            for _ in range(length):
                x = self.transition(x)
                X.append(x)

            return torch.stack(X)


class DiscreteODE(abc.ABC):
    r"""Discretized ordinary differential equation (ODE)

    Wikipedia:
        https://wikipedia.org/wiki/Ordinary_differential_equation
    """

    def __init__(self, dt: float = 0.01):
        super().__init__()
        self.dt = dt

    @staticmethod
    def rk4(f: Callable[[Tensor], Tensor], x: Tensor, dt: float) -> Tensor:
        r"""Performs a step of the fourth-order Runge-Kutta integration scheme.

        Wikipedia:
            https://wikipedia.org/wiki/Runge-Kutta_methods
        """

        k1 = f(x)
        k2 = f(x + dt * k1 / 2)
        k3 = f(x + dt * k2 / 2)
        k4 = f(x + dt * k3)

        return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def transition(self, x: Tensor) -> Tensor:
        return self.rk4(self.f, x, self.dt)
        

def coupled_rossler(t, state, args):
    """
    Coupled Rossler dynamics solved 
        Reference: Equation 1

    Parameters:
        t: time
        state: system state (6 variables / degree of freedom)
        args: scalar parameters, including c1 c2 as the coupling terms

    Returns:
        tendency: system state tendencies
    """
    x1, y1, z1, x2, y2, z2 = state
    phi1, phi2, a, b, d, c1, c2 = args

    # Define the equations
    dx1 = -phi1 * y1 - z1
    dy1 = phi1 * x1 + a * y1 + c1 * (y2 - y1)
    dz1 = b + z1 * (x1 - d)

    dx2 = -phi2 * y2 - z2
    dy2 = phi2 * x2 + a * y2 + c2 * (y1 - y2)
    dz2 = b + z2 * (x2 - d)

    return jnp.array([dx1, dy1, dz1, dx2, dy2, dz2])


def lorenz96(t, state, args):
    """
    Lorenz '96 model dynamics.
    
    Parameters:
        t: float
            Current time (required by diffrax but not used in dynamics).
        state: jnp.ndarray
            Current state vector (N variables).
        args: tuple
            Extra arguments (e.g., forcing term F).

    Returns:
        dxdt: jnp.ndarray
            Time derivative of the state vector.
    """
    F = args[0]  # Forcing term
    N = state.shape[0]  # Number of variables

    # Compute the derivatives with periodic boundary conditions
    dxdt = (
        (state[(jnp.arange(N) + 1) % N] - state[(jnp.arange(N) - 2) % N]) 
        * state[(jnp.arange(N) - 1) % N]
        - state
        + F
    )
    return dxdt
    

class kolmogorov_flow(MarkovChain):
    r"""2-D fluid dynamics with Kolmogorov forcing

    Wikipedia:
        https://wikipedia.org/wiki/Navier-Stokes_equations
    """

    def __init__(
        self,
        size: int = 256,
        dt: float = 0.01,
        reynolds = None
    ):
        super().__init__()
        
        self.size = size
        self.dt = dt
        
        reynolds = torch.tensor(1e4) if reynolds == None else reynolds

        grid = cfd.grids.Grid(
            shape=(size, size),
            domain=((0, 2 * math.pi), (0, 2 * math.pi)),
        )

        bc = cfd.boundaries.periodic_boundary_conditions(2)

        forcing = cfd.forcings.simple_turbulence_forcing(
            grid=grid,
            constant_magnitude=5.0,
            constant_wavenumber=4.0,
            linear_coefficient=-0.1,
            forcing_type='kolmogorov',
        )
        
        dt_min = cfd.equations.stable_time_step(
            grid=grid,
            max_velocity=5.0,
            max_courant_number=0.5,
            viscosity=1 / reynolds.item(),
        )

        if dt_min > dt:
            steps = 1
        else:
            steps = math.ceil(dt / dt_min)

        step = cfd.funcutils.repeated(
            f=cfd.equations.semi_implicit_navier_stokes(
                grid=grid,
                forcing=forcing,
                dt=dt / steps,
                density=1.0,
                viscosity=1 / reynolds.item(),
            ),
            steps=steps,
        )

        def prior(key: rng.PRNGKey) -> jax.Array:
            u, v = cfd.initial_conditions.filtered_velocity_field(
                key,
                grid=grid,
                maximum_velocity=10.0,
                peak_wavenumber=4.0,
            )

            return jnp.stack((u.data, v.data))

        def transition(uv: jax.Array) -> jax.Array:
            u, v = cfd.initial_conditions.wrap_variables(
                var=tuple(uv),
                grid=grid,
                bcs=(bc, bc),
            )

            u, v = step((u, v))

            return jnp.stack((u.data, v.data))

        self._prior = jax.jit(jnp.vectorize(prior, signature='(K)->(C,H,W)'))
        self._transition = jax.jit(jnp.vectorize(transition, signature='(C,H,W)->(C,H,W)'))

    def prior(self, shape: Size = ()) -> Tensor:
        seed = random.randrange(2**32)

        key = rng.PRNGKey(seed)
        keys = rng.split(key, Size(shape).numel())
        keys = keys.reshape(*shape, -1)
        
        x = self._prior(keys)
        x = torch.tensor(np.asarray(x))

        return x

    def transition(self, x: Tensor) -> Tensor:
        x = x.detach().cpu().numpy()
        x = self._transition(x)
        x = torch.tensor(np.asarray(x))

        return x

    @staticmethod
    def coarsen(x: Tensor, r: int = 2) -> Tensor:
        *batch, h, w = x.shape

        x = x.reshape(*batch, h // r, r, w // r, r)
        x = x.mean(dim=(-3, -1))

        return x

    @staticmethod
    def upsample(x: Tensor, r: int = 2, mode: str = 'bilinear') -> Tensor:
        *batch, h, w = x.shape

        x = x.reshape(-1, 1, h, w)
        x = torch.nn.functional.pad(x, pad=(1, 1, 1, 1), mode='circular')
        x = torch.nn.functional.interpolate(x, scale_factor=(r, r), mode=mode)
        x = x[..., r:-r, r:-r]
        x = x.reshape(*batch, r * h, r * w)

        return x

    @staticmethod
    def vorticity(x):
        *batch, _, h, w = x.shape

        y = x.reshape(-1, 2, h, w)
        y = jnp.pad(y, pad_width=((0, 0), (0, 0), (1, 1), (1, 1)), mode="wrap")

        du = jnp.gradient(y[:, 0], axis=-1)
        dv = jnp.gradient(y[:, 1], axis=-2)

        y = du - dv
        y = y[:, 1:-1, 1:-1]
        y = y.reshape(*batch, 1, h, w)

        return y