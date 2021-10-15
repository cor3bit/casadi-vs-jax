import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from jax import jacfwd, jacrev
import jax.numpy as jnp
from jax.config import config
from jax.experimental.ode import odeint

VERBOSE = 0

A = jnp.array([
    [-16., 12.],
    [12., -9.],
], dtype=jnp.float64)

T_matrix = jnp.array([
    [16., -13.],
    [-11., 9.],
], dtype=jnp.float64)


def dynamics(y, t) -> jnp.ndarray:
    T_vec = jnp.array([jnp.cos(t), jnp.sin(t)], dtype=jnp.float64)
    return A @ y + T_matrix @ T_vec


def jax_int():
    def _integrate_last(y0_inner, time_grid_inner):
        xs_inner = odeint(dynamics, y0_inner, time_grid_inner)
        xs_last = xs_inner[-1]
        return xs_last

    # params
    y0 = jnp.array([1, 0], dtype=jnp.float64)
    n_steps = 21
    t_start = 0
    t_end = 0.1
    # h = (t_end - t_start) / (n_steps - 1)
    time_grid = jnp.linspace(t_start, t_end, num=n_steps)

    xs = _integrate_last(y0, time_grid)
    G_x = jacrev(_integrate_last)(y0, time_grid)

    print(xs)
    print(G_x)


def casadi_int():
    # TODO
    pass

    # # Create an integrator
    # dae = {'x': x, 'z': z, 'p': u, 'ode': f_x, 'alg': f_z, 'quad': f_q}
    # opts = {"tf": 0.5}  # interval length
    # I = ca.integrator('I', "idas", dae, opts)


def run():
    config.update('jax_enable_x64', True)

    jax_int()

    casadi_int()


# -------------------------- Runner --------------------------

if __name__ == '__main__':
    run()
