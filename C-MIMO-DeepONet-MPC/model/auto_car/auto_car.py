"""
Kinematics of the autonomous racing car (ARC).

State   x = [X, Y, phi, v]      (4-dim)
Input   u = [omega, a]          (2-dim)

Progress p and progress increment vartheta are separate variables in OP1/OP2.
"""

import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from pathlib import Path
import yaml



CONFIG_PATH = Path(__file__).with_name("mpcc_parameter.yaml")


with open(CONFIG_PATH, 'r') as file:
    param = yaml.safe_load(file)


DT: float = float(param["DT"])
ZETA_P: float = float(param["ZETA_P"])

# State bounds
V_MIN: float = float(param["V_MIN"])
V_MAX: float = float(param["V_MAX"])

# Input bounds
OMEGA_MIN: float = float(param["OMEGA_MIN"])
OMEGA_MAX: float = float(param["OMEGA_MAX"])
A_MIN: float = float(param["A_MIN"])
A_MAX: float = float(param["A_MAX"])


def _bicycle_dynamics_ct(xp: jnp.ndarray, _t: float, up: jnp.ndarray) -> jnp.ndarray:
    """Continuous-time bicycle kinematics with constant input over the interval."""
    _, _, phi, v = xp
    omega, a = up
    return jnp.array([
        v * jnp.cos(phi),
        v * jnp.sin(phi),
        omega,
        a,
    ])


@jax.jit
def f_phys(xp: jnp.ndarray, up: jnp.ndarray, zeta_p: float = ZETA_P) -> jnp.ndarray:
    """
    x_{t+1} = f_p(x_t, u_t; zeta_p)

    xp : [X, Y, phi, v]
    up : [omega, a]
    """
    X, Y, phi, v = xp
    omega, a = up
    return jnp.array([
        X + DT * v * jnp.cos(phi),
        Y + DT * v * jnp.sin(phi),
        phi + DT * omega,
        v + DT * zeta_p * a,
    ])


@jax.jit
def f_phys_ct(xp: jnp.ndarray, up: jnp.ndarray, t: float = DT) -> jnp.ndarray:
    """
    Integrate the continuous-time dynamics to get x(t).

    xp : [X, Y, phi, v]
    up : [omega, a]
    t  : integration time [s]
    """
    ts = jnp.array([0.0, t])
    return odeint(_bicycle_dynamics_ct, xp, ts, up)[-1]


class AutoCar:
    DT: float = DT
    ZETA_P: float = ZETA_P
    
    # State bounds
    V_MIN: float = V_MIN
    V_MAX: float = V_MAX

    # Input bounds
    OMEGA_MIN: float = OMEGA_MIN
    OMEGA_MAX: float = OMEGA_MAX
    A_MIN: float = A_MIN
    A_MAX: float = A_MAX

    f_phys = staticmethod(f_phys)
    f_phys_ct = staticmethod(f_phys_ct)
