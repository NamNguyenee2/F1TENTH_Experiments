"""JAX C-MIMO DeepONet + PANOC MPC prototype."""

from .ipopt import make_deeponet_ipopt_solver
from .mpcc import (
    DEEPONET_MODELS,
    MPCCPANOCConfig,
    make_deeponet_panoc_solver,
)
from .panoc import PANOCConfig
from .track_jax import build_track_data