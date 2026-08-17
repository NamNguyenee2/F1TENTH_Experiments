"""Differentiable JAX track geometry for PANOC objectives."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from baseline.auto_car_dynamics_ipopt import track


@dataclass(frozen=True)
class TrackData:
    s_grid:   jnp.ndarray
    x_grid:   jnp.ndarray
    y_grid:   jnp.ndarray
    phi_grid: jnp.ndarray
    p_max:    float


def build_track_data(n_grid: int = 2048) -> TrackData:
    s_grid_np = np.linspace(0.0, float(track.p_max), n_grid, dtype=np.float32)
    x_np, y_np = track.centerline_xy(s_grid_np)
    phi_np = np.unwrap(track.tangent_angle(s_grid_np)).astype(np.float32)
    return TrackData(
        s_grid  =jnp.asarray(s_grid_np, dtype=jnp.float32),
        x_grid  =jnp.asarray(x_np, dtype=jnp.float32),
        y_grid  =jnp.asarray(y_np, dtype=jnp.float32),
        phi_grid=jnp.asarray(phi_np, dtype=jnp.float32),
        p_max   =float(track.p_max),
    )


def eval_track(track_data: TrackData, p_values: jnp.ndarray):
    p_mod   = jnp.mod(p_values, track_data.p_max)
    x_ref   = jnp.interp(p_mod, track_data.s_grid, track_data.x_grid)
    y_ref   = jnp.interp(p_mod, track_data.s_grid, track_data.y_grid)
    phi_ref = jnp.interp(p_mod, track_data.s_grid, track_data.phi_grid)
    return x_ref, y_ref, phi_ref
