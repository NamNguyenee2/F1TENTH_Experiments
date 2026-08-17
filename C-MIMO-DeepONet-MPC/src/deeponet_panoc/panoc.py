"""JAX PANOC implementation for box-constrained control sequences."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class PANOCConfig:
    gamma: float = 1e-2
    max_iter: int = 100
    tol: float = 1e-3
    lbfgs_memory: int = 10
    max_line_search: int = 20
    armijo_c: float = 1e-4


def project_box(u_flat: jnp.ndarray, lower: jnp.ndarray, upper: jnp.ndarray) -> jnp.ndarray:
    return jnp.clip(u_flat, lower, upper)


def fixed_point_residual(objective_grad, u_flat, lower, upper, gamma):
    grad = objective_grad(u_flat)
    z = project_box(u_flat - gamma * grad, lower, upper)
    return (u_flat - z) / gamma, z, grad


def lbfgs_two_loop(residual, s_hist, y_hist, valid_hist, head):
    mem_size = s_hist.shape[0]
    order = jnp.mod(head - 1 - jnp.arange(mem_size), mem_size)

    def backward(carry, idx):
        q, alpha = carry
        s_i = s_hist[idx]
        y_i = y_hist[idx]
        valid = valid_hist[idx]
        ys = jnp.dot(y_i, s_i)
        rho = jnp.where((ys > 1e-12) & valid, 1.0 / ys, 0.0)
        a_i = rho * jnp.dot(s_i, q)
        q = q - a_i * y_i
        return (q, alpha.at[idx].set(a_i)), None

    alpha0 = jnp.zeros((mem_size,), dtype=residual.dtype)
    (q, alpha), _ = jax.lax.scan(backward, (residual, alpha0), order)

    newest = order[0]
    ys_new = jnp.dot(y_hist[newest], s_hist[newest])
    yy_new = jnp.dot(y_hist[newest], y_hist[newest])
    scale = jnp.where((yy_new > 1e-12) & valid_hist[newest], ys_new / yy_new, 1.0)
    r = scale * q

    def forward(carry, idx):
        r_i = carry
        s_i = s_hist[idx]
        y_i = y_hist[idx]
        valid = valid_hist[idx]
        ys = jnp.dot(y_i, s_i)
        rho = jnp.where((ys > 1e-12) & valid, 1.0 / ys, 0.0)
        beta = rho * jnp.dot(y_i, r_i)
        r_i = r_i + s_i * (alpha[idx] - beta)
        return r_i, None

    r, _ = jax.lax.scan(forward, r, order[::-1])
    return r


def make_panoc_solver(objective_fn, lower, upper, config: PANOCConfig, value_and_grad_fn=None):
    lower = jnp.asarray(lower, dtype=jnp.float32)
    upper = jnp.asarray(upper, dtype=jnp.float32)
    tau_values = 0.5 ** jnp.arange(config.max_line_search, dtype=jnp.float32)
    if value_and_grad_fn is None:
        value_and_grad_fn = lambda u, *args: jax.value_and_grad(lambda z: objective_fn(z, *args))(u)

    @partial(jax.jit, static_argnames=())
    def solve(u0_flat, *objective_args):
        u0_flat = project_box(jnp.asarray(u0_flat, dtype=jnp.float32), lower, upper)
        objective_at_args = lambda u: objective_fn(u, *objective_args)
        value_and_grad_at_args = lambda u: value_and_grad_fn(u, *objective_args)
        obj0, grad0 = value_and_grad_at_args(u0_flat)
        z0 = project_box(u0_flat - config.gamma * grad0, lower, upper)
        r0 = (u0_flat - z0) / config.gamma
        n = u0_flat.shape[0]
        s_hist0 = jnp.zeros((config.lbfgs_memory, n), dtype=jnp.float32)
        y_hist0 = jnp.zeros((config.lbfgs_memory, n), dtype=jnp.float32)
        valid_hist0 = jnp.zeros((config.lbfgs_memory,), dtype=bool)

        init_state = {
            "u": u0_flat,
            "obj": obj0,
            "r": r0,
            "s_hist": s_hist0,
            "y_hist": y_hist0,
            "valid_hist": valid_hist0,
            "head": jnp.asarray(0, dtype=jnp.int32),
            "done": jnp.linalg.norm(r0) <= config.tol,
            "iters": jnp.asarray(0, dtype=jnp.int32),
        }
        initial_residual_norm = jnp.linalg.norm(r0)

        def active_step(state, _):
            u = state["u"]
            r = state["r"]
            obj = state["obj"]

            h_r = lbfgs_two_loop(r, state["s_hist"], state["y_hist"], state["valid_hist"], state["head"])
            d_lbfgs = -h_r
            d_pg = -config.gamma * r
            has_memory = jnp.any(state["valid_hist"])
            d = jnp.where(has_memory, d_lbfgs, d_pg)

            def candidate_value(args):
                tau, use_pg = args
                direction = jnp.where(use_pg, d_pg, d)
                u_cand = project_box(u + tau * direction, lower, upper)
                return objective_at_args(u_cand), u_cand

            # Try the L-BFGS/PANOC direction and a projected-gradient fallback.
            candidate_tau = jnp.concatenate([tau_values, tau_values, jnp.zeros((1,), dtype=jnp.float32)])
            candidate_is_pg = jnp.concatenate([
                jnp.zeros_like(tau_values, dtype=bool),
                jnp.ones_like(tau_values, dtype=bool),
                jnp.ones((1,), dtype=bool),
            ])
            values, candidates = jax.vmap(candidate_value)((candidate_tau, candidate_is_pg))
            direction_decrease = config.armijo_c * candidate_tau * jnp.sum(r * r)
            accepted = jnp.isfinite(values) & (values <= obj - direction_decrease)
            accepted = accepted.at[-1].set(True)
            any_accepted = jnp.any(accepted)
            first_accepted = jnp.argmax(accepted)
            best_idx = jnp.argmin(values)
            chosen_idx = jnp.where(any_accepted, first_accepted, best_idx)
            u_next = candidates[chosen_idx]
            obj_next, grad_next = value_and_grad_at_args(u_next)
            z_next = project_box(u_next - config.gamma * grad_next, lower, upper)
            r_next = (u_next - z_next) / config.gamma

            s = u_next - u
            y = r_next - r
            ys = jnp.dot(y, s)
            update_ok = (ys > 1e-10) & jnp.isfinite(ys)
            head = state["head"]
            s_hist = state["s_hist"].at[head].set(jnp.where(update_ok, s, state["s_hist"][head]))
            y_hist = state["y_hist"].at[head].set(jnp.where(update_ok, y, state["y_hist"][head]))
            valid_hist = state["valid_hist"].at[head].set(update_ok | state["valid_hist"][head])
            head_next = jnp.mod(head + jnp.asarray(1, dtype=jnp.int32), config.lbfgs_memory)
            r_norm = jnp.linalg.norm(r_next)

            return {
                "u": u_next,
                "obj": obj_next,
                "r": r_next,
                "s_hist": s_hist,
                "y_hist": y_hist,
                "valid_hist": valid_hist,
                "head": head_next,
                "done": r_norm <= config.tol,
                "iters": state["iters"] + jnp.asarray(1, dtype=jnp.int32),
            }, None

        def body(state, idx):
            return jax.lax.cond(
                state["done"],
                lambda st: (st, None),
                lambda st: active_step(st, idx),
                state,
            )

        final_state, _ = jax.lax.scan(body, init_state, jnp.arange(config.max_iter))
        return {
            "u": final_state["u"],
            "objective": final_state["obj"],
            "initial_residual_norm": initial_residual_norm,
            "residual_norm": jnp.linalg.norm(final_state["r"]),
            "iterations": final_state["iters"],
            "converged": final_state["done"],
        }

    return solve
