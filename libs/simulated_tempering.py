"""Simulated Tempering on an SMC tempering ladder."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax import lax, random, vmap
from tqdm import trange

from .smc import MAGIC_CONST, SMC


@dataclass
class TemperingState:
    """ST Markov state: ladder index + configuration."""

    t_idx: int
    x: jnp.ndarray


class SimulatedTempering:
    """Simulated Tempering on an SMC tempering ladder.

    Pass an SMC that has already run ``build_intermediate_dists()``.
    Ladder data (``lambda_path``, ``log_z_path``, ``cov_path``, ``dims``,
    ``proposed_fn``) is read from that object.

    Moves
    -----
    explore : Gaussian MH with SMC ``Cov_t``, or SMC ``proposed_fn`` if set
    jump    : t' in {t-1, t+1} (boundary-aware proposal q)
    """

    def __init__(
        self,
        smc: SMC,
        seed: int = 0,
        max_cold_keep: int = 64,
        cov_reg: float = 1e-6,
        step_size: float = 0.1,
    ):
        if smc.lambda_path is None or smc.log_z_path is None:
            raise ValueError(
                "SMC has no ladder yet; call smc.build_intermediate_dists() first"
            )
        self.smc = smc
        self.lambda_path = jnp.asarray(smc.lambda_path, dtype=float)
        self.log_z_path = jnp.asarray(smc.log_z_path, dtype=float)
        if self.lambda_path.shape != self.log_z_path.shape:
            raise ValueError("lambda_path and log_z_path must have the same length")
        if int(self.lambda_path.shape[0]) < 2:
            raise ValueError("need at least 2 ladder levels")
        self.n_levels = int(self.lambda_path.shape[0])
        self.t_cold = self.n_levels - 1
        self.t_hot = 0
        self.dims = int(smc.dims)
        self.proposed_fn = smc.proposed_fn
        self.max_cold_keep = int(max_cold_keep)
        self.cov_reg = float(cov_reg)
        self.step_size = float(step_size)
        self._key = random.key(seed)

        if self.proposed_fn is not None:
            self.covs = None
        elif smc.cov_path is not None:
            self.covs = jnp.asarray(smc.cov_path, dtype=float)
            expected = (self.n_levels, self.dims, self.dims)
            if self.covs.shape != expected:
                raise ValueError(f"cov_path shape {self.covs.shape} != {expected}")
        else:
            eye = jnp.eye(self.dims)
            self.covs = jnp.stack(
                [(self.step_size**2) * eye for _ in range(self.n_levels)]
            )

    def log_pi_t(self, t_idx, x):
        lam = self.lambda_path[t_idx]
        log_z = self.log_z_path[t_idx]
        return self.smc.get_intermediate_logpdf(x, lam) - log_z

    def proposal_cov(self, t_idx):
        """Magic-scaled proposal covariance at level t."""
        d = self.dims
        cov_t = self.covs[t_idx]
        return (MAGIC_CONST**2 / d) * cov_t + self.cov_reg * jnp.eye(d)

    def explore(self, t_idx, x, key, covs=None):
        """MH explore at fixed t (Gaussian or custom ``proposed_fn``)."""
        key, k_prop, k_u = random.split(key, 3)
        if self.proposed_fn is not None:
            x_prop = self.proposed_fn(x, k_prop)
        else:
            covs_use = self.covs if covs is None else covs
            d = self.dims
            prop_cov = (MAGIC_CONST**2 / d) * covs_use[t_idx] + self.cov_reg * jnp.eye(d)
            chol = jnp.linalg.cholesky(prop_cov)
            x_prop = x + chol @ random.normal(k_prop, shape=(d,))
        log_alpha = self.log_pi_t(t_idx, x_prop) - self.log_pi_t(t_idx, x)
        accept = jnp.log(random.uniform(k_u)) < jnp.minimum(log_alpha, 0.0)
        x_next = jnp.where(accept, x_prop, x)
        return x_next, key

    def jump(self, t_idx, x, key):
        key, k_dir, k_u = random.split(key, 3)
        T = self.n_levels - 1
        u = random.uniform(k_dir)
        t_prop = jnp.where(
            t_idx == 0,
            1,
            jnp.where(t_idx == T, T - 1, jnp.where(u < 0.5, t_idx - 1, t_idx + 1)),
        )
        q_fwd = jnp.where((t_idx == 0) | (t_idx == T), 1.0, 0.5)
        q_rev = jnp.where((t_prop == 0) | (t_prop == T), 1.0, 0.5)
        log_alpha = (
            self.log_pi_t(t_prop, x)
            - self.log_pi_t(t_idx, x)
            + jnp.log(q_rev)
            - jnp.log(q_fwd)
        )
        accept = jnp.log(random.uniform(k_u)) < jnp.minimum(log_alpha, 0.0)
        return jnp.where(accept, t_prop, t_idx), key

    def move(self, t_idx, x, key, covs=None):
        x, key = self.explore(t_idx, x, key, covs=covs)
        t_idx, key = self.jump(t_idx, x, key)
        return t_idx, x, key

    def run_excursion(self, x0, key, max_steps: int = 2_000, covs=None):
        """One hot→…→hot excursion; uses frozen per-level covs for explore."""
        x0 = jnp.asarray(x0, dtype=float).reshape((self.dims,))
        covs = self.covs if covs is None else covs
        cold_buf0 = jnp.zeros((self.max_cold_keep, self.dims), dtype=x0.dtype)
        visits0 = jnp.zeros((self.n_levels,), dtype=jnp.int32)

        def scan_step(carry, _):
            t_idx, x, key, done, cold_buf, cold_n, visits = carry
            key, k_move = random.split(key)

            def do_move(_):
                return self.move(t_idx, x, k_move, covs=covs)

            def stay(_):
                return t_idx, x, key

            t_new, x_new, key_new = lax.cond(done, stay, do_move, operand=None)
            # Done once we return to hot from a non-hot level.
            done_new = done | ((t_idx != self.t_hot) & (t_new == self.t_hot))
            was_live = ~done
            visits_new = visits.at[t_new].add(was_live.astype(jnp.int32))
            store = was_live & (t_new == self.t_cold) & (cold_n < self.max_cold_keep)
            cold_buf_new = jnp.where(store, cold_buf.at[cold_n].set(x_new), cold_buf)
            cold_n_new = cold_n + store.astype(jnp.int32)
            new_carry = (
                t_new,
                x_new,
                key_new,
                done_new,
                cold_buf_new,
                cold_n_new,
                visits_new,
            )
            return new_carry, was_live

        init = (
            jnp.asarray(self.t_hot, dtype=jnp.int32),
            x0,
            key,
            jnp.asarray(False),
            cold_buf0,
            jnp.asarray(0, dtype=jnp.int32),
            visits0,
        )
        (_t, _x, _k, _d, cold_buf, cold_n, visits), live = lax.scan(
            scan_step, init, xs=None, length=max_steps
        )
        n_steps = jnp.sum(live.astype(jnp.int32))
        return cold_buf, cold_n, n_steps, visits

    def run(
        self,
        n_excursions: int = 100_000,
        max_steps: int = 2_000,
        key=None,
    ):
        """Generate all x0's and keys, then vmap run_excursion with frozen Cov_t."""
        if key is None:
            self._key, key = random.split(self._key)

        key, k_x0, k_exc = random.split(key, 3)
        x0s = random.normal(k_x0, shape=(n_excursions, self.dims))
        keys = random.split(k_exc, n_excursions)
        covs = self.covs

        cold_buf, cold_n, n_steps, visits = vmap(
            lambda x0, k: self.run_excursion(x0, k, max_steps=max_steps, covs=covs)
        )(x0s, keys)

        ids = jnp.arange(self.max_cold_keep)[None, :]
        mask = ids < cold_n[:, None]
        cold_samples = cold_buf[mask]
        visit_counts = jnp.sum(visits, axis=0)
        return cold_samples, n_steps, visit_counts

    def run_and_stop_at_next_cold(
        self,
        x0,
        n_iters: int = 5_000,
        key=None,
        score_fn=None,
        start_at_hot: bool = True,
    ):
        """Sequential ST for n_iters; useful for discrete targets (e.g. Sudoku).

        Prints only when ``score_fn`` improves. Stops early if score hits 0.
        """
        if key is None:
            self._key, key = random.split(self._key)
        x = jnp.asarray(x0)
        t0 = self.t_hot if start_at_hot else self.t_cold
        t_idx = jnp.asarray(t0, dtype=jnp.int32)
        history = []
        x_best = None
        best_score = jnp.inf
        found = False

        for i in trange(n_iters):
            key, k_move = random.split(key)
            t_idx, x, key = self.move(t_idx, x, k_move, covs=self.covs)
            t_int = int(t_idx)

            if score_fn is None:
                if t_int == self.t_cold:
                    history.append({"step": i, "t_idx": t_int, "score": None})
                    x_best = x
                continue

            s_val = float(score_fn(x))
            if s_val < float(best_score):
                best_score = s_val
                x_best = x
                history.append(
                    {"step": i, "t_idx": t_int, "score": s_val, "x": np.asarray(x)}
                )
                print(f"[chain] new best score={s_val:.0f} at step {i+1} (t={t_int})")
                print(np.asarray(x).reshape(9, 9) if x.size == 81 else np.asarray(x))

            if s_val == 0.0:
                found = True
                break

        return x_best, history, found
