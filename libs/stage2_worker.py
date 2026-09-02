"""Process workers for Stage 2 bitmask SMC.

Spawned processes must import this module (no notebook closures).
JAX is forced onto CPU in each worker so jobs do not contend for one GPU.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.scipy.stats import norm
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .smc import SMC

LogisticRegParams = namedtuple("LogisticRegParams", ["alpha", "betas"])

_WORKER_DATA = None


def _to_named_params(params):
    if isinstance(params, LogisticRegParams):
        alpha = jnp.asarray(params.alpha).reshape(())
        betas = jnp.ravel(jnp.asarray(params.betas))
        return LogisticRegParams(alpha=alpha, betas=betas)
    flat = jnp.ravel(jnp.asarray(params))
    return LogisticRegParams(alpha=flat[0], betas=flat[1:])


def prior_log_pdf(params):
    p = _to_named_params(params)
    log_alpha = norm.logpdf(p.alpha, loc=0.0, scale=3.0)
    log_betas = jnp.sum(norm.logpdf(p.betas, loc=0.0, scale=1.5))
    return log_alpha + log_betas


def likelihood_log_pdf(params, X_observed, y_observed):
    p = _to_named_params(params)
    X_observed = jnp.atleast_2d(jnp.asarray(X_observed))
    y_observed = jnp.ravel(jnp.asarray(y_observed))
    eta = p.alpha + jnp.matmul(X_observed, p.betas[:, None]).squeeze(-1)
    return jnp.sum(y_observed * eta - jnp.logaddexp(0.0, eta))


def posterior_log_pdf(params, X_observed, y_observed):
    return prior_log_pdf(params) + likelihood_log_pdf(params, X_observed, y_observed)


def proposed_fn(params, key):
    p = _to_named_params(params)
    key1, key2 = random.split(key)
    new_alpha = p.alpha + 0.1 * random.normal(key1)
    new_betas = p.betas + 0.1 * random.normal(key2, shape=p.betas.shape)
    return LogisticRegParams(alpha=new_alpha, betas=new_betas)


def gen_initial_samples(no_samples: int, p: int, key):
    k1, k2 = random.split(key)
    alphas = 3.0 * random.normal(k1, shape=(no_samples, 1))
    if p == 0:
        return alphas
    betas = 1.5 * random.normal(k2, shape=(no_samples, p))
    return jnp.concatenate([alphas, betas], axis=1)


def _vector_to_named(vec):
    vec = jnp.ravel(vec)
    return LogisticRegParams(alpha=vec[0], betas=vec[1:])


def _target_logpdf_single(vec, X_observed, y_observed):
    return posterior_log_pdf(_vector_to_named(vec), X_observed, y_observed)


def _prior_logpdf_single(vec):
    return prior_log_pdf(_vector_to_named(vec))


def target_logpdf_vec(vec_or_batch, X_observed, y_observed):
    arr = jnp.asarray(vec_or_batch)
    if arr.ndim == 1:
        return _target_logpdf_single(arr, X_observed, y_observed)
    return jax.vmap(lambda row: _target_logpdf_single(row, X_observed, y_observed))(arr)


def prior_logpdf_vec(vec_or_batch):
    arr = jnp.asarray(vec_or_batch)
    if arr.ndim == 1:
        return _prior_logpdf_single(arr)
    return jax.vmap(_prior_logpdf_single)(arr)


def proposed_fn_vec(vec, key):
    out = proposed_fn(_vector_to_named(vec), key)
    return jnp.concatenate([jnp.array([out.alpha]), jnp.ravel(out.betas)])


def posterior_predictive_probs(particles, X):
    particles = np.asarray(particles)
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    alpha = particles[:, 0]
    betas = particles[:, 1:]
    if betas.shape[1] == 0:
        logits = np.broadcast_to(alpha[:, None], (alpha.shape[0], X.shape[0])).copy()
    else:
        logits = alpha[:, None] + betas @ X.T
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
    return probs.mean(axis=0)


def eval_posterior_predictive(particles, X, y):
    p_hat = posterior_predictive_probs(particles, X)
    y = np.ravel(np.asarray(y))
    y_hat = (p_hat >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y, p_hat))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y, y_hat)),
        "roc_auc": auc,
        "brier": float(brier_score_loss(y, p_hat)),
        "log_loss": float(log_loss(y, np.clip(p_hat, 1e-15, 1 - 1e-15))),
        "log_score": float(
            np.sum(
                y * np.log(np.clip(p_hat, 1e-15, 1 - 1e-15))
                + (1 - y) * np.log(np.clip(1 - p_hat, 1e-15, 1 - 1e-15))
            )
        ),
    }


def run_smc_on_data(X_obs, y_obs, no_samples=5_000, seed=42, param_prior=None):
    from .param_prior import DEFAULT_PARAM_PRIOR, make_target_logpdf_vec

    prior = param_prior or DEFAULT_PARAM_PRIOR
    X_obs = jnp.asarray(X_obs, dtype=float)
    y_obs = jnp.ravel(jnp.asarray(y_obs, dtype=float))
    p_obs = int(X_obs.shape[1])
    k_init, k_smc = random.split(random.key(seed))
    smc_run = SMC(
        dims=1 + p_obs,
        target_dist_logpdf=make_target_logpdf_vec(prior, X_obs, y_obs),
        prior_dist_logpdf=prior.log_pdf_vec,
        proposed_fn=proposed_fn_vec,
        key=k_smc,
    )
    smc_run.reset(samples=prior.gen_initial_samples(no_samples, p_obs, k_init))
    lam_path, log_z = smc_run.build_intermediate_dists()
    particles = np.array(smc_run.get_current_sample_list())
    return {
        "particles": particles,
        "log_evidence": float(log_z),
        "lambda_path": lam_path,
    }


def mask_to_set_indexes(mask: int, n_predictors: int):
    return [bit for bit in range(n_predictors) if (mask & (1 << bit)) > 0]


def init_worker(payload: dict):
    """Runs once per process. Stores shared train/test arrays."""
    global _WORKER_DATA
    _WORKER_DATA = payload


def run_mask(mask: int) -> dict:
    """Entry point for one model subset (called by the process pool)."""
    if _WORKER_DATA is None:
        raise RuntimeError("Worker was not initialized. Call init_worker first.")

    data = _WORKER_DATA
    names_all = list(data["predictor_names"])
    n_predictors = len(names_all)
    idx = mask_to_set_indexes(mask, n_predictors)
    names = [names_all[j] for j in idx]

    X_train = data["X_train"]
    X_test = data["X_test"]
    Xtr = X_train[:, idx] if idx else X_train[:, :0]
    Xte = X_test[:, idx] if idx else X_test[:, :0]

    smc_out = run_smc_on_data(
        Xtr,
        data["y_train"],
        no_samples=int(data["n_particles"]),
        seed=int(data["base_seed"]) + int(mask),
    )
    metrics = eval_posterior_predictive(smc_out["particles"], Xte, data["y_test"])

    return {
        "mask": int(mask),
        "variables": ", ".join(names) if names else "(intercept only)",
        "n_predictors": len(idx),
        "dimension": 1 + len(idx),
        "log_evidence": smc_out["log_evidence"],
        "n_lambda_steps": len(smc_out["lambda_path"]),
        "test_accuracy": metrics["accuracy"],
        "test_roc_auc": metrics["roc_auc"],
        "test_brier": metrics["brier"],
        "test_log_loss": metrics["log_loss"],
        "test_log_score": metrics["log_score"],
    }
