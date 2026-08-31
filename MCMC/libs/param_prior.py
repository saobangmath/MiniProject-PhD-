"""Configurable parameter priors for logistic regression SMC / model search."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.stats import laplace, norm

from .stage2_worker import LogisticRegParams, _to_named_params, _vector_to_named


class ParamPrior(ABC):
    """Prior p(alpha, betas | M) used inside SMC and the S(M) score."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short label for reporting."""

    @abstractmethod
    def log_pdf(self, params) -> jnp.ndarray:
        """Log prior density for one parameter vector."""

    def log_pdf_vec(self, vec_or_batch) -> jnp.ndarray:
        arr = jnp.asarray(vec_or_batch)
        if arr.ndim == 1:
            return self.log_pdf(_vector_to_named(arr))
        return jax.vmap(lambda row: self.log_pdf(_vector_to_named(row)))(arr)

    @abstractmethod
    def gen_initial_samples(self, no_samples: int, p: int, key) -> jnp.ndarray:
        """Draw SMC initial particles from the prior."""


@dataclass(frozen=True)
class GaussianParamPrior(ParamPrior):
    """Independent Gaussian priors: alpha ~ N(0, alpha_scale^2), beta_j ~ N(0, beta_scale^2)."""

    alpha_scale: float = 3.0
    beta_scale: float = 1.5
    label: str | None = None

    def __post_init__(self):
        if self.alpha_scale <= 0 or self.beta_scale <= 0:
            raise ValueError("scales must be > 0")

    @property
    def name(self) -> str:
        if self.label:
            return self.label
        return f"N(0,{self.alpha_scale:g}^2) + N(0,{self.beta_scale:g}^2)"

    def log_pdf(self, params) -> jnp.ndarray:
        p = _to_named_params(params)
        log_alpha = norm.logpdf(p.alpha, loc=0.0, scale=self.alpha_scale)
        if p.betas.size == 0:
            return log_alpha
        log_betas = jnp.sum(norm.logpdf(p.betas, loc=0.0, scale=self.beta_scale))
        return log_alpha + log_betas

    def gen_initial_samples(self, no_samples: int, p: int, key) -> jnp.ndarray:
        k1, k2 = random.split(key)
        alphas = self.alpha_scale * random.normal(k1, shape=(no_samples, 1))
        if p == 0:
            return alphas
        betas = self.beta_scale * random.normal(k2, shape=(no_samples, p))
        return jnp.concatenate([alphas, betas], axis=1)


@dataclass(frozen=True)
class LaplaceBetaPrior(ParamPrior):
    """alpha ~ N(0, alpha_scale^2), beta_j ~ Laplace(0, beta_scale)."""

    alpha_scale: float = 3.0
    beta_scale: float = 1.5
    label: str | None = None

    def __post_init__(self):
        if self.alpha_scale <= 0 or self.beta_scale <= 0:
            raise ValueError("scales must be > 0")

    @property
    def name(self) -> str:
        if self.label:
            return self.label
        return f"N(0,{self.alpha_scale:g}^2) + Laplace(0,{self.beta_scale:g})"

    def log_pdf(self, params) -> jnp.ndarray:
        p = _to_named_params(params)
        log_alpha = norm.logpdf(p.alpha, loc=0.0, scale=self.alpha_scale)
        if p.betas.size == 0:
            return log_alpha
        log_betas = jnp.sum(laplace.logpdf(p.betas, loc=0.0, scale=self.beta_scale))
        return log_alpha + log_betas

    def gen_initial_samples(self, no_samples: int, p: int, key) -> jnp.ndarray:
        k1, k2 = random.split(key)
        alphas = self.alpha_scale * random.normal(k1, shape=(no_samples, 1))
        if p == 0:
            return alphas
        # Inverse-CDF sampling for Laplace(0, b): b * sign(U-0.5) * log(1 - 2|U-0.5|)
        u = random.uniform(k2, shape=(no_samples, p))
        betas = self.beta_scale * jnp.sign(u - 0.5) * jnp.log1p(-2.0 * jnp.abs(u - 0.5))
        return jnp.concatenate([alphas, betas], axis=1)


# Paper / Stage-1 default
DEFAULT_PARAM_PRIOR = GaussianParamPrior(alpha_scale=3.0, beta_scale=1.5, label="baseline")


def make_target_logpdf_vec(param_prior: ParamPrior, X_observed, y_observed):
    from .stage2_worker import likelihood_log_pdf

    def posterior_log_pdf(params):
        return param_prior.log_pdf(params) + likelihood_log_pdf(params, X_observed, y_observed)

    def _single(vec):
        return posterior_log_pdf(_vector_to_named(vec))

    def target_logpdf_vec(vec_or_batch):
        arr = jnp.asarray(vec_or_batch)
        if arr.ndim == 1:
            return _single(arr)
        return jax.vmap(_single)(arr)

    return target_logpdf_vec
