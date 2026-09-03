from functools import partial
from tqdm import trange

import jax
import jax.numpy as jnp
from jax import lax, random
from jax.scipy.stats import multivariate_normal as jax_mvn

from .mcmc import MCMC, AcceptanceTracker

MAGIC_CONST = 2.38
ESS_THRESHOLD = 0.75

class SMC:
    def __init__(self, dims, target_dist_logpdf, prior_dist_logpdf = None, proposed_fn = None, key = random.key(0)):
        self.__dims = dims
        self.__prior_dist_logpdf = prior_dist_logpdf
        self.__target_dist_logpdf = target_dist_logpdf
        self.__inter_dist_logpdfs = [target_dist_logpdf]
        self.__next_dist_id = 0
        self.__cur_samples = None
        self.__cur_samples_logpdf = None
        self.__proposed_fn = proposed_fn
        self.__key = key

    def _split_key(self):
        self.__key, sub_key = random.split(self.__key)
        return sub_key

    def _base_logpdf(self, x):
        if self.__prior_dist_logpdf:
            return self.__prior_dist_logpdf(x)
                
        return jax_mvn.logpdf(
            x, mean=jnp.zeros(self.__dims), cov=jnp.eye(self.__dims)
        )

    def get_intermediate_logpdf(self, x, _lambda):
        r"""log pdf of pi_0^(1-lambda) * pi_1^lambda"""
        return (
            (1.0 - _lambda) * self._base_logpdf(x)
            + _lambda * self.__target_dist_logpdf(x)
        )

    def add_intermediate(self, inter_dist_logpdf):
        self.__inter_dist_logpdfs.append(inter_dist_logpdf)

    def get_current_sample_list(self):
        return self.__cur_samples

    def reset(self, no_samples: int | None = None, samples = None):
        self.__next_dist_id = len(self.__inter_dist_logpdfs) - 1
        key = self._split_key()
        if samples is not None: 
            self.__cur_samples = samples 
        else: 
            self.__cur_samples = random.multivariate_normal(
                key,
                mean=jnp.zeros(self.__dims),
                cov=jnp.eye(self.__dims),
                shape=(no_samples,),
            )
        self.__cur_samples_logpdf = self._base_logpdf(self.__cur_samples)

    def sample_loop(self, no_samples: int):
        self.reset(no_samples)
        for _ in trange(len(self.__inter_dist_logpdfs)):
            self.__sample_next_dist(self.__inter_dist_logpdfs[self.__next_dist_id])
            self.__next_dist_id -= 1

    # ------------------------------------------------------------------
    # ESS helpers (pure JAX)
    # ------------------------------------------------------------------
    def __relative_ess(self, log_ratio, lambda_diff):
        r"""Relative ESS in [0, 1] for weights ∝ exp(lambda_diff * log_ratio)."""
        log_w = lambda_diff * log_ratio
        w = jnp.exp(log_w - jax.nn.logsumexp(log_w))
        return 1.0 / (w.shape[0] * jnp.sum(w * w))

    def __ess(self, samples, log_pi_0, log_pi_1, lambda_diff):
        log_ratio = log_pi_1(samples) - log_pi_0(samples)
        return self.__relative_ess(log_ratio, lambda_diff)

    def __find_next_lambda(self, log_ratio, lambda_cur, n_bisect=30):
        r"""
        Binary search (lax.fori_loop): largest m in [lambda_cur, 1] such that
        ESS(m - lambda_cur) >= ESS_THRESHOLD.
        """
        def body(_, state):
            lo, hi = state
            mid = 0.5 * (lo + hi)
            # IMPORTANT: delta is always from fixed lambda_cur, not from lo
            ok = self.__relative_ess(log_ratio, mid - lambda_cur) >= ESS_THRESHOLD
            new_lo = jnp.where(ok, mid, lo)
            new_hi = jnp.where(ok, hi, mid)
            return new_lo, new_hi

        lo, _ = lax.fori_loop(0, n_bisect, body, (lambda_cur, 1.0))
        return jnp.clip(lo, lambda_cur + 1e-6, 1.0)

    # ------------------------------------------------------------------
    # One tempering step: reweight -> resample -> MCMC
    # ------------------------------------------------------------------
    def __temper_step(self, samples, cur_logpdf, key, lam, mcmc_iters=10):
        next_logpdf = lambda x: self.get_intermediate_logpdf(x, lam)

        log_w = next_logpdf(samples) - cur_logpdf
        weight = jnp.exp(log_w - jax.nn.logsumexp(log_w))

        # log(z_(i+1)) - log(z_i) = log(1 / N * sigma weight)
        diff_log_z = jax.nn.logsumexp(log_w) - jnp.log(samples.shape[0])

        key, k_resample, k_mcmc = random.split(key, 3)
        n = samples.shape[0]
        idx = random.choice(k_resample, a=n, shape=(n,), p=weight, replace=True)
        samples = samples[idx]

        proposed_fn = self._get_proposed_fn(samples)
        mcmc = MCMC(next_logpdf, proposed_fn, tracker=None)
        next_samples = mcmc.run_batch(samples, k_mcmc, iters=mcmc_iters)
        return next_samples, next_logpdf(next_samples), diff_log_z, key

    def __sample_next_dist(self, next_dist_logpdf):
        r"""Manual-schedule step used by sample_loop."""
        no_samples = self.__cur_samples.shape[0]

        log_w = next_dist_logpdf(self.__cur_samples) - self.__cur_samples_logpdf
        weight = jnp.exp(log_w - jax.nn.logsumexp(log_w))

        key = self._split_key()
        indexes = random.choice(
            key, a=no_samples, shape=(no_samples,), p=weight, replace=True
        )
        next_samples = self.__cur_samples[indexes]
        
        proposed_fn = self._get_proposed_fn(next_samples)
        mcmc = MCMC(next_dist_logpdf, proposed_fn, tracker=None)
        key = self._split_key()
        next_samples = mcmc.run_batch(next_samples, key, iters=10)

        self.__cur_samples = next_samples
        self.__cur_samples_logpdf = next_dist_logpdf(next_samples)

    def build_intermediate_dists(
        self, max_steps=64, n_bisect=30, mcmc_iters=10, verbose=False
    ):
        r"""
        Build pi_0 -> ... -> pi_T adaptively.

        Outer loop:  lax.while_loop  (keep adding lambda while ESS to 1 is low)
        Inner search: lax.fori_loop  (binary search next lambda by ESS)

        Parameters
        ----------
        verbose : bool, default False
            If True, print current lambda at each adaptive tempering step via
            ``jax.debug.print`` (visible during JIT / lax loop execution).

        Returns
        -------
        lambda_list : list[float]
            Tempering schedule ``[0 = lambda_0, ..., 1 = lambda_T]``.
        log_z_list : list[float]
            Cumulative normalizing constants ``log Z_i`` for each intermediate
            distribution (``log_z_list[i]`` pairs with ``lambda_list[i]``).
            Assumes the base/prior is normalized, so ``log Z_0 = 0``.
        tot_log_z : float
            Final evidence estimate ``log Z_T`` (same as ``log_z_list[-1]``).
        """
        base = self._base_logpdf
        target = self.__target_dist_logpdf

        # Parallel arrays: lambdas[i] <-> log_zs[i] = log Z_{lambda_i}
        # Convention: prior/base is normalized => log Z_0 = 0 at lambda=0.
        lambdas = jnp.full((max_steps + 2,), jnp.nan).at[0].set(0.0)
        log_zs = jnp.full((max_steps + 2,), jnp.nan).at[0].set(0.0)
        init = (
            self.__cur_samples,
            self.__cur_samples_logpdf,
            self.__key,
            jnp.asarray(0.0),   # lambda_cur
            jnp.asarray(0),     # step index
            lambdas,
            log_zs,
            jnp.asarray(0.0),   # tot_diff_log_z (== log_zs[t] while running)
        )

        def cond(carry):
            samples, _, _, lam, t, _, _, _ = carry
            log_ratio = target(samples) - base(samples)
            ess_to_one = self.__relative_ess(log_ratio, 1.0 - lam)
            return (ess_to_one < ESS_THRESHOLD) & (t < max_steps) & (lam < 1.0 - 1e-5)

        def body(carry):
            samples, cur_lp, key, lam, t, lambdas, log_zs, tot_diff_log_z = carry
            log_ratio = target(samples) - base(samples)
            lam_next = self.__find_next_lambda(log_ratio, lam, n_bisect=n_bisect)
            samples, cur_lp, diff_log_z, key = self.__temper_step(
                samples, cur_lp, key, lam_next, mcmc_iters=mcmc_iters
            )
            if verbose:
                jax.debug.print(
                    "[SMC] step {t}: lambda={lam} -> {lam_next}, diff_log_z={dz}",
                    t=t,
                    lam=lam,
                    lam_next=lam_next,
                    dz=diff_log_z,
                )
            tot_next = tot_diff_log_z + diff_log_z
            lambdas = lambdas.at[t + 1].set(lam_next)
            log_zs = log_zs.at[t + 1].set(tot_next)
            return samples, cur_lp, key, lam_next, t + 1, lambdas, log_zs, tot_next

        samples, cur_lp, key, lam, t, lambdas, log_zs, tot_diff_log_z = lax.while_loop(
            cond, body, init
        )

        # final jump to lambda = 1
        if verbose:
            print(f"[SMC] final step: lambda={float(lam)} -> 1.0")
        samples, cur_lp, diff_log_z, key = self.__temper_step(
            samples, cur_lp, key, 1.0, mcmc_iters=mcmc_iters
        )
        tot_diff_log_z = tot_diff_log_z + diff_log_z
        lambdas = lambdas.at[t + 1].set(1.0)
        log_zs = log_zs.at[t + 1].set(tot_diff_log_z)

        self.__cur_samples = samples
        self.__cur_samples_logpdf = cur_lp
        self.__key = key

        valid = ~jnp.isnan(lambdas)
        lambda_list = [float(x) for x in lambdas[valid]]
        log_z_list = [float(x) for x in log_zs[valid]]
        # (lambdas, intermediate log Z_i, total log Z_T)
        return lambda_list, log_z_list, tot_diff_log_z

    def _get_proposed_fn(self, samples):
        r"""
            get proposed fn to move samples
        """
        
        # incase there is a proposed_fn already define for the specific problem domain
        if self.__proposed_fn is not None:
            return self.__proposed_fn

        move_cov = jnp.atleast_2d(jnp.cov(samples, rowvar=False))
        move_cov = (MAGIC_CONST / self.__dims) * move_cov
        mean0 = jnp.zeros(self.__dims)

        return lambda x, key: x + random.multivariate_normal(key, mean=mean0, cov=move_cov)

if __name__ == "__main__":
    pass 