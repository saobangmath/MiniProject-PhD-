from tqdm import trange
import jax
import jax.numpy as jnp
from jax import lax, random


class AcceptanceTracker(object):
    def __init__(self):
        self.__accept_count = 0
        self.__total_count = 0

    def record(self, accepted: bool):
        if accepted:
            self.__accept_count += 1
        self.__total_count += 1

    def reset(self):
        self.__accept_count = 0
        self.__total_count = 0

    def get_acceptance_rate(self):
        return self.__accept_count / self.__total_count


class MCMC:
    r"""
    JAX MCMC.
    proposed_fn must be: (state, key) -> proposed_state
    """

    def __init__(self, log_pdf_fn, proposed_fn, tracker=None):
        self.__log_pdf_fn = log_pdf_fn
        self.__proposed_fn = proposed_fn
        self.__tracker = tracker

    def run(self, initial_state, key, iters=10_000, only_keep_last=True, track_process=False):
        r"""
        Run a single MCMC chain.
        Returns array of shape (iters + 1, ...) if only_keep_last=False,
        else shape matching initial_state (last state only).
        """
        # Python loop only when tracking acceptance; otherwise use lax.scan (JIT-friendly).
        if track_process or self.__tracker is not None:
            state = initial_state
            cur_logpdf = self.__log_pdf_fn(state)
            states = [state]

            for _ in (trange(iters) if track_process else range(iters)):
                key, k_prop, k_u = random.split(key, 3)
                proposed = self.__proposed_fn(state, k_prop)
                log_alpha = jnp.minimum(self.__log_pdf_fn(proposed) - cur_logpdf, 0.0)
                accept = log_alpha > jnp.log(random.uniform(k_u))
                if self.__tracker:
                    self.__tracker.record(bool(accept))
                state = jnp.where(accept, proposed, state)
                cur_logpdf = self.__log_pdf_fn(state)
                if only_keep_last:
                    states[-1] = state
                else:
                    states.append(state)

            return states[-1] if only_keep_last else jnp.stack(states)

        def step(carry, _):
            state, cur_logpdf, key = carry
            key, k_prop, k_u = random.split(key, 3)
            proposed = self.__proposed_fn(state, k_prop)
            log_alpha = jnp.minimum(self.__log_pdf_fn(proposed) - cur_logpdf, 0.0)
            accept = log_alpha > jnp.log(random.uniform(k_u))
            next_state = jnp.where(accept, proposed, state)
            next_logpdf = jnp.where(accept, self.__log_pdf_fn(proposed), cur_logpdf)
            return (next_state, next_logpdf, key), next_state

        init_logpdf = self.__log_pdf_fn(initial_state)
        (last_state, _, _), all_states = lax.scan(
            step, (initial_state, init_logpdf, key), None, length=iters
        )
        if only_keep_last:
            return last_state
        return jnp.concatenate([initial_state[None, ...], all_states], axis=0)

    def run_batch(self, initial_states, key, iters=10):
        r"""
        Run many independent chains in parallel with vmap.
        initial_states: (n_chains, dims)
        returns: (n_chains, dims) last states
        """
        keys = random.split(key, initial_states.shape[0])
        return jax.vmap(
            lambda s, k: self.run(s, k, iters=iters, only_keep_last=True, track_process=False)
        )(initial_states, keys) 

if __name__ == "__main__":
    pass 