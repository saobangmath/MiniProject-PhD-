"""Stage 3 search-strategy interface.

Paper score used by search:
    S(M) = log p(M) + log p_M(y) + E[log p(theta | M)]

where log p_M(y) is SMC evidence and log p(theta | M) is the Stage-1
Gaussian prior on (alpha, betas), averaged over posterior particles.

A strategy searches over subsets M and returns the selected model together
with posterior summaries of (alpha, betas) for that model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .stage2_worker import mask_to_set_indexes, prior_logpdf_vec, run_smc_on_data


@dataclass
class SearchResult:
    """Best model found by a search strategy."""

    mask: int
    variables: list[str]
    alpha: float
    betas: np.ndarray
    log_evidence: float
    log_model_prior: float
    log_param_prior: float
    score: float
    included_indices: list[int] = field(default_factory=list)
    particles: np.ndarray | None = None
    history: list[dict] = field(default_factory=list)

    @property
    def n_predictors(self) -> int:
        return len(self.included_indices)

    @property
    def dimension(self) -> int:
        """Intercept plus included slopes."""
        return 1 + self.n_predictors


class ModelPrior(ABC):
    """Prior p(M) over variable subsets encoded as bitmasks."""

    @abstractmethod
    def log_prob(self, mask: int, n_predictors: int) -> float:
        """Return log p(M) for the subset encoded by `mask`."""


class IndependentInclusionPrior(ModelPrior):
    """p(M) = rho^{|M|} (1-rho)^{p-|M|}."""

    def __init__(self, rho: float = 0.5):
        if not (0.0 < rho < 1.0):
            raise ValueError("rho must be in (0, 1)")
        self.rho = float(rho)

    def log_prob(self, mask: int, n_predictors: int) -> float:
        size = mask.bit_count() if hasattr(int, "bit_count") else bin(mask).count("1")
        return size * np.log(self.rho) + (n_predictors - size) * np.log(1.0 - self.rho)


class SizePenaltyPrior(ModelPrior):
    """p(M) ∝ exp(-lambda |M|)."""

    def __init__(self, lam: float = 1.0):
        if lam < 0.0:
            raise ValueError("lam must be >= 0")
        self.lam = float(lam)

    def log_prob(self, mask: int, n_predictors: int) -> float:
        del n_predictors
        size = mask.bit_count() if hasattr(int, "bit_count") else bin(mask).count("1")
        return -self.lam * size


class SearchStrategy(ABC):
    """Interface for Stage 3 model-space search.

    Implementations: greedy, forward-backward, stochastic, ...
    """

    def __init__(
        self,
        model_prior: ModelPrior | None = None,
        n_particles: int = 5_000,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.model_prior = model_prior or IndependentInclusionPrior(rho=0.5)
        self.n_particles = int(n_particles)
        self.seed = int(seed)
        self.verbose = bool(verbose)

    def _names(self, p: int, predictor_names: Sequence[str] | None) -> list[str]:
        if predictor_names is None:
            return [f"x{j}" for j in range(p)]
        names = list(predictor_names)
        if len(names) != p:
            raise ValueError(f"Expected {p} predictor names, got {len(names)}")
        return names

    def _score_model(self, mask: int, X, y, n_predictors: int) -> dict:
        """Score a subset.

        S(M) = log p(M) + log p_M(y) + E[log p(alpha, betas | M)]
        """
        idx = mask_to_set_indexes(mask, n_predictors)
        X_m = X[:, idx] if idx else X[:, :0]
        smc_out = run_smc_on_data(
            X_m,
            y,
            no_samples=self.n_particles,
            seed=self.seed + int(mask),
        )
        particles = np.asarray(smc_out["particles"])
        log_evidence = float(smc_out["log_evidence"])
        log_model_prior = float(self.model_prior.log_prob(mask, n_predictors))
        log_param_prior = float(np.mean(np.asarray(prior_logpdf_vec(particles))))
        return {
            "mask": int(mask),
            "included_indices": idx,
            "log_evidence": log_evidence,
            "log_model_prior": log_model_prior,
            "log_param_prior": log_param_prior,
            "score": log_model_prior + log_evidence + log_param_prior,
            "particles": particles,
            "alpha": float(particles[:, 0].mean()),
            "betas": particles[:, 1:].mean(axis=0) if particles.shape[1] > 1 else np.array([]),
        }

    def _add_remove_neighbours(self, mask: int, n_predictors: int) -> list[tuple[str, int, int]]:
        """Neighbours obtained by adding or removing one variable.

        Returns (action, index, new_mask) with action in {'add', 'remove'}.
        """
        neighbours = []
        for j in range(n_predictors):
            bit = 1 << j
            if mask & bit:
                neighbours.append(("remove", j, mask ^ bit))
            else:
                neighbours.append(("add", j, mask | bit))
        return neighbours

    def _to_result(self, scored: dict, names: Sequence[str], history: list[dict]) -> SearchResult:
        idx = scored["included_indices"]
        return SearchResult(
            mask=scored["mask"],
            variables=[names[j] for j in idx],
            alpha=scored["alpha"],
            betas=np.asarray(scored["betas"]),
            log_evidence=scored["log_evidence"],
            log_model_prior=scored["log_model_prior"],
            log_param_prior=scored["log_param_prior"],
            score=scored["score"],
            included_indices=list(idx),
            particles=scored["particles"],
            history=history,
        )

    @abstractmethod
    def find(
        self,
        X,
        y,
        predictor_names: Sequence[str] | None = None,
    ) -> SearchResult:
        """Search model space and return the best model with (alpha, betas).

        Parameters
        ----------
        X : array (n, p)
            Predictor matrix (already standardized as in Stage 1/2).
        y : array (n,)
            Binary response.
        predictor_names : optional names for the p columns.

        Returns
        -------
        SearchResult
            Selected bitmask/variables, score, and posterior mean
            `alpha` plus `betas` for the included predictors.
        """


class GreedyForwardSearch(SearchStrategy):
    """Start from intercept-only; add one variable at a time.

    At each step consider neighbours M ∪ {j} for j not in M.
    Move to the candidate with the largest score
        S(M) = log p(M) + log p_M(y)
    and stop when no addition improves on the current model.
    """

    def find(self, X, y, predictor_names: Sequence[str] | None = None) -> SearchResult:
        X = np.asarray(X)
        y = np.ravel(np.asarray(y))
        n_predictors = X.shape[1]
        names = self._names(n_predictors, predictor_names)
        cache: dict[int, dict] = {}

        def evaluate(mask: int) -> dict:
            if mask not in cache:
                cache[mask] = self._score_model(mask, X, y, n_predictors)
            return cache[mask]

        current = evaluate(0)  # intercept-only
        history = [
            {
                "step": 0,
                "added": None,
                "mask": 0,
                "variables": [],
                "log_evidence": current["log_evidence"],
                "log_param_prior": current["log_param_prior"],
                "score": current["score"],
            }
        ]
        if self.verbose:
            print(
                f"[greedy] start intercept-only | "
                f"log p_M(y)={current['log_evidence']:.4f} | "
                f"log p(theta)={current['log_param_prior']:.4f} | S={current['score']:.4f}"
            )

        while True:
            unused = [j for j in range(n_predictors) if (current["mask"] & (1 << j)) == 0]
            if not unused:
                break

            best_cand = None
            for j in unused:
                cand = evaluate(current["mask"] | (1 << j))
                if self.verbose:
                    print(
                        f"  try add {names[j]:<20} | "
                        f"log p_M(y)={cand['log_evidence']:.4f} | "
                        f"log p(theta)={cand['log_param_prior']:.4f} | S={cand['score']:.4f}"
                    )
                if best_cand is None or cand["score"] > best_cand["score"]:
                    best_cand = cand

            if best_cand is None or best_cand["score"] <= current["score"]:
                if self.verbose:
                    print("[greedy] stop: no adding move improves S(M)")
                break

            added_idx = int(best_cand["mask"] ^ current["mask"]).bit_length() - 1
            current = best_cand
            history.append(
                {
                    "step": len(history),
                    "added": names[added_idx],
                    "mask": current["mask"],
                    "variables": [names[j] for j in current["included_indices"]],
                    "log_evidence": current["log_evidence"],
                    "log_param_prior": current["log_param_prior"],
                    "score": current["score"],
                }
            )
            if self.verbose:
                print(
                    f"[greedy] accept {names[added_idx]} | "
                    f"vars={history[-1]['variables']} | S={current['score']:.4f}"
                )

        return self._to_result(current, names, history)


class ForwardBackwardSearch(SearchStrategy):
    """Stepwise search over add-one and remove-one neighbours.

    Start from the intercept-only model. At each step consider
        N(M) = {M ∪ {j} : j ∉ M} ∪ {M \\ {j} : j ∈ M}
    and move to the neighbour with the largest score
        S(M) = log p(M) + log p_M(y)
    Stop when no neighbour improves S(M). Already-visited models are skipped
    to avoid cycles from noisy SMC evidence.
    """

    def find(self, X, y, predictor_names: Sequence[str] | None = None) -> SearchResult:
        X = np.asarray(X)
        y = np.ravel(np.asarray(y))
        n_predictors = X.shape[1]
        names = self._names(n_predictors, predictor_names)
        cache: dict[int, dict] = {}

        def evaluate(mask: int) -> dict:
            if mask not in cache:
                cache[mask] = self._score_model(mask, X, y, n_predictors)
            return cache[mask]

        current = evaluate(0)
        visited = {0}
        history = [
            {
                "step": 0,
                "action": None,
                "variable": None,
                "mask": 0,
                "variables": [],
                "log_evidence": current["log_evidence"],
                "log_param_prior": current["log_param_prior"],
                "score": current["score"],
            }
        ]
        if self.verbose:
            print(
                f"[fwd-bwd] start intercept-only | "
                f"log p_M(y)={current['log_evidence']:.4f} | "
                f"log p(theta)={current['log_param_prior']:.4f} | S={current['score']:.4f}"
            )

        while True:
            best_move = None  # (action, idx, scored)
            for action, j, new_mask in self._add_remove_neighbours(current["mask"], n_predictors):
                if new_mask in visited:
                    continue
                cand = evaluate(new_mask)
                if self.verbose:
                    print(
                        f"  try {action:6} {names[j]:<20} | "
                        f"log p_M(y)={cand['log_evidence']:.4f} | "
                        f"log p(theta)={cand['log_param_prior']:.4f} | S={cand['score']:.4f}"
                    )
                if cand["score"] <= current["score"]:
                    continue
                if best_move is None or cand["score"] > best_move[2]["score"]:
                    best_move = (action, j, cand)

            if best_move is None:
                if self.verbose:
                    print("[fwd-bwd] stop: no add/remove neighbour improves S(M)")
                break

            action, j, cand = best_move
            current = cand
            visited.add(current["mask"])
            history.append(
                {
                    "step": len(history),
                    "action": action,
                    "variable": names[j],
                    "mask": current["mask"],
                    "variables": [names[k] for k in current["included_indices"]],
                    "log_evidence": current["log_evidence"],
                    "log_param_prior": current["log_param_prior"],
                    "score": current["score"],
                }
            )
            if self.verbose:
                print(
                    f"[fwd-bwd] {action} {names[j]} | "
                    f"vars={history[-1]['variables']} | S={current['score']:.4f}"
                )

        return self._to_result(current, names, history)


class RestrictedEnumerationSearch(SearchStrategy):
    """Enumerate all 2^{|A|} subsets of a restricted predictor set A.

    Used after Lasso screening: A = top-k Lasso variables, then pick
    argmax_M S(M) over Msubseteq A (including intercept-only).
    """

    def __init__(self, allowed_indices: Sequence[int], **kwargs):
        super().__init__(**kwargs)
        if len(allowed_indices) == 0:
            raise ValueError("allowed_indices must be non-empty")
        self.allowed_indices = [int(j) for j in allowed_indices]

    def find(self, X, y, predictor_names: Sequence[str] | None = None) -> SearchResult:
        X = np.asarray(X)
        y = np.ravel(np.asarray(y))
        n_predictors = X.shape[1]
        names = self._names(n_predictors, predictor_names)
        for j in self.allowed_indices:
            if not (0 <= j < n_predictors):
                raise ValueError(f"allowed index {j} is outside 0..{n_predictors-1}")

        k = len(self.allowed_indices)
        n_models = 1 << k
        if self.verbose:
            allowed_names = [names[j] for j in self.allowed_indices]
            print(
                f"[enum] enumerating {n_models} subsets of {allowed_names}"
            )

        best = None
        history = []
        for sub in range(n_models):
            mask = 0
            for b, j in enumerate(self.allowed_indices):
                if sub & (1 << b):
                    mask |= 1 << j
            scored = self._score_model(mask, X, y, n_predictors)
            row = {
                "submask": sub,
                "mask": scored["mask"],
                "variables": [names[j] for j in scored["included_indices"]],
                "n_predictors": len(scored["included_indices"]),
                "log_evidence": scored["log_evidence"],
                "log_model_prior": scored["log_model_prior"],
                "log_param_prior": scored["log_param_prior"],
                "score": scored["score"],
            }
            history.append(row)
            if self.verbose:
                vars_txt = ", ".join(row["variables"]) if row["variables"] else "(intercept only)"
                print(
                    f"  M={vars_txt:<40} | "
                    f"log p_M(y)={row['log_evidence']:.4f} | "
                    f"log p(theta)={row['log_param_prior']:.4f} | S={row['score']:.4f}"
                )
            if best is None or scored["score"] > best["score"]:
                best = scored

        history_sorted = sorted(history, key=lambda r: r["score"], reverse=True)
        if self.verbose:
            winner_vars = [names[j] for j in best["included_indices"]]
            print(f"[enum] best S(M)={best['score']:.4f} | vars={winner_vars}")

        result = self._to_result(best, names, history_sorted)
        return result


class StochasticSearch(SearchStrategy):
    """Metropolis-Hastings on model space targeting p(M) p_M(y)."""

    def find(self, X, y, predictor_names: Sequence[str] | None = None) -> SearchResult:
        raise NotImplementedError("StochasticSearch.find is not implemented yet")
