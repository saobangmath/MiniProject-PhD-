"""Your NICE implementation — build from scratch.

Reference (compare later): libs/nice.py
"""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn


class CouplingNet(nn.Module):
    """MLP for m(·): ReLU hidden layers, linear output. Maps odd→even or even→odd."""

    hidden_sizes: Sequence[int]
    output_size: int

    @nn.compact
    def __call__(self, x):
        for width in self.hidden_sizes:
            x = nn.relu(nn.Dense(width)(x))
        return nn.Dense(self.output_size)(x)  # linear last layer


class CouplingLayer(nn.Module):
    """Additive coupling: updated_half += m(other_half), m = MLP."""

    layer_size: int
    transform_even: bool  # True → change even; False → change odd
    hidden_sizes: Sequence[int] = (64, 64)

    @nn.compact
    def __call__(self, x, reverse: bool = False):
        length = x.shape[-1]
        odd = x[..., 0::2]
        even = x[..., 1::2]

        if self.transform_even:
            # m: odd_size → even_size
            shift = CouplingNet(self.hidden_sizes, even.shape[-1], name="m")(odd)
            even = even - shift if reverse else even + shift
        else:
            # m: even_size → odd_size
            shift = CouplingNet(self.hidden_sizes, odd.shape[-1], name="m")(even)
            odd = odd - shift if reverse else odd + shift

        out = jnp.zeros(x.shape[:-1] + (length,), dtype=x.dtype)
        out = out.at[..., 0::2].set(odd)
        out = out.at[..., 1::2].set(even)
        return out


class ScalingLayer(nn.Module):
    """Diagonal scaling: y_i = s_i * x_i. Length of ``scaling_factors`` should be D."""

    scaling_factors: Sequence[float]

    @nn.compact
    def __call__(self, x, reverse: bool = False):
        s = jnp.asarray(self.scaling_factors)
        if reverse:
            return x / s
        return x * s


class NICE(nn.Module):
    layer_size: int  # data / latent dim D (bijection: in == out)
    no_layers: int = 4  # coupling layers; try 4 or 5 when testing
    hidden_sizes: Sequence[int] = (64, 64)  # widths inside each coupling MLP m

    def setup(self):
        # alternate which half is updated: even, odd, even, odd, ...
        self.couplings = [
            CouplingLayer(
                layer_size=self.layer_size,
                transform_even=(i % 2 == 0),
                hidden_sizes=self.hidden_sizes,
            )
            for i in range(self.no_layers)
        ]
        # identity diagonal for now (fixed, not learned)
        self.scaling = ScalingLayer(
            scaling_factors=[1.0] * self.layer_size
        )
