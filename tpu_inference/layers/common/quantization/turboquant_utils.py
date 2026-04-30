# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Lloyd-Max optimal scalar quantizer and WHT utilities for TurboQuant."""

import math
from functools import lru_cache
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


def _gaussian_pdf(x: float, sigma2: float) -> float:
    return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(-x * x /
                                                              (2 * sigma2))


def _trapz(f, a: float, b: float, n: int = 200) -> float:
    """Trapezoidal numerical integration."""
    h = (b - a) / n
    result = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        result += f(a + i * h)
    return result * h


@lru_cache(maxsize=32)
def solve_lloyd_max(
    d: int,
    bits: int,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve Lloyd-Max optimal quantizer for N(0, 1/d) distribution."""
    n_levels = 2**bits
    sigma2 = 1.0 / d
    sigma = math.sqrt(sigma2)

    def pdf(x):
        return _gaussian_pdf(x, sigma2)

    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [
        lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)
    ]

    for _ in range(max_iter):
        boundaries = [(centroids[i] + centroids[i + 1]) / 2.0
                      for i in range(n_levels - 1)]
        edges = [lo * 3] + boundaries + [hi * 3]
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            num = _trapz(lambda x: x * pdf(x), a, b)
            den = _trapz(pdf, a, b)
            new_centroids.append(num / den if den > 1e-15 else centroids[i])

        if max(abs(new_centroids[i] - centroids[i])
               for i in range(n_levels)) < tol:
            break
        centroids = new_centroids

    boundaries = [(centroids[i] + centroids[i + 1]) / 2.0
                  for i in range(n_levels - 1)]
    return (
        np.array(centroids, dtype=np.float32),
        np.array(boundaries, dtype=np.float32),
    )


def get_centroids(d: int, bits: int) -> jnp.ndarray:
    """Get precomputed Lloyd-Max centroids as JAX array."""
    centroids, _ = solve_lloyd_max(d, bits)
    return jnp.array(centroids)


def get_midpoints(d: int, bits: int) -> jnp.ndarray:
    """Get precomputed Lloyd-Max midpoints as JAX array."""
    _, boundaries = solve_lloyd_max(d, bits)
    return jnp.array(boundaries)


def build_hadamard(d: int) -> jnp.ndarray:
    """Orthonormal Hadamard matrix (Sylvester construction)."""
    h = np.array([[1.0]])
    while h.shape[0] < d:
        h = np.vstack((np.hstack((h, h)), np.hstack((h, -h))))
    return jnp.array(h / math.sqrt(d))


def generate_wht_signs(d: int, seed: int) -> jnp.ndarray:
    """Generate deterministic random +-1 signs for WHT rotation."""
    key = jax.random.PRNGKey(seed)
    bits = jax.random.bernoulli(key, p=0.5, shape=(d, ))
    return bits.astype(jnp.float32) * 2.0 - 1.0


def get_turboquant_rotation(d: int,
                            seed: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate Pi and Pi^T for a given layer."""
    h = build_hadamard(d)
    signs = generate_wht_signs(d, seed)
    # Pi = signs * H
    # Pi^T = (signs * H)^T = H^T * signs^T = H * signs (since H is symmetric)
    pi_t = (signs[:, None] * h).T
    pi = pi_t.T
    return pi, pi_t


def dequantize_tq_kv(
    key_packed: jnp.ndarray,
    value_packed: jnp.ndarray,
    tq_config: Any,
    pi: jnp.ndarray | None = None,
    centroids: jnp.ndarray | None = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Pure JAX dequantization for TurboQuant KV cache."""
    # 1. Key Dequantization
    if tq_config.key_fp8:
        # Cast uint8 back to float8 and then to bfloat16
        key_q = jax.lax.bitcast_convert_type(key_packed, jnp.float8_e4m3fn)
        key = key_q.astype(jnp.bfloat16)
    else:
        # MSE keys: Unpack, lookup centroids, de-normalize, rotate back
        # This is a simplified draft
        if tq_config.key_mse_bits == 4:
            # Unpack two 4-bit values from uint8
            idx0 = key_packed & 0x0F
            idx1 = (key_packed >> 4) & 0x0F
            idx = jnp.stack([idx0, idx1],
                            axis=-1).reshape(key_packed.shape[:-1] + (-1, ))
        else:
            idx = key_packed.astype(jnp.int32)  # Placeholder

        # Centroid lookup
        y_recon = centroids[idx]

        # Rotate back: k = y_recon @ pi
        # key_packed: (..., head_dim_packed)
        # y_recon: (..., model_head_dim)
        # pi: (model_head_dim, model_head_dim)
        key = jnp.dot(y_recon, pi)

        # Apply norm (needs to be unpacked from somewhere)
        # For POC, let's assume norm was 1.0 or stored separately
        # (In a real implementation, we'd slice key_packed to get norm bytes)
        pass

    # 2. Value Dequantization (Uniform)
    value = value_packed.astype(jnp.bfloat16)  # Placeholder

    return key, value


def ref_turboquant_attention(
    q: jnp.ndarray,
    k_packed: jnp.ndarray,
    v_packed: jnp.ndarray,
    tq_config: Any,
    pi: jnp.ndarray | None = None,
    centroids: jnp.ndarray | None = None,
    sm_scale: float = 1.0,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Reference SDPA implementation using TurboQuant dequantization."""
    k, v = dequantize_tq_kv(k_packed, v_packed, tq_config, pi, centroids)

    # Standard SDPA
    # q: (T, N, H), k: (S, K, H), v: (S, K, H)
    # Replicate k, v if GQA
    if k.shape[1] < q.shape[1]:
        factor = q.shape[1] // k.shape[1]
        k = jnp.repeat(k, factor, axis=1)
        v = jnp.repeat(v, factor, axis=1)

    attn_weights = jnp.einsum("tnh,skh->nth s", q, k) * sm_scale
    if mask is not None:
        attn_weights = jnp.where(mask, attn_weights, -jnp.inf)
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)
    output = jnp.einsum("nth s,skh->tnh", attn_weights, v)
    return output
