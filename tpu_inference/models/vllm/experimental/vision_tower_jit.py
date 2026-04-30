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

# Utilities to support JIT compilation of VisionTower.

import math
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torchax.ops.jtorch import register_function
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_5 import \
    Qwen3_5MoeForConditionalGeneration

from tpu_inference.kernels.flash_attention.kernel import MIN_BLOCK_SIZE
from tpu_inference.layers.common.attention_interface import \
    sharded_flash_attention
from tpu_inference.logger import init_logger
from tpu_inference.utils import to_jax_dtype

logger = init_logger(__name__)

# Per-image patch lengths set before jax.jit by embed_multimodal_func_torch.
# cu_seqlens is a traced JAX array inside jit and cannot be concretized
# directly, so concrete lengths are communicated via this module-level variable.
_vit_image_patch_lens: Optional[list[int]] = None
_jittable_vit_sdpa_registered: bool = False


def set_vit_image_patch_lens(lens: Optional[list[int]]) -> None:
    """Set per-image patch lengths before calling jax.jit for the vision encoder.

    Raises RuntimeError if the jittable ViT SDPA has not been registered
    (i.e., the model architecture is not in JITTABLE_ARCHS).
    """
    if not _jittable_vit_sdpa_registered:
        raise RuntimeError(
            "set_vit_image_patch_lens: jittable ViT SDPA is not registered. "
            "Call maybe_register_jittable_vit_sdpa first.")
    global _vit_image_patch_lens
    _vit_image_patch_lens = lens


def register_function_if(op, condition: bool):
    """Like @register_function(op) but only registers if condition is True."""

    def decorator(fn: Callable) -> Callable:
        if condition:
            register_function(op)(fn)
        return fn

    return decorator


# Architectures whose embed_multimodal function is safe to wrap with jax.jit.
JITTABLE_ARCHS = {
    Qwen3_5MoeForConditionalGeneration,
}


def is_jittable_architecture(vllm_model) -> bool:
    """Check if the given vLLM model is of an architecture that supports JIT compilation."""
    is_jittable = any(isinstance(vllm_model, arch) for arch in JITTABLE_ARCHS)
    if is_jittable:
        logger.info_once(
            f"{type(vllm_model)}'s vision tower supports JIT compilation.")
    else:
        logger.warning_once(
            f"{type(vllm_model)}'s vision tower does NOT support JIT compilation."
        )
    return is_jittable


def maybe_jit_embed_multimodal_func(embed_multimodal_func_jax: Callable,
                                    vllm_model) -> Callable:
    """Conditionally wrap `embed_multimodal_func_jax` with jax.jit based on the VllmConfig.

    Args:
        embed_multimodal_func_jax: The JAX function to be potentially JIT-compiled.
        vllm_model: The Vllm model instance containing the configuration.
    """
    if is_jittable_architecture(vllm_model):
        return jax.jit(static_argnames=("image_grid_thw", "video_grid_thw",
                                        "grid_thw"))(embed_multimodal_func_jax)
    else:
        return embed_multimodal_func_jax


class GridTHW(tuple):
    """Tensor-like wrapper for image/video grid_thw arguments.

    - tuple subclass so isinstance(x, tuple) is True — passes vLLM's
    tensor_schema type check (e.g. https://github.com/vllm-project/vllm/blob/9744b699bafed423909ed10da96b80eb0542424b/vllm/model_executor/models/qwen3_vl.py#L2026). 
    - Implements a minimal tensor-like API (ndim, shape, tolist, prod) expected by vLLM's
    _process_image_input (https://github.com/vllm-project/vllm/blob/9744b699bafed423909ed10da96b80eb0542424b/vllm/model_executor/models/qwen3_vl.py#L2072)

    We cannot use torch.Tensor[tuple] because jax.jit would complain.
    """

    def __new__(cls, values):

        def _nested_to_tuple(v):
            if isinstance(v, (list, tuple)):
                return tuple(_nested_to_tuple(x) for x in v)
            return int(v)

        flat: tuple = _nested_to_tuple(values)
        return super().__new__(cls, flat)

    # ---- tensor-like API expected by _process_image_input ----

    @property
    def ndim(self):
        return 2

    @property
    def shape(self):
        return (len(self), 3)

    def tolist(self):
        return [list(row) for row in self]

    def prod(self, dim=-1):
        if dim in (-1, 1):
            return np.array([row[0] * row[1] * row[2] for row in self])
        raise NotImplementedError(f"GridTHW.prod({dim}) not supported")

    def __repr__(self):
        return f"GridTHW({tuple(self)})"


def maybe_precompile_vision_encoder_fn(
        params: Any, embed_multimodal_fn: Optional[Callable], vllm_model,
        vllm_config: VllmConfig) -> Optional[Callable]:
    """Return a precompile function for jittable vision encoders, or None.

    The returned function accepts a single argument (run_compilation_fn) and
    calls embed_multimodal_fn with dummy pixel_value tensors of various sizes
    so that JAX/XLA compilation is done upfront rather than at first inference.
    Only architectures listed in JITTABLE_ARCHS are supported.
    """
    if embed_multimodal_fn is None:
        return None

    if not is_jittable_architecture(vllm_model):
        return None

    # patch_input_dim is the flattened input feature dimension per raw patch:
    #   in_channels * temporal_patch_size * patch_size * patch_size
    # e.g. for Qwen3.5: 3 * 2 * 16 * 16 = 1536
    # Ref: https://github.com/vllm-project/vllm/blob/eb6661d52/vllm/model_executor/models/qwen3_vl.py#L1941
    vc = vllm_config.model_config.hf_config.vision_config
    patch_input_dim = (vc.in_channels * vc.temporal_patch_size *
                       vc.patch_size * vc.patch_size)
    spatial_merge_unit = vc.spatial_merge_size**2
    max_patches = (vllm_config.scheduler_config.max_num_batched_tokens //
                   spatial_merge_unit)
    min_shift = 4  # 1 << 4 = 16 patches minimum
    max_shift = max(min_shift, (max(max_patches, 1) - 1).bit_length())
    num_patches_paddings = [1 << i for i in range(min_shift, max_shift + 1)]

    jax_dtype = to_jax_dtype(vllm_config.model_config.dtype)

    def precompile_fn(run_compilation_fn: Callable) -> None:
        for num_patches in num_patches_paddings:
            # Split num_patches into (h, w) by distributing bits evenly.
            # For any power-of-2 num_patches = 2^k: h=2^(k//2), w=2^(k-k//2).
            k = int(round(math.log2(num_patches)))
            h = 1 << (k // 2)
            w = 1 << (k - k // 2)

            dummy_pixel_values = jnp.ones((num_patches, patch_input_dim),
                                          dtype=jax_dtype)
            dummy_image_grid_thw = GridTHW([(1, h, w)])

            run_compilation_fn(
                f"vllm embed_multimodal {dummy_image_grid_thw}",
                embed_multimodal_fn,
                params,
                call_kwargs={
                    "pixel_values": dummy_pixel_values,
                    "image_grid_thw": dummy_image_grid_thw,
                },
                num_patches=num_patches,
            )

    return precompile_fn


def maybe_prepare_for_jit(kwargs: dict, vllm_model) -> dict:
    """Convert certain kwargs to JIT-friendly formats, if needed.

    Specifically, convert "image_grid_thw", "video_grid_thw", and "grid_thw" to
    GridTHW instances, which are tuple subclasses that can be hashed in jax.jit.
    """
    if not is_jittable_architecture(vllm_model):
        return kwargs

    for k, v in kwargs.items():
        if k in ("image_grid_thw", "video_grid_thw", "grid_thw"):
            kwargs[k] = GridTHW(v.tolist())
    return kwargs


def maybe_register_jittable_vit_sdpa(vllm_model) -> None:
    """Override torch_sdpa_wrapper with a per-image-chunking implementation
    for jittable architectures.

    For jittable archs, the standard segment-ID implementation in
    scaled_dot_product_attention.py would OOM on VMEM because flash attention
    sees the full concatenated patch sequence.  The per-image chunking
    implementation registered here bounds VMEM to the largest single image.

    Must be called once after the model is loaded and before any inference.
    """
    global _jittable_vit_sdpa_registered

    if not is_jittable_architecture(vllm_model):
        return

    @register_function_if(torch.ops.vllm.torch_sdpa_wrapper, True)
    def _vllm_vit_sdpa_jittable(
        query,
        key,
        value,
        scale=None,
        cu_seqlens=None,
        enable_gqa=False,
    ):
        """Per-image chunking ViT SDPA for jittable architectures.

        cu_seqlens is a traced JAX array inside jit and cannot be concretized,
        so per-image patch lengths are read from _vit_image_patch_lens instead,
        which is set before jax.jit by embed_multimodal_func_torch.
        """
        # (batch, seq_len, num_heads, head_dim) → (batch, num_heads, seq_len, head_dim)
        query = jnp.swapaxes(query, 1, 2)
        key = jnp.swapaxes(key, 1, 2)
        value = jnp.swapaxes(value, 1, 2)

        mesh = jax.sharding.get_abstract_mesh()
        attn_fn = sharded_flash_attention(mesh,
                                          causal=False,
                                          sm_scale=scale,
                                          use_attention_bias=False)

        if cu_seqlens is not None and _vit_image_patch_lens is not None:
            outputs = []
            start = 0
            for length in _vit_image_patch_lens:
                q_chunk = query[:, :, start:start + length, :]
                k_chunk = key[:, :, start:start + length, :]
                v_chunk = value[:, :, start:start + length, :]

                pad = (MIN_BLOCK_SIZE -
                       (length % MIN_BLOCK_SIZE)) % MIN_BLOCK_SIZE
                if pad > 0:
                    q_chunk = jnp.pad(q_chunk,
                                      ((0, 0), (0, 0), (0, pad), (0, 0)))
                    k_chunk = jnp.pad(k_chunk,
                                      ((0, 0), (0, 0), (0, pad), (0, 0)))
                    v_chunk = jnp.pad(v_chunk,
                                      ((0, 0), (0, 0), (0, pad), (0, 0)))

                out_chunk = attn_fn(q_chunk, k_chunk, v_chunk, None)

                if pad > 0:
                    out_chunk = out_chunk[:, :, :length, :]
                outputs.append(out_chunk)
                start += length

            out = jnp.concatenate(outputs, axis=2)
        else:
            q_seq_len = query.shape[2]
            kv_seq_len = key.shape[2]

            q_pad = (MIN_BLOCK_SIZE -
                     (q_seq_len % MIN_BLOCK_SIZE)) % MIN_BLOCK_SIZE
            kv_pad = (MIN_BLOCK_SIZE -
                      (kv_seq_len % MIN_BLOCK_SIZE)) % MIN_BLOCK_SIZE

            if q_pad > 0:
                query = jnp.pad(query, ((0, 0), (0, 0), (0, q_pad), (0, 0)))
            if kv_pad > 0:
                key = jnp.pad(key, ((0, 0), (0, 0), (0, kv_pad), (0, 0)))
                value = jnp.pad(value, ((0, 0), (0, 0), (0, kv_pad), (0, 0)))

            out = attn_fn(query, key, value, None)

            if q_pad > 0:
                out = out[:, :, :q_seq_len, :]

        # (batch, num_heads, seq_len, head_dim) → (batch, seq_len, num_heads, head_dim)
        return jnp.swapaxes(out, 1, 2)

    _jittable_vit_sdpa_registered = True
    logger.info("Registered jittable per-image-chunking ViT SDPA for %s.",
                type(vllm_model).__name__)
