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
"""TurboQuant configuration for vLLM."""

import math
from dataclasses import dataclass

from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig

TQ_PRESETS = {
    "turboquant_k8v4": {
        "key_quant_bits": 8,
        "value_quant_bits": 4,
        "norm_correction": False,
    },
    "turboquant_4bit_nc": {
        "key_quant_bits": 4,
        "value_quant_bits": 4,
        "norm_correction": True,
    },
    "turboquant_k3v4_nc": {
        "key_quant_bits": 3,
        "value_quant_bits": 4,
        "norm_correction": True,
    },
    "turboquant_3bit_nc": {
        "key_quant_bits": 3,
        "value_quant_bits": 3,
        "norm_correction": True,
    },
}


@dataclass
class VllmTurboQuantConfig(VllmQuantConfig):
    """Configuration for TurboQuant KV-cache quantization."""
    head_dim: int = 128
    key_quant_bits: int = 3
    value_quant_bits: int = 4
    seed: int = 42
    norm_correction: bool = False

    @property
    def key_fp8(self) -> bool:
        return self.key_quant_bits == 8

    @property
    def mse_bits(self) -> int:
        return self.key_quant_bits if not self.key_fp8 else self.value_quant_bits

    @property
    def key_mse_bits(self) -> int:
        return self.key_quant_bits if not self.key_fp8 else 0

    @property
    def n_centroids(self) -> int:
        return 2**self.mse_bits

    @property
    def key_packed_size(self) -> int:
        if self.key_fp8:
            return self.head_dim
        mse_bytes = math.ceil(self.head_dim * self.key_mse_bits / 8)
        norm_bytes = 2  # vec_norm fp16
        return mse_bytes + norm_bytes

    @property
    def value_packed_size(self) -> int:
        data_bytes = math.ceil(self.head_dim * self.value_quant_bits / 8)
        return data_bytes + 4  # +2 scale(fp16) +2 zero(fp16)

    @property
    def slot_size(self) -> int:
        return self.key_packed_size + self.value_packed_size

    @property
    def slot_size_aligned(self) -> int:
        s = self.slot_size
        return s + (s % 2)

    @staticmethod
    def get_name() -> str:
        return "turboquant"

    @classmethod
    def from_cache_dtype(cls, cache_dtype: str,
                         head_dim: int) -> "VllmTurboQuantConfig":
        if cache_dtype not in TQ_PRESETS:
            valid = ", ".join(TQ_PRESETS.keys())
            raise ValueError(f"Unknown TurboQuant cache dtype: {cache_dtype}. "
                             f"Valid presets: {valid}")
        preset = TQ_PRESETS[cache_dtype]
        return cls(
            head_dim=head_dim,
            key_quant_bits=preset["key_quant_bits"],
            value_quant_bits=preset["value_quant_bits"],
            norm_correction=preset["norm_correction"],
        )
