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

import torch
from vllm.model_executor.layers.layernorm import RMSNorm


@RMSNorm.register_oot
class VllmRMSNorm(RMSNorm):

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        weight_param = None
        if getattr(self, "has_weight", True) and hasattr(self, "weight"):
            w = self.weight
            if isinstance(w, torch.nn.Parameter):
                w = w.data
            if type(w) is torch.Tensor:
                w = w.to(device="jax")
            weight_param = w

        return self.forward_static(
            x,
            self.variance_epsilon,
            self.hidden_size,
            x.dtype,
            weight_param,
            residual,
            self.variance_size_override,
        )
