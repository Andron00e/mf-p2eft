import math

import torch
import torch.nn as nn
from typing import Any, Optional
from mfp2eft.models.pam import pam_ops
from peft.tuners.lora.layer import LoraLayer
from peft.tuners.lora.config import LoraConfig
from peft.tuners.lora.model import LoraModel
from peft.tuners.tuners_utils import BaseTunerLayer


class PAMLoraLinear(nn.Module, LoraLayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: bool = True,
        use_rslora: bool = False,
        **kwargs,
    ):
        super().__init__()
        LoraLayer.__init__(self, base_layer)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora)

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()
        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        self.lora_A[adapter_name] = pam_ops.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = pam_ops.Linear(r, self.out_features, bias=False)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        result = self.base_layer(x)

        if self.disable_adapters:
            return result

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            output = lora_B(lora_A(dropout(x)))
            output = output * scaling
            result += output
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


def dispatch_pam(
    target: torch.nn.Module,
    adapter_name: str,
    **kwargs: Any,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    # TODO: when we contribute to PEFT with pam_is_available() in src/peft/import_utils.py
    # if pam_is_available():
    #     new_module = PAMLoraLinear(target, adapter_name, **kwargs)
    #     return new_module
    new_module = PAMLoraLinear(target, adapter_name, **kwargs)
    return new_module


class PAMLoraModel(LoraModel):
    def __init__(self, model: torch.nn.Module, config: LoraConfig, adapter_name: str):
        super().__init__(model, config, adapter_name)

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        pass
