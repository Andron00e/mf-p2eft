import math
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.lora.layer import LoraLayer


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
        self._active_adapter = adapter_name

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora):
        pass
