from typing import Any, Optional, Union
from peft.tuners.lora.layer import (
    Conv2d, LoraLayer, dispatch_default, Linear as LoraLinear
)
from peft.tuners.lora.model import LoraModel
from peft.tuners.lora.config import LoraConfig
import torch.nn as nn
from submodules.minmaxplus.abx import TopK
from submodules.pam_ops import Linear as PAMLinear
from functools import partial
import torch

topg = partial(TopK, k=3)


class CustomLinear(LoraLinear):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        special_layer = nn.Linear,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False, 
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        self.special_layer = special_layer
        # if hasattr(special_layer(10,10), 'bias'):
        #     self.special_layer = partial(special_layer, bias=False)
        
        
        super().__init__(  # basically copying the CustLay args
            base_layer, adapter_name, r, lora_alpha, lora_dropout,
            fan_in_fan_out, is_target_conv_1d_layer, init_lora_weights,
            use_rslora, use_dora, **kwargs,
        )

    # LoraLayer overload, all other methods inherited from Linear are fine
    def update_layer(  # I just replaced nn.Linear with self.special_layer
        self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora,
        use_dora: bool = False
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        self.lora_A[adapter_name] = self.special_layer(self.in_features, r)
        self.lora_B[adapter_name] = self.special_layer(r, self.out_features)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            self.pissa_init(adapter_name, init_lora_weights)
        elif init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        # check weight and qweight (for GPTQ)
        for weight_name in ("weight", "qweight"):
            weight = getattr(self.get_base_layer(), weight_name, None)
            if weight is not None:
                # the layer is already completely initialized, this is an update
                if weight.dtype.is_floating_point or weight.dtype.is_complex:
                    self.to(weight.device, dtype=weight.dtype)
                else:
                    self.to(weight.device)
                break

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)


def dispatch_topg(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: LoraConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    kwargs.update(lora_config.loftq_config)
    new_module = CustomLinear(target, adapter_name, special_layer=topg, **kwargs)
    return new_module


def dispatch_pam(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: LoraConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    kwargs.update(lora_config.loftq_config)
    new_module = CustomLinear(target, adapter_name, special_layer=PAMLinear, **kwargs)
    return new_module


class CustomLoraModel(LoraModel):
    def __init__(self, model, config, adapter_name) -> None:
        super().__init__(model, config, adapter_name)

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        dispatcher = {
            "defaut": dispatch_default,
            "pam": dispatch_pam,
            "topg": dispatch_topg,
        }[adapter_name]

        new_module = dispatcher(target, adapter_name, lora_config=lora_config, **kwargs)

        return new_module


# =======================
# the testing goes below
# =======================
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraModel, LoraConfig
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
tokenizer = AutoTokenizer.from_pretrained("t5-small")
    
input_text = "Translate English to French: How are you?"
inputs = tokenizer(input_text, return_tensors="pt")

decoder_start_token_id = tokenizer.pad_token_id  
decoder_input_ids = torch.tensor([[decoder_start_token_id]])

config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.01,
)

lora_model = CustomLoraModel(model, config, "topg")
outputs = lora_model(
    input_ids=inputs['input_ids'],
    decoder_input_ids=decoder_input_ids
)
print(outputs.logits)
