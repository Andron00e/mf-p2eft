import torch
from torch import nn

from typing import Optional, List


class LoRAConfig():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

    dict = lambda self: self.__dict__

class CustomLora(LoRAConfig, nn.Module):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        LoRALayer,
        MainLayer,
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        pretrained: bool = False,
        **kwargs
    ):
        nn.Module.__init__(self)
        LoRAConfig.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        # can be executed either way; kinda bad to have different types for an arg, but for now it's fine
        # pretrained_layer = CustomLora(MainLayer=pretrained_layer, ...)
        # layer = CustomLora(MainLayer=nn.Liner, ...)
        self.main_layer = Main_Layer if pretrained else MainLayer(in_features, out_features, **kwargs) 

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = LoRALayer(in_features, r)
            self.lora_B = LoRALayer(r, out_features)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.main_layer.requires_grad = False
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        self.main_layer.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    # I hope this works 
                    self.weight.data -= T(self.lora_B.weight @ self.lora_A.weight) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    # I hope this works 
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        result = self.main_layer(x)
        result += self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return result



