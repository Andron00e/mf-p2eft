# from mmfreelm.models import HGRNBitConfig
# from transformers import AutoModel
from peft import get_peft_model, LoraConfig
import torch

# from atlislib import altislayer
# from transpplib import transmodel
# from mmplib import mmplayer
# from mmplib import mmpmodel
# from utils import apply_lora


def model_factory(base_model, lora):
    if base_model == "transformer++":
        base_model = ...
    else:
        base_model = ...

    if lora == "atli":
        lora_layer = ...
    else:
        lora_layer = ...

    return apply_lora(base_model, lora_layer)
