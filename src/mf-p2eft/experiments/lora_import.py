from functools import partial
from transformers import AutoModelForSeq2SeqLM
from peft import LoraModel, LoraConfig
from models.minmaxplus.abx import TopK
from models.customlora import CustomLoraModel

config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.01,
)

model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")

topg = partial(TopK, k=5)

lora_model = CustomLoraModel(model, config, "topg")
