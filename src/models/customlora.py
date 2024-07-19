from peft.tuners.lora.layer import Conv2d, LoraLayer, dispatch_default
from peft.tuners.lora.model import LoraModel



class CustomLoraModel(LoraModel):
    def __init__(self, model, config, adapter_name) -> None:
        super().__init__(model, config, adapter_name)

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        dispatchers = []

        dispatchers.extend(
            [
                dispatch_default,
            ]
        )
        print(dispatchers)

        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(target, adapter_name, lora_config=lora_config, **kwargs)
            if new_module is not None:  # first match wins
                break

        if new_module is None:
            # no module could be matched
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`."
            )

        return new_module


# # Initialize LoraModel
# lora_model = LoraModel(model, config, "default")

# # Prepare input data
# input_text = "Translate English to French: How are you?"
# inputs = tokenizer(input_text, return_tensors="pt")

# # Prepare decoder inputs (start with the special token)
# decoder_start_token_id = tokenizer.pad_token_id  # or use tokenizer.eos_token_id, depending on the model
# decoder_input_ids = torch.tensor([[decoder_start_token_id]])

# # Feed data into lora_model
# outputs = lora_model(input_ids=inputs['input_ids'], decoder_input_ids=decoder_input_ids)
        
