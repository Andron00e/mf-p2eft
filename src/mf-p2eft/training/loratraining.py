from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
from transformers import T5Tokenizer

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.customlora import config, CustomLoraModel


model = AutoModelForSeq2SeqLM.from_pretrained("t5-small", num_labels=2)
lora_model = CustomLoraModel(model, config, "pam")

tokenizer = T5Tokenizer.from_pretrained("t5-small")  # or other variants like "t5-base"

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

def tokenize_function(examples):
    # Tokenize the input text and use the same text as target
    model_inputs = tokenizer(examples['text'], padding="max_length", truncation=True)
    
    # Use the input text as the target for language modeling
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['text'], padding="max_length", truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

input_text = "translate English to French: The house is wonderful."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
outputs = lora_model.generate(input_ids)
print(input_ids, outputs)

# ____________

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    remove_unused_columns=False,
    num_train_epochs=3,
    weight_decay=0.01,
    # else "RuntimeError: Some tensors share memory,..." arises
    save_safetensors=False,  
)

# Create the Trainer
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets["validation"],
)

# Start training
trainer.train()

trainer.save_model("path_to_save_model")

results = trainer.evaluate()
print(results)
