from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq

# Load a dataset (let's assume you're using a dataset with longer texts, like the CNN/DailyMail dataset)
dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:5%]")

# Load the tokenizer for T5
tokenizer = AutoTokenizer.from_pretrained("t5-small")

# Tokenize the dataset
def tokenize_function(examples):
    # Prefix the input text with a task-specific prefix
    inputs = ["summarize: " + doc for doc in examples["article"]]
    
    # Tokenize input text
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    
    # Tokenize the summaries (labels)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["highlights"], padding="max_length", truncation=True, max_length=150)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small", num_labels=2)
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
 
from models.customlora import config, CustomLoraModel
lora_model = CustomLoraModel(model, config, "pam")
# print(lora_model)
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Use a data collator that dynamically pads inputs and labels during batch processing
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Initialize the Trainer
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,  # In practice, use a separate validation set
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)

# Save the model
model.save_pretrained("./my_model")
tokenizer.save_pretrained("./my_model")
