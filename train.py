
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

from config import (
    MODEL_NAME,
    DATASET_NAME,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
)


# 1. Select device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# 2. Load dataset
dataset = load_dataset(DATASET_NAME)
print(dataset)

# 3. Load pretrained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# 4. Tokenize dialogues and summaries
def tokenize_function(example):
    prompts = [
        "Summarize the following conversation:\n\n"
        + dialogue
        + "\n\nSummary:"
        for dialogue in example["dialogue"]
    ]

    inputs = tokenizer(
        prompts,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length",
    )

    labels = tokenizer(
        text_target=example["summary"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length",
    )

    inputs["labels"] = [
        [
            token if token != tokenizer.pad_token_id else -100
            for token in label
        ]
        for label in labels["input_ids"]
    ]

    return inputs


tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# 5. Start with a small subset to test the training pipeline
train_data = tokenized_datasets["train"].select(
    range(min(500, len(tokenized_datasets["train"])))
)

print("Training examples:", len(train_data))

# 6. Apply LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q", "v"],
)

peft_model = get_peft_model(base_model, lora_config)
peft_model.print_trainable_parameters()

# 7. Configure training
training_args = TrainingArguments(
    output_dir="./peft-dialogue-summary-training",
    learning_rate=1e-3,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    fp16=torch.cuda.is_available(),
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none",
    dataloader_pin_memory=torch.cuda.is_available(),
)

# 8. Create trainer
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_data,
)

# 9. Train
trainer.train()

# 10. Save the LoRA adapter and tokenizer
output_path = "./peft-dialogue-summary-checkpoint-local"

peft_model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print("Training complete!")
print("Saved model adapter to:", output_path)
