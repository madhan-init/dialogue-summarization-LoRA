
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


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

dataset = load_dataset(DATASET_NAME)
print(dataset)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

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

train_data = tokenized_datasets["train"].select(
    range(min(500, len(tokenized_datasets["train"])))
)
eval_data = tokenized_datasets["validation"].select(
    range(min(100, len(tokenized_datasets["validation"])))
)

print("Training examples:", len(train_data))
print("Validation examples:", len(eval_data))

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q", "v"],
)

peft_model = get_peft_model(base_model, lora_config)
peft_model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="./peft-dialogue-summary-training",
    learning_rate=1e-3,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    fp16=torch.cuda.is_available(),
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=20,
    save_strategy="steps",
    save_steps=20,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_dir="./runs/peft-dialogue-summary",
    report_to="tensorboard",
    dataloader_pin_memory=torch.cuda.is_available(),
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
)

trainer.train()

output_path = "./peft-dialogue-summary-checkpoint-local"

peft_model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print("Training complete!")
print("Saved model adapter to:", output_path)
