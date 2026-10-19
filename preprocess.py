
from load_data import get_dataset
from load_model import get_model_and_tokenizer
from config import MAX_INPUT_LENGTH, MAX_TARGET_LENGTH


dataset = get_dataset()
model, tokenizer = get_model_and_tokenizer()


def tokenize_function(example):
    prompts = [
        "Summarize the following conversation.\n\n"
        + dialogue
        + "\n\nSummary:"
        for dialogue in example["dialogue"]
    ]

    model_inputs = tokenizer(
        prompts,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        text_target=example["summary"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length"
    )

    # Ignore padding tokens when calculating the loss
    labels["input_ids"] = [
        [
            token if token != tokenizer.pad_token_id else -100
            for token in label
        ]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

print(tokenized_datasets)
print("\nTokenized sample:")
print(tokenized_datasets["train"][0].keys())
