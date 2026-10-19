
from load_data import get_dataset
from load_model import get_model_and_tokenizer


def print_number_of_trainable_model_parameters(model):
    trainable_params = 0
    all_params = 0

    for _, param in model.named_parameters():
        all_params += param.numel()

        if param.requires_grad:
            trainable_params += param.numel()

    percentage = 100 * trainable_params / all_params

    return (
        f"Trainable parameters: {trainable_params:,}\n"
        f"All parameters: {all_params:,}\n"
        f"Trainable percentage: {percentage:.2f}%"
    )


dataset = get_dataset()
base_model, tokenizer = get_model_and_tokenizer()

print("\n--- Model Parameters ---")
print(print_number_of_trainable_model_parameters(base_model))

# Baseline inference
i = 20
dialogue = dataset["test"][i]["dialogue"]
summary = dataset["test"][i]["summary"]

prompt = (
    "Summarize the following conversation.\n\n"
    f"{dialogue}\n\nSummary:"
)

device = next(base_model.parameters()).device
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=512
).to(device)

outputs = base_model.generate(
    **inputs,
    max_new_tokens=128
)

generated_summary = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("\n--- Original Dialogue ---")
print(dialogue)

print("\n--- Human-Written Summary ---")
print(summary)

print("\n--- Baseline Generated Summary ---")
print(generated_summary)
