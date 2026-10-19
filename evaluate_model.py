
import torch
import evaluate
from datasets import load_dataset
from peft import AutoPeftModelForSeq2SeqLM
from transformers import AutoTokenizer

from config import DATASET_NAME

ADAPTER_PATH = "./peft-dialogue-summary-checkpoint-local"
NUM_EXAMPLES = 20

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Load the fine-tuned LoRA model
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

model = AutoPeftModelForSeq2SeqLM.from_pretrained(
    ADAPTER_PATH,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)
model = model.to(device)
model.eval()

# Load test data
dataset = load_dataset(DATASET_NAME)
test_data = dataset["test"].select(
    range(min(NUM_EXAMPLES, len(dataset["test"])))
)

predictions = []
references = []

# Generate summaries
for i, example in enumerate(test_data):
    prompt = (
        "Summarize the following conversation:\n\n"
        + example["dialogue"]
        + "\n\nSummary:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=2,
        )

    prediction = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    predictions.append(prediction)
    references.append(example["summary"])

    if i < 3:
        print(f"\n--- Example {i + 1} ---")
        print("Dialogue:", example["dialogue"])
        print("Reference:", example["summary"])
        print("Generated:", prediction)

# Calculate ROUGE scores
rouge = evaluate.load("rouge")
scores = rouge.compute(
    predictions=predictions,
    references=references,
    use_stemmer=True,
)

print("\n--- Fine-tuned Model ROUGE Scores ---")
for metric, score in scores.items():
    print(f"{metric}: {score:.4f}")
