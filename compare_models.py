
import torch
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import AutoPeftModelForSeq2SeqLM

from config import DATASET_NAME, MODEL_NAME

device = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EXAMPLES = 100
ADAPTER_PATH = "./peft-dialogue-summary-checkpoint-local"

dataset = load_dataset(DATASET_NAME)["test"]
dataset = dataset.select(range(min(NUM_EXAMPLES, len(dataset))))

base_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
base_model.eval()

ft_tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
ft_model = AutoPeftModelForSeq2SeqLM.from_pretrained(
    ADAPTER_PATH,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)
ft_model.eval()


def generate_summaries(model, tokenizer):
    predictions = []

    for example in dataset:
        prompt = (
            "Summarize the following conversation:\n\n"
            + example["dialogue"]
            + "\n\nSummary:"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=128,
                num_beams=2,
            )

        predictions.append(
            tokenizer.decode(output[0], skip_special_tokens=True)
        )

    return predictions


references = [example["summary"] for example in dataset]

print("Generating baseline summaries...")
baseline_predictions = generate_summaries(base_model, base_tokenizer)

print("Generating fine-tuned summaries...")
ft_predictions = generate_summaries(ft_model, ft_tokenizer)

rouge = evaluate.load("rouge")

baseline_scores = rouge.compute(
    predictions=baseline_predictions,
    references=references,
    use_stemmer=True,
)

ft_scores = rouge.compute(
    predictions=ft_predictions,
    references=references,
    use_stemmer=True,
)

print("\n--- Baseline vs Fine-tuned ---")
for metric in baseline_scores:
    before = baseline_scores[metric]
    after = ft_scores[metric]
    print(
        f"{metric}: baseline={before:.4f}, "
        f"fine-tuned={after:.4f}, "
        f"change={after - before:+.4f}"
    )

for i in range(min(3, len(dataset))):
    print(f"\n--- Example {i + 1} ---")
    print("Reference:", references[i])
    print("Baseline: ", baseline_predictions[i])
    print("Fine-tuned:", ft_predictions[i])
