# Dialogue Summarization using LoRA

Fine-tuning **FLAN-T5-Small** for dialogue summarization using LoRA (Low-Rank Adaptation).

### Tech Stack
- Python, PyTorch
- Hugging Face Transformers and Datasets
- PEFT
- ROUGE Evaluation

### Dataset
[DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum)

### Approach
- Loaded the pretrained FLAN-T5-Small model.
- Prepared and tokenized dialogue data.
- Applied LoRA for parameter-efficient fine-tuning.
- Trained on 500 examples for one epoch.
- Evaluated summaries using ROUGE metrics.

### Run
```bash
python train.py
python evaluate_model.py