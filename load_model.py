
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from config import MODEL_NAME

def get_model_and_tokenizer():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model = model.to(device)

    print("Model:", MODEL_NAME)
    print("Device:", device)

    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = get_model_and_tokenizer()
