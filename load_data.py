
from datasets import load_dataset
from config import DATASET_NAME

def get_dataset():
    dataset = load_dataset(DATASET_NAME)
    print(dataset)
    print("\nSample:")
    print(dataset["train"][0])
    return dataset


if __name__ == "__main__":
    dataset = get_dataset()
