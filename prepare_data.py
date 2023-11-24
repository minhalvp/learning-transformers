from datasets import load_dataset
from transformers import PreTrainedTokenizerFast


tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer/")
def load_data() -> list[int]:
    tokens = []
    def tokenize(example):
        encoded_text = tokenizer.encode(example['text'])
        tokens.extend(encoded_text)
    dataset = load_dataset("roneneldan/TinyStories")
    dataset = dataset.map(tokenize)

    return tokens

tokens = load_data()
tokens[100:]