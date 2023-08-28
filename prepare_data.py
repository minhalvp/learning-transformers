import tiktoken
from datasets import load_dataset

enc = tiktoken.encoding_for_model("gpt2")

def load_data():
    enc = tiktoken.encoding_for_model("gpt2")
    tokens = []
    def tokenize(example):
        text = "###"+example['title'] + ".\n" + example['abstract'].strip()
        encoded_text = enc.encode(text)
        tokens.extend(encoded_text)
    dataset = load_dataset("CShorten/ML-ArXiv-Papers").remove_columns(["Unnamed: 0.1", "Unnamed: 0"])
    dataset = dataset.map(tokenize).remove_columns(["title", "abstract"])

    return tokens
