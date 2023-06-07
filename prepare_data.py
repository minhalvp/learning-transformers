from datasets import load_dataset
import tiktoken
import os
import pickle

def load_data():
    enc = tiktoken.encoding_for_model("gpt2")
    text = ""
    if not os.path.exists("ml_papers.bin"):
        dataset = load_dataset("CShorten/ML-ArXiv-Papers")
        for i in dataset['train']:
            text += "###"+i['title'] + ".\n" + i['abstract'].strip() + "\n"
        encoded_text = enc.encode(text)
        with open("ml_papers.bin", "wb") as f:
            pickle.dump(encoded_text, f)
        return encoded_text
    else:
         with open("ml_papers.bin", "rb") as f:
            return pickle.load(f)