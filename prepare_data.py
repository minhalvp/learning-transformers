from datasets import load_dataset
import tiktoken
import os

def load_data():
    text = ""
    if not os.path.exists("ml_papers.txt"):
        dataset = load_dataset("CShorten/ML-ArXiv-Papers")
        for i in dataset['train']:
            text += "###"+i['title'] + ".\n" + i['abstract'].strip() + "\n"
        with open("ml_papers.txt", "w") as f:
            f.write(text)
        return text
    else:
         with open("ml_papers.txt", "r") as f:
            return f.read()

papers = load_data()
enc = tiktoken.get_encoding("cl100k_base")
encoded = enc.encode(papers)