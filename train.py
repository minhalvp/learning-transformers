import torch
import tiktoken
from model import TransformerModel
from prepare_data import load_data


# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
epochs = 60000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 1000
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0

tokenizer = tiktoken.get_encoding("cl100k_base")

def encode(x):
    return tokenizer.encode(x)

def decode(x):
    return tokenizer.decode(x)

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_ids if split == 'train' else val_ids
    ix = torch.randint(len(ids) - block_size, (batch_size,))
    x = torch.stack([ids[i:i+block_size] for i in ix])
    y = torch.stack([ids[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# convert text to training and validation data
text = load_data()
encoded_text = encode(text)
print("Text encoded")
ids = torch.tensor(encoded_text, dtype=torch.long)
n = 0.9 # percent of data to use for training
train_ids = ids[:int(n*len(ids))]
val_ids = ids[int(n*len(ids)):]

model = TransformerModel(
    n_head=n_head,
    n_embd=n_embd,
    n_layer=n_layer,
    vocab_size=tokenizer.n_vocab,
    block_size=block_size,
    device=device,
    dropout=dropout
).to(device)
# find model paramaters
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(epochs):
    model.train()
    x, y = get_batch('train')
    _, loss = model(x, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % eval_interval == 0 or iter == epochs - 1:
        model.eval()
        with torch.no_grad():
            x, y = get_batch('val')
            logits, loss = model(x, y)
            print(f'epoch {epoch} | loss {loss.item()}')
            x = x[:1]
    if epoch % eval_iters == 0:
        model.eval()
        with torch.no_grad():
            x, y = get_batch('val')
            logits, loss = model(x, y)
            print(f'epoch {epoch} | loss {loss.item()}')
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))
            x = x[:1]