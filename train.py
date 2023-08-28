import torch
import tiktoken
from model import TransformerModel
from prepare_data import load_data
import wandb
wandb.init(project='my-project', name='my-run', mode='offline')
# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
epochs = 60000
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 1000
n_embd = 72
n_head = 8
n_layer = 8
dropout = 0.1
# wandb log hyperparameters
wandb.config.batch_size = batch_size
wandb.config.block_size = block_size
wandb.config.epochs = epochs
wandb.config.learning_rate = learning_rate
wandb.config.n_embd = n_embd
wandb.config.n_head = n_head
wandb.config.n_layer = n_layer
wandb.config.dropout = dropout
tokenizer = tiktoken.encoding_for_model("gpt2")

def encode(x):
    return tokenizer.encode(x)

def decode(x):
    return tokenizer.decode(x)

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_ids if split == 'train' else val_ids
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y

# convert text to training and validation data
encoded_text = load_data()
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
model = torch.compile(model)
# find model paramaters
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(epochs):
    model.train()
    x, y = get_batch('train')
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % eval_interval == 0 or iter == epochs - 1:
        model.eval()
        with torch.no_grad():
            x, y = get_batch('val')
            logits, loss = model(x, y)
            wandb.log({'epoch': epoch, 'loss': loss.item()})
            print(f'epoch {epoch} | loss {loss.item()}')
            x = x[:1]
    if epoch % eval_iters == 0:
        model.eval()
        with torch.no_grad():
            x, y = get_batch('val')
            logits, loss = model(x, y)
            wandb.log({'epoch': epoch, 'loss': loss.item()})
            print(f'epoch {epoch} | loss {loss.item()}')
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))
            x = x[:1]