import torch
import torch.nn as nn
from torch.nn import functional as F
import sentencepiece as spm

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 64 # what is the maximum context length for predictions?
epochs = 60000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 1000
n_embd = 128
n_head = 16
n_layer = 16
dropout = 0.0
vocab_size = 5000

def train_tokenizer():
    spm.SentencePieceTrainer.train(input='learning-transformers/alex.txt', model_prefix='alex', vocab_size=vocab_size)
# load tokenizer model
tokenizer = spm.SentencePieceProcessor()
tokenizer.Load('learning-transformers/alex.model')
encode = lambda x: tokenizer.EncodeAsIds(x)
decode = lambda x: tokenizer.DecodeIds(x) 

# convert text to training and validation data
with open('learning-transformers/alex.txt', 'r', encoding="utf-8") as f:
    text = f.read()
encoded_text = encode(text)
ids = torch.tensor(encoded_text, dtype=torch.long)
n = 0.9 # percent of data to use for training
train_ids = ids[:int(n*len(ids))]
val_ids = ids[int(n*len(ids)):]


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_ids if split == 'train' else val_ids
    ix = torch.randint(len(ids) - block_size, (batch_size,))
    x = torch.stack([ids[i:i+block_size] for i in ix])
    y = torch.stack([ids[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.MHA = nn.MultiheadAttention(n_embd, n_head, dropout=dropout)
        
    def forward(self, x, is_causal=True):
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        x, _ = self.MHA(q, k, v, need_weights=False)       
        return x
    
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.feedforward(x)

class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = MultiHeadAttention()
            self.ff = FeedForward()
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)
        
        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.ff(self.ln2(x))
            return x

class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)
        self.ln_f = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, x, targets=None):
        B, T = x.shape
        tok_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, x, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = x[:, -block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            x = torch.cat((x, idx_next), dim=1) # (B, T+1)
        return x

def train():
    model = TransformerModel().to(device)
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

train()