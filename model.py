import torch
import torch.nn as nn
from torch.nn import functional as F
import sentencepiece as spm

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
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
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.feedforward(x)

class Block(nn.Module):
        def __init__(self, n_embd, n_head, dropout):
            super().__init__()
            self.attn = MultiHeadAttention(n_embd=n_embd, n_head=n_head, dropout=dropout)
            self.ff = FeedForward(n_embd=n_embd, dropout=dropout)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)
        
        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.ff(self.ln2(x))
            return x

class TransformerModel(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, vocab_size, block_size, dropout, device):
        super().__init__()
        self.device = device
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)
        self.ln_f = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd=n_embd, n_head=n_head, dropout=dropout) for _ in range(n_layer)])
        self.head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, x, targets=None):
        B, T = x.shape
        tok_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding(torch.arange(T, device=self.device))
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
            idx_cond = x[:, -self.block_size:]
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