import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import *
from tqdm import tqdm


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        # vocab_size x vocab_size matrix containing the probability of next token based on the previous token 
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, target=None):
        logits = self.token_embedding_table(idx) # dimension (B, T, C)

        loss = None

        if target is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), target.view(B*T))

        return logits, loss 
    

    def generate(self, idx, token_amount):

        for _ in range(token_amount):

            logits, loss = self(idx)

            logits = logits[:,-1,:] # only the last one from each batch

            prob = F.softmax(logits, dim=1)

            next_idx = torch.multinomial(prob, num_samples=1) # sample form the probability distribution

            idx = torch.cat((idx, next_idx), dim=1) # b, T+1

        return idx

    
    def fit(self, training_data, batch_size, epoch, lr, eval_interval=None, eval_iters=None, test_data=None):

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        if test_data is not None:
            eval_data = (training_data, test_data)
        else:
            eval_data = (training_data)

        print("training...")
        for step in tqdm(range(epoch)):

            if eval_interval is not None and eval_iters is not None and step % eval_interval == 0:
                out = estimate_loss(self, eval_data, eval_iters)
                s = f"step {step+1}/{epoch}: train loss {out[0]:.4f}" + ("" if training_data is None else f" test loss {out[1]:.4f}")
                tqdm.write(s)

            xb, yb = get_batch(training_data, batch_size=batch_size)

            logits, loss = self(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


class AttentionHead(nn.Module):

    def __init__(self, n_embed, head_size, context_size):
        super().__init__()

        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))

    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * C ** -0.5 # B, T, C @ B, C, T -> B, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        out = wei @ self.value(x) # B, T, T @ B, T, C -> B, T, C

        return out

        
class BigramLanguageModel_V2(nn.Module):

    def __init__(self, vocab_size, n_embed, context_size, device="cpu"):
        super().__init__()

        # vocab_size x vocab_size matrix containing the probability of next token based on the previous token 
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(context_size, n_embed)
        self.sa_head = AttentionHead(n_embed, n_embed, context_size)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.device = device
        self.context_size = context_size

    def forward(self, idx, target=None):
        B, T = idx.shape

        token_embed = self.token_embedding_table(idx) # dimension (B, T, n_embed)
        pos_embed = self.position_embedding_table(torch.arange(T, device=self.device)) # (T, C)
        x = token_embed + pos_embed
        x = self.sa_head(x) # applying orn head of self attention
        logits = self.lm_head(x) # dimension (B, T, vocab_size)

        loss = None

        if target is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), target.view(B*T))

        return logits, loss 
    

    def generate(self, idx, token_amount):

        for _ in range(token_amount):

            idx_context = idx[:, -self.context_size:] # at most context_size from each batch

            logits, loss = self(idx_context)

            logits = logits[:,-1,:] # only the last one from each batch -> B, C
            
            prob = F.softmax(logits, dim=1)
            
            next_idx = torch.multinomial(prob, num_samples=1) # sample form the probability distribution
            
            idx = torch.cat((idx, next_idx), dim=1) # B, T+1

        return idx

    
    def fit(self, training_data, batch_size, epoch, lr, eval_interval=None, eval_iters=None, test_data=None):

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        if test_data is not None:
            eval_data = (training_data, test_data)
        else:
            eval_data = (training_data)

        print("training...")
        for step in tqdm(range(epoch)):

            if eval_interval is not None and eval_iters is not None and step % eval_interval == 0:
                out = estimate_loss(self, eval_data, eval_iters)
                s = f"step {step+1}/{epoch}: train loss {out[0]:.4f}" + ("" if training_data is None else f" test loss {out[1]:.4f}")
                tqdm.write(s)

            xb, yb = get_batch(training_data, batch_size=batch_size)

            logits, loss = self(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()