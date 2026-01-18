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

    def __init__(self, n_embed, head_size, context_size, dropout):
        super().__init__()
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        head_size = k.shape[-1] # so i can scale the dot product by head size
        wei = q @ k.transpose(-2, -1) * head_size ** -0.5 
        
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ self.value(x) 

        return out
    

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self attention """

    def __init__(self, head_num, n_embed, head_size, context_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(n_embed, head_size, context_size, dropout) for _ in range(head_num)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(x))
    

class FeedForward(nn.Module):
    
    def __init__(self, input_size, output_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, input_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    

class TransformerBlock(nn.Module):
    """ A transformer block """
        
    def __init__(self, head_num, vocab_size, n_embed, context_size, dropout):
        super().__init__()
        self.head_size = n_embed // head_num
        self.sa_head = MultiHeadAttention(head_num, n_embed, self.head_size, context_size, dropout)
        self.f_fwd = FeedForward(n_embed, n_embed * 4, dropout)
        
        # layer normalization for attentions head
        self.ln1 = nn.LayerNorm(n_embed)
        # layer normalization for feedforward layer
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x)) # applying multi head of self attention
        x = x + self.f_fwd(self.ln2(x)) # dimension (B, T, vocab_size)
        return x


class TinyGPT(nn.Module):

    def __init__(self, vocab_size, n_embed, context_size, head_num, layer_num, dropout):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(context_size, n_embed)
        self.tf_blocks = nn.Sequential(
            *[TransformerBlock(head_num, vocab_size, n_embed, context_size, dropout) for _ in range(layer_num)]
        )
        self.f_ln = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.context_size = context_size

    def forward(self, idx, target=None):
        B, T = idx.shape
        
        device = idx.device

        token_embed = self.token_embedding_table(idx) # dimension (B, T, n_embed)
        pos_embed = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = token_embed + pos_embed
        x = self.tf_blocks(x) # (B, T, C)
        x = self.f_ln(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

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
                out = estimate_loss(self, eval_data, eval_iters, batch_size=32, context_size=self.context_size)
                s = f"step {step+1}/{epoch}: train loss {out[0]:.4f}" + ("" if training_data is None else f" | test loss {out[1]:.4f}")
                tqdm.write(s)

            xb, yb = get_batch(training_data, batch_size, self.context_size)

            logits, loss = self(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()