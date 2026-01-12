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