"""
Language model implementations for a small-scale transformer-based GPT model.

This module contains implementations of various neural network architectures
for language modeling, ranging from simple bigram models to more sophisticated
transformer-based architectures with multi-head attention and positional embeddings.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import *
from tqdm import tqdm


class BigramLanguageModel(nn.Module):
    """
    A simple bigram language model that predicts the next token based on the previous token.
    
    This is the simplest form of a language model where the probability of the next token
    depends only on the current token. It learns a simple embedding table that maps each
    token to the logits for the next token.
    """

    def __init__(self, vocab_size):
        """
        Initialize the Bigram Language Model.
        
        Args:
            vocab_size (int): The size of the vocabulary (number of unique tokens)
        """
        super().__init__()

        # vocab_size x vocab_size matrix containing the probability of next token based on the previous token
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, target=None):
        """
        Forward pass through the bigram model.
        
        Args:
            idx (torch.Tensor): Input token indices of shape (B, T) where B is batch size and T is sequence length
            target (torch.Tensor, optional): Target token indices of shape (B, T). If provided, computes cross-entropy loss.
            
        Returns:
            tuple: (logits, loss)
                - logits (torch.Tensor): Raw output scores of shape (B, T, vocab_size)
                - loss (torch.Tensor or None): Cross-entropy loss if target is provided, else None
        """
        logits = self.token_embedding_table(idx) # dimension (B, T, C)

        loss = None

        if target is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), target.view(B*T))

        return logits, loss 
    

    def generate(self, idx, token_amount):
        """
        Generate new tokens autoregressively using the trained model.
        
        Args:
            idx (torch.Tensor): Initial token indices of shape (B, 1)
            token_amount (int): Number of tokens to generate
            
        Returns:
            torch.Tensor: Generated token indices of shape (B, 1 + token_amount)
        """

        for _ in range(token_amount):

            logits, loss = self(idx)

            logits = logits[:,-1,:] # only the last one from each batch

            prob = F.softmax(logits, dim=1)

            next_idx = torch.multinomial(prob, num_samples=1) # sample from the probability distribution

            idx = torch.cat((idx, next_idx), dim=1) # B, T+1

        return idx

    
    def fit(self, training_data, batch_size, epoch, lr, eval_interval=None, eval_iters=None, test_data=None):
        """
        Train the bigram model using AdamW optimizer.
        
        Args:
            training_data (torch.Tensor): Training dataset
            batch_size (int): Batch size for training
            epoch (int): Number of training epochs
            lr (float): Learning rate for AdamW optimizer
            eval_interval (int, optional): Evaluate every N steps
            eval_iters (int, optional): Number of iterations for evaluation
            test_data (torch.Tensor, optional): Test dataset for evaluation
        """

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
    """
    A single attention head implementing scaled dot-product attention.
    
    This is the fundamental building block of multi-head attention in transformers.
    It computes attention weights between queries and keys, scales them, applies
    a causal mask (to prevent attending to future tokens), and uses these weights
    to compute a weighted sum of values.
    """

    def __init__(self, n_embed, head_size, context_size, dropout):
        """
        Initialize an attention head.
        
        Args:
            n_embed (int): Embedding dimension (size of the model)
            head_size (int): Size of this particular attention head
            context_size (int): Maximum context length (for causal mask)
            dropout (float): Dropout probability for attention weights
        """
        super().__init__()
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through the attention head.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, n_embed)
            
        Returns:
            torch.Tensor: Attention output of shape (B, T, head_size)
        """
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
    """
    Multi-head self-attention layer.
    
    Runs multiple attention heads in parallel and concatenates their outputs.
    This allows the model to attend to different representation subspaces simultaneously.
    The concatenated output is then projected back to the original embedding dimension.
    """

    def __init__(self, head_num, n_embed, head_size, context_size, dropout):
        """
        Initialize multi-head attention.
        
        Args:
            head_num (int): Number of attention heads
            n_embed (int): Total embedding dimension
            head_size (int): Size of each individual head
            context_size (int): Maximum context length for causal masking
            dropout (float): Dropout probability
        """
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(n_embed, head_size, context_size, dropout) for _ in range(head_num)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through multi-head attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, n_embed)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, T, n_embed)
        """
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(x))
    

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    A simple two-layer fully connected network applied to each position independently.
    The first layer expands the dimension (typically by a factor of 4), applies ReLU activation,
    and the second layer projects back to the original dimension.
    """
    
    def __init__(self, input_size, output_size, dropout):
        """
        Initialize the feed-forward network.
        
        Args:
            input_size (int): Input dimension
            output_size (int): Hidden layer dimension
            dropout (float): Dropout probability
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, input_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Forward pass through the feed-forward network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, T, input_size)
        """
        return self.net(x)
    

class TransformerBlock(nn.Module):
    """
    A transformer block combining multi-head attention and feed-forward layers.
    
    This is the core building block of transformer models. It consists of:
    1. Multi-head self-attention with residual connection and layer normalization
    2. Position-wise feed-forward network with residual connection and layer normalization
    
    The architecture follows the "post-normalization" style where normalization is applied
    after the residual addition (also called "pre-norm" when applied before the sublayer).
    """
        
    def __init__(self, head_num, vocab_size, n_embed, context_size, dropout):
        """
        Initialize a transformer block.
        
        Args:
            head_num (int): Number of attention heads
            vocab_size (int): Vocabulary size (used for compatibility, not directly needed)
            n_embed (int): Embedding dimension
            context_size (int): Maximum context length
            dropout (float): Dropout probability
        """
        super().__init__()
        self.head_size = n_embed // head_num
        self.sa_head = MultiHeadAttention(head_num, n_embed, self.head_size, context_size, dropout)
        self.f_fwd = FeedForward(n_embed, n_embed * 4, dropout)
        
        # layer normalization for attention head
        self.ln1 = nn.LayerNorm(n_embed)
        # layer normalization for feedforward layer
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        """
        Forward pass through the transformer block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, n_embed)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, T, n_embed)
        """
        x = x + self.sa_head(self.ln1(x)) # applying multi head of self attention
        x = x + self.f_fwd(self.ln2(x)) # dimension (B, T, n_embed)
        return x


class TinyGPT(nn.Module):
    """
    A small-scale GPT-like language model.
    
    This model combines token embeddings, positional embeddings, multiple transformer blocks,
    and a language modeling head to perform next-token prediction. It supports both training
    (with loss computation) and inference (text generation).
    
    Architecture overview:
    1. Token embedding: Maps token IDs to embedding vectors
    2. Positional embedding: Adds positional information to embeddings
    3. Transformer blocks: Multiple layers of self-attention and feed-forward networks
    4. Layer normalization: Normalizes features before the output layer
    5. Language modeling head: Projects embeddings to vocabulary logits
    """

    def __init__(self, vocab_size, n_embed, context_size, head_num, layer_num, dropout):
        """
        Initialize TinyGPT model.
        
        Args:
            vocab_size (int): Size of the vocabulary
            n_embed (int): Embedding dimension
            context_size (int): Maximum context length (positional embeddings)
            head_num (int): Number of attention heads per block
            layer_num (int): Number of transformer blocks
            dropout (float): Dropout probability throughout the model
        """
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
        """
        Forward pass through the model.
        
        Args:
            idx (torch.Tensor): Input token indices of shape (B, T)
            target (torch.Tensor, optional): Target token indices of shape (B, T) for loss computation
            
        Returns:
            tuple: (logits, loss)
                - logits (torch.Tensor): Model predictions of shape (B, T, vocab_size)
                - loss (torch.Tensor or None): Cross-entropy loss if target is provided, else None
        """
        B, T = idx.shape
        
        device = idx.device

        token_embed = self.token_embedding_table(idx) # dimension (B, T, n_embed)
        pos_embed = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embed)
        x = token_embed + pos_embed
        x = self.tf_blocks(x) # (B, T, n_embed)
        x = self.f_ln(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None

        if target is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), target.view(B*T))

        return logits, loss 
    

    def generate(self, idx, token_amount):
        """
        Generate new tokens autoregressively.
        
        Starting from the provided token indices, this method generates new tokens one at a time
        by sampling from the model's output distribution. The context is limited to context_size
        to match the model's training constraints.
        
        Args:
            idx (torch.Tensor): Starting token indices of shape (B, T) where T <= context_size
            token_amount (int): Number of new tokens to generate
            
        Returns:
            torch.Tensor: Generated token indices of shape (B, T + token_amount)
        """

        for _ in range(token_amount):

            idx_context = idx[:, -self.context_size:] # at most context_size from each batch

            logits, loss = self(idx_context)

            logits = logits[:,-1,:] # only the last one from each batch -> B, C
            
            prob = F.softmax(logits, dim=1)
            
            next_idx = torch.multinomial(prob, num_samples=1) # sample from the probability distribution
            
            idx = torch.cat((idx, next_idx), dim=1) # B, T+1

        return idx

    
    def fit(self, training_data, batch_size, epoch, lr, eval_interval=None, eval_iters=None, test_data=None):
        """
        Train the TinyGPT model using AdamW optimizer.
        
        Trains the model on batches of data with optional evaluation on training and test sets.
        Uses learning rate as specified and optimizes using the AdamW optimizer.
        
        Args:
            training_data (torch.Tensor): Training dataset
            batch_size (int): Number of samples per batch
            epoch (int): Number of training epochs
            lr (float): Learning rate for AdamW optimizer
            eval_interval (int, optional): Evaluate every N steps. If None, no evaluation is performed
            eval_iters (int, optional): Number of iterations for loss estimation during evaluation
            test_data (torch.Tensor, optional): Test dataset for evaluation. If None, only training loss is evaluated
        """

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