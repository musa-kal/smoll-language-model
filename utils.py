"""
Utility functions for training the TinyGPT model.

This module provides helper functions for:
1. Creating batches of data for training
2. Evaluating model performance on training/test splits
"""

import torch
from config import *

def get_batch(data, batch_size=1, context_size=1):
    """
    Create a batch of training examples from the dataset.
    
    This function randomly samples starting positions from the data and creates
    input-target pairs where the target is shifted by one position. This is used
    for the causal language modeling task where the model predicts the next token
    given the previous tokens.
    
    Args:
        data (torch.Tensor): The input data tensor containing token indices
        batch_size (int, optional): Number of sequences in the batch. Default: 1
        context_size (int, optional): Length of each sequence in the batch. Default: 1
        
    Returns:
        tuple: (x, y) where:
            - x (torch.Tensor): Input sequences of shape (batch_size, context_size) on device
            - y (torch.Tensor): Target sequences of shape (batch_size, context_size) on device
                               Each y[i] contains the tokens that follow x[i] by one position
    """
    ix = torch.randint(len(data) - context_size, (batch_size,))
    x = torch.stack([data[i : i+context_size] for i in ix])
    y = torch.stack([data[i+1 : i+context_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model:torch.nn.Module, splits:tuple, eval_iters=1, batch_size=1, context_size=1):
    """
    Evaluate the model's average loss on one or more data splits.
    
    This function evaluates the model in evaluation mode and computes the average
    cross-entropy loss over multiple batches. It's commonly used to assess model
    performance on training and test sets during training.
    
    Args:
        model (torch.nn.Module): The language model to evaluate
        splits (tuple): One or more data tensors to evaluate on. Can be a single tensor
                       or a tuple of tensors (e.g., (train_data, test_data))
        eval_iters (int, optional): Number of batches to evaluate on for each split. Default: 1
        batch_size (int, optional): Batch size for evaluation. Default: 1
        context_size (int, optional): Sequence length for each sample. Default: 1
        
    Returns:
        list: Average losses for each split. If splits is a tuple of N tensors,
              returns a list of N average loss values.
              
    Notes:
        - The function uses @torch.no_grad() decorator to disable gradient computation
        - Model is set to eval mode during evaluation and restored to train mode after
        - Loss is computed on the entire batch (all time steps) using cross-entropy
    """
    out = []

    model.eval()

    for data in splits:
        losses = torch.zeros(eval_iters)

        for i in range(eval_iters):
            xb, yb = get_batch(data, batch_size, context_size)
            logits, loss = model(xb, yb)
            losses[i] = loss.item()

        out.append(losses.mean())

    model.train()

    return out