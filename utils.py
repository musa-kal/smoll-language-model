import torch
from config import *

def get_batch(data, batch_size=1, context_size=1):
    ix = torch.randint(len(data) - context_size, (batch_size,))
    x = torch.stack([data[i : i+context_size] for i in ix])
    y = torch.stack([data[i+1 : i+context_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model:torch.nn.Module, splits:tuple, eval_iters=1, batch_size=1, context_size=1):
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