import torch

def get_batch(data, batch_size=1, context_size=1):
    ix = torch.randint(len(data) - context_size, (batch_size,))
    x = torch.stack([data[i : i+context_size] for i in ix])
    y = torch.stack([data[i+1 : i+context_size+1] for i in ix])
    return x, y
