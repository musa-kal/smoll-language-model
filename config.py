import torch

# ======== Model Params ========
training_text_path = r'dataset\tinyshakespeare.txt'
training_split = 0.85
context_size = 10
batch_size = 4
torch_seed = 1337
evaluate_iteration = 500
evaluate_interval = 1000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 1e-3
n_embd = 32
epoch = 5000
# ==============================