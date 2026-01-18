import torch

# ======== Model Params ========
training_text_path = r'dataset\tinyshakespeare.txt'
model_save_path = r'models\tinygpt.pth'
training_split = 0.85
context_size = 256
batch_size = 64
torch_seed = None#1337
evaluate_iteration = 250
evaluate_interval = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 3e-4
n_embed = 384
epoch = 5000
head_num = 6
layer_num = 6
dropout = 0.2
# ==============================