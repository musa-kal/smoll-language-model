import torch
from torch.nn import functional as F

# ======== Model Params ========
training_text_path = r'dataset\tinyshakespeare.txt'
training_split = 0.85
context_size = 10
batch_size = 4
torch_seed = 1337
evaluate_iteration = 250
evaluate_interval = 1000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ==============================

# importing required modules

if type(torch_seed) == int:
    torch.manual_seed(torch_seed)
    print(f"=== torch seed seed to {torch_seed} ===")

# training loading text
text = ""
with open(training_text_path) as f:
    text = f.read()


# getting all the chars from within the text
chars = sorted(set(text))
vocab_size = len(chars)

# maps to encode and decode text
char_i_map = {c:i for i,c in enumerate(chars)}
i_char_map = {i:c for i,c in enumerate(chars)}

encode = lambda seq: [char_i_map[c] for c in seq]
decode = lambda seq: "".join(i_char_map[i] for i in seq)


# tokenizing training data
data = torch.tensor(encode(text), dtype=torch.long)
print("=== Data Encoded ===")
print(data.shape, data.dtype)

# train test split
if training_split > 1 or training_split <= 0:
    raise ValueError(f"training_split was {training_split}, it must be 0 < training_split <= 1")

num = int(len(data)*training_split)
training_data = data[:num]
testing_data = data[num:]

from models import BigramLanguageModel

m = BigramLanguageModel(vocab_size)
m = m.to(device)


# print(decode(m.generate(torch.zeros((1,1), dtype=torch.long), token_amount=100)[0].tolist()))

# m.fit(training_data, 32, 10000, 1e-3, evaluate_interval, evaluate_iteration, testing_data)

# print(decode(m.generate(torch.zeros((1,1), dtype=torch.long), token_amount=100)[0].tolist()))