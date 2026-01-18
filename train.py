import torch
from torch.nn import functional as F
from config import *

# importing required modules

print(f"=== Device Selected as [{device}] ===")

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

from models import nGramLanguageModel_V2

m = nGramLanguageModel_V2(vocab_size, n_embed, context_size, head_num, layer_num, dropout)
m = m.to(device)

input_text = ""

print(decode(m.generate(torch.zeros((1,1), dtype=torch.long, device=device), token_amount=100)[0].tolist()))

m.fit(training_data, batch_size, epoch, learning_rate, evaluate_interval, evaluate_iteration, testing_data)

print(decode(m.generate(torch.zeros((1,1), dtype=torch.long, device=device), token_amount=100)[0].tolist()))