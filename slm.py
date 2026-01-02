# ======== Model Params ========
training_text_path = r'dataset\tinyshakespeare.txt'
training_split = 0.85
context_size = 10
batch_size = 4
torch_seed = 1337
# ==============================

# importing required modules
import torch

if type(torch_seed) == int:
    torch.manual_seed(torch_seed)
    print(f"=== torch seed seed to {torch_seed} ===")

# training loading text
text = ""
with open(training_text_path) as f:
    text = f.read()


# getting all the chars from within the text
chars = sorted(set(text))

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

def get_batch(data, batch_size=batch_size):
    ix = torch.randint(len(data) - context_size, (batch_size,))
    x = torch.stack([data[i : i+context_size] for i in ix])
    y = torch.stack([data[i+1 : i+context_size+1] for i in ix])
    return x, y

print(get_batch(training_data))