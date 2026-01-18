"""
Training script for TinyGPT language model.

This script handles the complete training pipeline including:
1. Loading and preprocessing text data
2. Building character-level vocabulary
3. Creating train/test split
4. Initializing the TinyGPT model
5. Training the model with evaluation
6. Saving trained model weights
7. Generating sample text before and after training

Configuration parameters are imported from config.py
"""

import torch
from torch.nn import functional as F
from config import *

print(f"=== Device Selected as [{device}] ===")

if type(torch_seed) == int:
    torch.manual_seed(torch_seed)
    print(f"=== torch seed seed to {torch_seed} ===")

# ============================================================================
# STEP 1: LOAD AND PREPROCESS TEXT DATA
# ============================================================================

# Load the text file for training
text = ""
with open(training_text_path) as f:
    text = f.read()


# ============================================================================
# STEP 2: BUILD CHARACTER-LEVEL VOCABULARY
# ============================================================================

# Extract all unique characters from the text to create vocabulary
chars = sorted(set(text))
vocab_size = len(chars)

# Create bidirectional mappings between characters and indices
# This allows us to encode text to token indices and decode indices back to text
char_i_map = {c:i for i,c in enumerate(chars)}
i_char_map = {i:c for i,c in enumerate(chars)}

# Lambda functions for convenient encoding and decoding
encode = lambda seq: [char_i_map[c] for c in seq]
decode = lambda seq: "".join(i_char_map[i] for i in seq)


# ============================================================================
# STEP 3: TOKENIZE AND PREPARE DATA
# ============================================================================

# Convert entire text to tensor of token indices
data = torch.tensor(encode(text), dtype=torch.long)
print("=== Data Encoded ===")
print(data.shape, data.dtype)

# ============================================================================
# STEP 4: TRAIN/TEST SPLIT
# ============================================================================

# Validate training split parameter
if training_split > 1 or training_split <= 0:
    raise ValueError(f"training_split was {training_split}, it must be 0 < training_split <= 1")

# Split data into training and testing sets
num = int(len(data)*training_split)
training_data = data[:num]
testing_data = data[num:]

# ============================================================================
# STEP 5: INITIALIZE MODEL
# ============================================================================

from models import TinyGPT

# Create model with hyperparameters from config
m = TinyGPT(vocab_size, n_embed, context_size, head_num, layer_num, dropout)
m = m.to(device)

# ============================================================================
# STEP 6: SAMPLE GENERATION BEFORE TRAINING
# ============================================================================

# Generate 100 tokens from an untrained model to show random output
print("\n--- Generated Text BEFORE Training (Random) ---")
print(decode(m.generate(torch.zeros((1,1), dtype=torch.long, device=device), token_amount=100)[0].tolist()))

# ============================================================================
# STEP 7: TRAIN THE MODEL
# ============================================================================

m.train()
m.fit(training_data, batch_size, epoch, learning_rate, evaluate_interval, evaluate_iteration, testing_data)

# ============================================================================
# STEP 8: SAVE MODEL WEIGHTS
# ============================================================================

torch.save(m.state_dict(), model_save_path)
print(f"\n+++ Model saved to {model_save_path} +++")

# ============================================================================
# STEP 9: SAMPLE GENERATION AFTER TRAINING
# ============================================================================

# Generate 100 tokens from the trained model to demonstrate learned patterns
print("\n--- Generated Text AFTER Training ---")
print(decode(m.generate(torch.zeros((1,1), dtype=torch.long, device=device), token_amount=100)[0].tolist()))