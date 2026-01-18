"""
Interactive playground for the TinyGPT model.

This script provides an interactive interface to:
1. Load a trained TinyGPT model
2. Input text sequences
3. Generate new tokens based on the input
4. Save generated text to .txt files
5. Experiment with different prompt lengths and generation parameters
"""

import torch
import os
from pathlib import Path
from config import *
from models import TinyGPT

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PARAM_PATH = "models/model_weights.pth"
TRAINING_TEXT_PATH = r'dataset\tinyshakespeare.txt'
OUTPUT_DIR = "out"

# Create output directory if it doesn't exist
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# ============================================================================
# SETUP: LOAD TEXT AND BUILD VOCABULARY
# ============================================================================

print("\n" + "="*70)
print("Loading training data and building vocabulary...")
print("="*70)

# Load the text file to rebuild vocabulary mappings
text = ""
with open(TRAINING_TEXT_PATH) as f:
    text = f.read()

# Extract all unique characters from the text
chars = sorted(set(text))
vocab_size = len(chars)

# Create bidirectional mappings between characters and indices
char_i_map = {c: i for i, c in enumerate(chars)}
i_char_map = {i: c for i, c in enumerate(chars)}

# Lambda functions for convenient encoding and decoding
encode = lambda seq: [char_i_map[c] for c in seq]
decode = lambda seq: "".join(i_char_map[i] for i in seq)

print(f"✓ Vocabulary size: {vocab_size}")
print(f"✓ Sample characters: {chars[:10]}")

# ============================================================================
# SETUP: LOAD MODEL
# ============================================================================

print("\n" + "="*70)
print("Loading trained model...")
print("="*70)

# Initialize model with same hyperparameters used for training
m = TinyGPT(vocab_size, n_embed, context_size, head_num, layer_num, dropout)
m = m.to(device)

# Load trained weights
try:
    m.load_state_dict(torch.load(MODEL_PARAM_PATH, map_location=device))
    print(f"✓ Model loaded from: {MODEL_PARAM_PATH}")
except FileNotFoundError:
    print(f"✗ Error: Model file not found at {MODEL_PARAM_PATH}")
    print("  Please run train.py first to train and save the model.")
    exit(1)

m.eval()  # Set to evaluation mode
print(f"✓ Device: {device}")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_text(prompt_text, num_tokens):
    """
    Generate text based on a prompt.
    
    Args:
        prompt_text (str): The starting text/prompt
        num_tokens (int): Number of tokens to generate
        
    Returns:
        str: The generated text
    """
    with torch.no_grad():
        # Encode the prompt
        try:
            prompt_tokens = encode(prompt_text)
        except KeyError as e:
            print(f"✗ Error: Character {e} not in vocabulary")
            return None
        
        # Limit prompt to context_size
        if len(prompt_tokens) > context_size:
            print(f"⚠ Prompt truncated from {len(prompt_tokens)} to {context_size} tokens")
            prompt_tokens = prompt_tokens[-context_size:]
        
        # Convert to tensor
        idx = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        # Generate tokens
        generated_idx = m.generate(idx, num_tokens)
        
        # Decode and return
        generated_text = decode(generated_idx[0].tolist())
        return generated_text


def save_text(text, filename=None):
    """
    Save generated text to a file in the output directory.
    
    Args:
        text (str): The text to save
        filename (str, optional): Custom filename without extension. 
                                 If None, generates a timestamp-based name.
    """
    import time
    
    if filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"generated_{timestamp}"
    
    # Ensure .txt extension
    if not filename.endswith(".txt"):
        filename += ".txt"
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"✓ Text saved to: {filepath}")
        return True
    except Exception as e:
        print(f"✗ Error saving file: {e}")
        return False


def display_menu():
    """Display the main menu."""
    print("\n" + "="*70)
    print("TinyGPT PLAYGROUND")
    print("="*70)
    print("1. Generate text from a prompt")
    print("2. Batch generate multiple times")
    print("3. Exit")
    print("="*70)


# ============================================================================
# MAIN INTERACTIVE LOOP
# ============================================================================

def main():
    """Main interactive playground loop."""
    print("\n✓ Ready to generate! Type 'help' for commands.")
    
    while True:
        display_menu()
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == "1":
            # Single text generation
            print("\n" + "-"*70)
            print("GENERATE FROM PROMPT")
            print("-"*70)
            
            prompt = input("Enter prompt text: ").strip()
            if not prompt:
                print("⚠ Prompt cannot be empty")
                continue
            
            try:
                num_tokens = int(input("Number of tokens to generate: ").strip())
                if num_tokens <= 0:
                    print("⚠ Number of tokens must be positive")
                    continue
            except ValueError:
                print("⚠ Please enter a valid number")
                continue
            
            print("\n⏳ Generating...")
            generated = generate_text(prompt, num_tokens)
            
            if generated:
                print("\n" + "-"*70)
                print("GENERATED TEXT:")
                print("-"*70)
                print(generated)
                print("-"*70)
                
                # Ask if user wants to save
                save_choice = input("\nSave this text? (y/n): ").strip().lower()
                if save_choice == 'y':
                    custom_name = input("Enter filename (press Enter for auto-generated): ").strip()
                    save_text(generated, custom_name if custom_name else None)
        
        elif choice == "2":
            # Batch generation
            print("\n" + "-"*70)
            print("BATCH GENERATE")
            print("-"*70)
            
            prompt = input("Enter prompt text: ").strip()
            if not prompt:
                print("⚠ Prompt cannot be empty")
                continue
            
            try:
                num_tokens = int(input("Number of tokens per generation: ").strip())
                num_generations = int(input("Number of generations: ").strip())
                
                if num_tokens <= 0 or num_generations <= 0:
                    print("⚠ Values must be positive")
                    continue
            except ValueError:
                print("⚠ Please enter valid numbers")
                continue
            
            print(f"\n⏳ Generating {num_generations} samples...")
            
            all_text = ""
            for i in range(num_generations):
                generated = generate_text(prompt, num_tokens)
                if generated:
                    all_text += f"\n--- Generation {i+1} ---\n{generated}\n"
            
            if all_text:
                print("\n✓ Batch generation complete!")
                save_choice = input("Save all generations to file? (y/n): ").strip().lower()
                if save_choice == 'y':
                    custom_name = input("Enter filename (press Enter for auto-generated): ").strip()
                    save_text(all_text, custom_name if custom_name else None)
        
        elif choice == "3":
            print("\n👋 Thanks for using TinyGPT Playground! Goodbye!")
            break
        
        else:
            print("⚠ Invalid option. Please select 1, 2, or 3.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")