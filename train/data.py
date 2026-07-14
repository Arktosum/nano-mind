import os
import requests
import torch

# --- Configuration ---
DATA_URL = 'https://www.gutenberg.org/cache/epub/11/pg11.txt'
FILE_PATH = 'alice.txt'
BLOCK_SIZE = 64  # Maximum context length for predictions
BATCH_SIZE = 16  # How many independent sequences will we process in parallel?

def download_data():
    """Downloads Alice in Wonderland from Project Gutenberg if not already local."""
    if not os.path.exists(FILE_PATH):
        print(f"Downloading {FILE_PATH}...")
        response = requests.get(DATA_URL)
        response.raise_for_status()
        
        text = response.text
        # Optional: Strip standard Gutenberg Header and Footer to get just the story
        start_idx = text.find('*** START OF THE PROJECT GUTENBERG EBOOK')
        end_idx = text.find('*** END OF THE PROJECT GUTENBERG EBOOK')
        if start_idx != -1 and end_idx != -1:
            # Skip the actual header line
            start_idx = text.find('\n', start_idx) + 1
            text = text[start_idx:end_idx]
            
        with open(FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(text)
        print("Download complete!")
    
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        return f.read()

# 1. Load the raw text
text = download_data()
print(f"Length of dataset in characters: {len(text)}")

# 2. Build the Vocabulary
# Get all unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")
print(f"Vocabulary: {''.join(chars)}")

# 3. Create the Tokenizer
# Dictionaries to map characters to integers and vice versa
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: string -> list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: list of integers -> string

# 4. Encode the dataset into PyTorch Tensors
data = torch.tensor(encode(text), dtype=torch.long)

# 5. Split into Train and Validation sets (90% / 10%)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# 6. Data Loader Function
def get_batch(split):
    """
    Generates a small batch of data of inputs x and targets y.
    x is the context, y is the target character we want to predict.
    """
    data_source = train_data if split == 'train' else val_data
    
    # Generate BATCH_SIZE random starting indices
    ix = torch.randint(len(data_source) - BLOCK_SIZE, (BATCH_SIZE,))
    
    # x is the context window
    x = torch.stack([data_source[i:i+BLOCK_SIZE] for i in ix])
    
    # y is the target (shifted one character to the right)
    y = torch.stack([data_source[i+1:i+BLOCK_SIZE+1] for i in ix])
    
    return x, y

if __name__ == '__main__':
    # Test the batch generation
    xb, yb = get_batch('train')
    print("\n--- Batch Test ---")
    print("Input shape (x):", xb.shape)
    print("Target shape (y):", yb.shape)
    print("\nContext (x[0]):", decode(xb[0].tolist()))
    print("Target (y[0]): ", decode(yb[0].tolist()))