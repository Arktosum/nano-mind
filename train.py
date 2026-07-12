import torch
import json
import torch.nn.functional as F
# Note: Assuming your architecture file is named model.py.
# If you named it architecture.py, change this to: from architecture import ...
from model import NanoGPT, N_EMBD, BLOCK_SIZE, N_HEAD, N_LAYER
from data import get_batch, decode, vocab_size

# --- Training Hyperparameters ---
max_iters = 10_000          # How many training steps to run
eval_interval = 100       # How often to print the loss
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Training on {device}...")

# 1. Initialize the Model
model = NanoGPT(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 2. The Training Loop
for iter in range(max_iters):

    # Every once in a while, evaluate the loss on the validation set
    if iter % eval_interval == 0:
        model.eval()
        with torch.no_grad():
            xb, yb = get_batch('val')
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            print(f"Step {iter}: Validation Loss = {loss.item():.4f}")
        model.train()

    # Sample a batch of data
    xb, yb = get_batch('train')
    xb, yb = xb.to(device), yb.to(device)

    # Evaluate the loss
    logits, loss = model(xb, yb)

    # Backpropagation (The actual learning part)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"Final Loss: {loss.item():.4f}")

# 3. Test the model (Generate some text to see what it learned!)
print("\n--- Generating Sample Text ---")
model.eval()
# Start with a blank token
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_chars = []

with torch.no_grad():
    for _ in range(200):  # Generate 200 characters
        logits, _ = model(context)
        # Focus only on the very last time step prediction
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)

        # Append to our generated list and update the context
        generated_chars.append(next_idx.item())
        context = torch.cat((context, next_idx), dim=1)

        # Crop context if it exceeds our BLOCK_SIZE
        if context.shape[1] > BLOCK_SIZE:
            context = context[:, -BLOCK_SIZE:]

print(decode(generated_chars))

# 4. Export the Weights to JSON for the Browser Engine
print("\n--- Exporting Weights for the Browser ---")
state_dict = model.state_dict()
export_dict = {}

for key, tensor in state_dict.items():
    # Convert PyTorch tensors to standard nested Python lists
    # We use .cpu().numpy().tolist() to safely extract the raw numbers
    export_dict[key] = tensor.cpu().numpy().tolist()

with open('weights.json', 'w') as f:
    json.dump(export_dict, f)

print("Successfully exported to weights.json!")
