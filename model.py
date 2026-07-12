import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Hyperparameters ---
# We keep these small so it runs easily in the browser later
BLOCK_SIZE = 64     # Context window
N_EMBD = 64         # Embedding dimension size (d_model)
N_HEAD = 4          # Number of attention heads
N_LAYER = 3         # Number of Transformer blocks
DROPOUT = 0.0       # Keeping at 0 for deterministic browser porting


class Head(nn.Module):
    """ One head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        # Linear projections for Query, Key, and Value
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)

        # tril is a lower triangular matrix used to mask future tokens
        # (Tokens shouldn't be able to "look ahead" into the future to guess the next word)
        self.register_buffer('tril', torch.tril(
            torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        # Batch, Time (Block Size), Channels (Embedding Size)
        B, T, C = x.shape

        k = self.key(x)   # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # --- Core Self-Attention Math ---
        # 1. Compute attention scores ("affinities"): Q dot K^T
        # Transpose the last two dimensions of K to do matrix multiplication
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1)

        # 2. Scale by 1 / sqrt(head_size) to prevent variance explosion
        wei = wei * (k.shape[-1] ** -0.5)

        # 3. Apply the causal mask (replace upper triangle with -inf)
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)

        # 4. Softmax turns raw scores into beautiful percentages (0.0 to 1.0)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        # 5. Weighted aggregation of the Values
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v     # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)

        return out


class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        # Create a list of independent attention heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # Projection layer to mix the outputs of the heads back together
        self.proj = nn.Linear(N_EMBD, N_EMBD)

    def forward(self, x):
        # Run all heads in parallel and concatenate them along the channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, C)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    """ A simple linear network followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),  # Using ReLU instead of GELU to make JS porting easier later
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication (Attention) followed by computation (FFWD) """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

        # Layer Normalization
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # x + ... is the Residual (Skip) Connection.
        # It helps gradients flow backwards through deep networks.
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class NanoGPT(nn.Module):
    """ The full Decoder-Only Language Model """

    def __init__(self, vocab_size):
        super().__init__()
        # Lookup tables for the characters and their positions
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)

        # The stack of Transformer blocks
        self.blocks = nn.Sequential(
            *[Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])

        # Final Layer Norm and Linear Output Head
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # 1. Embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device))  # (T, C)

        # 2. Add embeddings together
        x = tok_emb + pos_emb  # (B, T, C)

        # 3. Pass through Transformer Blocks
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)   # (B, T, C)

        # 4. Final output layer to get predictions for next character
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # Reshape for PyTorch's CrossEntropyLoss which expects (B*T, C)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
