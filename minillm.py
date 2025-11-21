"""
minilm.py — Teaching Edition, Fully Configurable Hyperparameters.

This script demonstrates:
1. Loading the Salesforce/WikiText dataset (wikitext-2)
2. A character-level tokenizer
3. A GPT-style architecture with configurable layers, heads, MLP ratio, etc.
4. Training loop with evaluation interval
5. Text generation

Run training:
    python minillm.py --train --steps 500 --embed_dim 256 --num_layers 4

Run generation:
    python minillm.py --generate --prompt "Hello world:"

Dependencies:
    pip install torch datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import argparse
from pathlib import Path

# ---------------------------------------------------------
# DATA LOADING — uses HuggingFace Wikitext
# ---------------------------------------------------------
def load_wikitext(split="train"):
    """
    Load Salesforce/WikiText (wikitext-2-v1). Falls back if offline.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-v1")
        text = "\n".join(ds[split]["text"])
        print(f"[INFO] Loaded WikiText ({split}) with {len(text)} characters.")
        return text
    except Exception:
        print("[WARN] Could not load WikiText from HuggingFace; using fallback text.")
        return (
            "This is fallback sample text used only when the real dataset "
            "cannot be downloaded.\n"
        )

# ---------------------------------------------------------
# CHARACTER TOKENIZER
# ---------------------------------------------------------
class CharTokenizer:
    """
    Simple transparent character-level tokenizer.
    
    This tokenizer converts text into sequences of integers (tokens) and back,
    operating at the character level rather than word or subword level.
    
    Example:
        text = "hello world"
        tokenizer = CharTokenizer(text)
        encoded = tokenizer.encode("hello")  # Returns list of integers
        decoded = tokenizer.decode(encoded)  # Returns "hello"
    """
    
    def __init__(self, text):
        """
        Initialize the tokenizer by building a vocabulary from the input text.
        
        Args:
            text (str): The corpus used to build the character vocabulary.
                        All unique characters in this text will form the vocabulary.
        
        Process:
            1. Extract all unique characters from the text
            2. Sort them alphabetically for consistent ordering
            3. Create bidirectional mappings between characters and integers
        """
        # Extract unique characters and sort them alphabetically
        # Example: "hello" -> ['e', 'h', 'l', 'o']
        chars = sorted(set(text))
        
        # Create "string to integer" mapping: character -> unique integer ID
        # Example: {'e': 0, 'h': 1, 'l': 2, 'o': 3}
        # This allows us to convert characters to tokens (integers)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        
        # Create "integer to string" mapping: integer ID -> character
        # Example: {0: 'e', 1: 'h', 2: 'l', 3: 'o'}
        # This allows us to convert tokens back to readable text
        self.itos = {i: ch for ch, i in self.stoi.items()}
        
        # Store the size of our vocabulary (number of unique characters)
        # This is needed when initializing the embedding layer in the model
        self.vocab_size = len(self.stoi)

    def encode(self, s):
        """
        Convert a string into a list of integer tokens.
        
        Args:
            s (str): The text string to encode
        
        Returns:
            list[int]: List of integer tokens representing the input string
        
        Process:
            - Iterates through each character in the input string
            - Looks up the character's integer ID in the stoi dictionary
            - Skips any characters not in the vocabulary (silently filters them)
        
        Example:
            If stoi = {'h': 1, 'e': 2, 'l': 3, 'o': 4}
            encode("hello") -> [1, 2, 3, 3, 4]
        """
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        """
        Convert a list of integer tokens back into a readable string.
        
        Args:
            ids (list[int]): List of integer tokens to decode
        
        Returns:
            str: The reconstructed text string
        
        Process:
            - Iterates through each integer token
            - Looks up the corresponding character in the itos dictionary
            - Joins all characters together into a single string
        
        Example:
            If itos = {1: 'h', 2: 'e', 3: 'l', 4: 'o'}
            decode([1, 2, 3, 3, 4]) -> "hello"
        """
        return "".join(self.itos[i] for i in ids)

# ---------------------------------------------------------
# TRANSFORMER MODULES
# ---------------------------------------------------------
class SelfAttention(nn.Module):
    """
    Multi-head causal self-attention mechanism.
    
    This is the core component of the transformer architecture. It allows each position
    in a sequence to "attend to" (gather information from) all previous positions,
    computing context-aware representations.
    
    Key features:
        - Multi-head: Runs multiple attention operations in parallel
        - Causal: Each position can only attend to itself and previous positions
        - Self-attention: Each token attends to tokens within the same sequence
    
    Architecture:
        Input (B, T, C) -> Q, K, V projections -> Multi-head attention -> 
        Concatenate heads -> Output projection -> Output (B, T, C)
    
    Args:
        dim (int): Embedding dimension (must be divisible by num_heads)
        num_heads (int): Number of parallel attention heads
        block_size (int): Maximum sequence length for the causal mask
    """
    
    def __init__(self, dim, num_heads, block_size):
        super().__init__()
        
        # Verify that we can evenly split the embedding dimension across heads
        # Example: dim=128, num_heads=4 -> each head gets 32 dimensions
        assert dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        # Calculate dimension per attention head
        # Splitting allows each head to learn different attention patterns
        self.head_dim = dim // num_heads

        # Linear projections to create Query, Key, and Value representations
        # These are learnable transformations that prepare the input for attention
        # - Query: "What am I looking for?"
        # - Key: "What do I contain/offer?"
        # - Value: "What information should be passed forward?"
        self.query = nn.Linear(dim, dim)  # Projects to same dimension
        self.key   = nn.Linear(dim, dim)  # Will be split across heads
        self.value = nn.Linear(dim, dim)  # Contains actual information to aggregate

        # Final output projection to combine information from all heads
        # Maps concatenated head outputs back to embedding dimension
        self.proj  = nn.Linear(dim, dim)

        # Create causal mask: a lower triangular matrix
        # This ensures each position can only attend to previous positions
        # Example for block_size=4:
        #   [[1, 0, 0, 0],     Position 0 can only see itself
        #    [1, 1, 0, 0],     Position 1 can see positions 0-1
        #    [1, 1, 1, 0],     Position 2 can see positions 0-2
        #    [1, 1, 1, 1]]     Position 3 can see positions 0-3
        # register_buffer ensures this is saved with the model but not trained
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        """
        Apply multi-head causal self-attention to the input sequence.
        
        Args:
            x: Input tensor of shape (B, T, C) where:
               - B = batch size (number of sequences processed in parallel)
               - T = sequence length (number of tokens)
               - C = embedding dimension (channels/features per token)
        
        Returns:
            Output tensor of shape (B, T, C) with context-aware representations
        
        Process:
            1. Project input to Q, K, V
            2. Split into multiple heads
            3. Compute attention scores (Q·K^T)
            4. Apply causal mask and softmax
            5. Apply attention to values (weights·V)
            6. Concatenate heads and project
        """
        B, T, C = x.shape

        # Step 1: Project input through Q, K, V linear layers
        # Each results in (B, T, C) - same shape as input
        q = self.query(x)  # (B, T, C) - what each position is looking for
        k = self.key(x)    # (B, T, C) - what each position offers as context
        v = self.value(x)  # (B, T, C) - the actual information to aggregate

        # Step 2: Reshape to separate attention heads
        # Split embedding dimension (C) across heads
        # Process: (B, T, C) -> (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        # 
        # Example: B=2, T=10, C=128, num_heads=4
        #   (2, 10, 128) -> (2, 10, 4, 32) -> (2, 4, 10, 32)
        # 
        # The transpose moves heads to dimension 1 so we can process them in parallel
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, T, head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, T, head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, T, head_dim)

        # Step 3: Compute attention scores using scaled dot-product attention
        # Formula: scores = (Q @ K^T) / sqrt(head_dim)
        # 
        # @ is matrix multiplication: (B, heads, T, head_dim) @ (B, heads, head_dim, T)
        # Result: (B, heads, T, T) - attention score from each position to every other
        # 
        # Scaling by sqrt(head_dim) prevents dot products from becoming too large
        # (which would cause softmax to have very small gradients)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B, heads, T, T)
        
        # Step 4: Apply causal mask
        # Set future positions to -inf so they become 0 after softmax
        # Only use the top-left TxT portion of the mask (in case T < block_size)
        # 
        # Example: If scores[i, j] represents attention from position i to position j,
        #          we set scores[i, j] = -inf whenever j > i (future positions)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))

        # Step 5: Convert scores to attention weights using softmax
        # Softmax normalizes scores to probabilities that sum to 1
        # The -inf values become 0, effectively blocking attention to future tokens
        weights = F.softmax(scores, dim=-1)  # (B, heads, T, T)
        
        # Step 6: Apply attention weights to values
        # This is a weighted sum: each position gets a mixture of value vectors
        # Matrix multiply: (B, heads, T, T) @ (B, heads, T, head_dim)
        # Result: (B, heads, T, head_dim) - attended information for each position
        out = weights @ v  # (B, heads, T, head_dim)
        
        # Step 7: Concatenate all heads back together
        # Transpose to move sequence length back to dimension 1: (B, T, heads, head_dim)
        # Then reshape to merge heads: (B, T, heads * head_dim) = (B, T, C)
        # 
        # .contiguous() ensures memory layout is compatible with view()
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Step 8: Apply final output projection
        # This allows the model to learn how to best combine information from all heads
        return self.proj(out)  # (B, T, C)


# === ARCHITECTURAL DESIGN NOTES ===
#
# 1. Pre-norm vs Post-norm:
#    - Pre-norm (used here): LayerNorm(x) -> Sublayer -> Add
#    - Post-norm: Sublayer(x) -> LayerNorm -> Add
#    - Pre-norm is generally more stable and easier to train
#
# 2. Why MLP after Attention?
#    - Attention mixes information across positions (communication)
#    - MLP refines each position independently (computation)
#    - This separation of concerns is key to transformer success
#
# 3. Why the expansion in MLP?
#    - Projecting to higher dimensions gives the model more "room"
#      to learn complex non-linear transformations
#    - The bottleneck structure (expand -> contract) is a form of
#      feature engineering in the learned representation space
#
# 4. Typical hyperparameters:
#    - mlp_ratio = 4 (standard in most transformers)
#    - num_heads = 8, 12, or 16 for larger models
#    - Multiple blocks stacked (6, 12, 24, or more layers)
class TransformerBlock(nn.Module):
    """
    Single transformer block combining multi-head attention and feedforward layers.
    
    This is the fundamental building block of transformer models. Modern transformers
    stack many of these blocks to create deep networks that can learn complex patterns.
    
    Architecture (pre-norm variant):
        Input -> LayerNorm -> Self-Attention -> Add (residual) ->
        LayerNorm -> MLP -> Add (residual) -> Output
    
    Key components:
        1. Self-attention: Allows tokens to communicate and share information
        2. MLP (feedforward): Processes each token independently
        3. Layer normalization: Stabilizes training by normalizing activations
        4. Residual connections: Enables gradient flow in deep networks
    
    Args:
        dim (int): Embedding dimension (size of token representations)
        num_heads (int): Number of parallel attention heads
        mlp_ratio (int): Expansion factor for MLP hidden layer (typically 4)
        block_size (int): Maximum sequence length for causal masking
    
    Note:
        This uses "pre-norm" architecture where LayerNorm is applied BEFORE
        the sublayers, which tends to be more stable than "post-norm".
    """
    
    def __init__(self, dim, num_heads, mlp_ratio, block_size):
        super().__init__()
        
        # Layer Normalization before attention
        # Normalizes activations to have mean=0 and variance=1 across the embedding dimension
        # This stabilizes training and allows higher learning rates
        self.ln1 = nn.LayerNorm(dim)
        
        # Multi-head self-attention mechanism
        # Allows each token to gather contextual information from other tokens
        # This is where "communication" between tokens happens
        self.attn = SelfAttention(dim, num_heads, block_size)
        
        # Layer Normalization before MLP
        # Applied after attention and before the feedforward network
        self.ln2 = nn.LayerNorm(dim)
        
        # MLP (Multi-Layer Perceptron) / Feedforward Network
        # This is a position-wise feedforward network that processes each token independently
        # 
        # Structure: Linear -> ReLU -> Linear
        # - First layer expands: dim -> (mlp_ratio * dim)
        #   Example: 128 -> 512 if mlp_ratio=4
        # - ReLU adds non-linearity (enables learning complex patterns)
        # - Second layer contracts back: (mlp_ratio * dim) -> dim
        #   Example: 512 -> 128
        # 
        # The expansion and contraction (bottleneck) allows the model to:
        # 1. Project into a higher-dimensional space where patterns are easier to separate
        # 2. Apply non-linear transformations
        # 3. Project back to the original dimension for residual addition
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),    # Expand dimension
            nn.ReLU(),                           # Non-linear activation
            nn.Linear(mlp_ratio * dim, dim),    # Contract back to original dimension
        )

    def forward(self, x):
        """
        Process input through attention and feedforward layers with residual connections.
        
        Args:
            x: Input tensor of shape (B, T, C) where:
               - B = batch size
               - T = sequence length
               - C = embedding dimension
        
        Returns:
            Output tensor of shape (B, T, C) with enriched representations
        
        Process:
            1. Attention block: x = x + Attention(LayerNorm(x))
            2. MLP block: x = x + MLP(LayerNorm(x))
        
        Key concepts:
            - Pre-normalization: LayerNorm applied BEFORE each sublayer
            - Residual connections: Original input is added back after transformation
            - Sequential processing: Attention first, then MLP
        """
        
        # === ATTENTION BLOCK ===
        # Step 1: Normalize the input
        # LayerNorm helps stabilize training by ensuring activations are well-scaled
        normalized = self.ln1(x)  # (B, T, C)
        
        # Step 2: Apply self-attention
        # Tokens exchange information and create context-aware representations
        # Each token can "look at" previous tokens and aggregate their information
        attn_output = self.attn(normalized)  # (B, T, C)
        
        # Step 3: Add residual connection
        # Formula: x = x + f(x)
        # 
        # Why residual connections matter:
        # 1. Gradient flow: Gradients can flow directly through the addition
        #    during backpropagation, preventing vanishing gradients
        # 2. Identity mapping: The network can learn to skip this layer if needed
        #    by setting attn_output close to zero
        # 3. Feature preservation: Original information is preserved and combined
        #    with new attended information
        x = x + attn_output  # (B, T, C)
        
        # === MLP BLOCK ===
        # Step 4: Normalize again before MLP
        # Each sublayer gets normalized inputs for stability
        normalized = self.ln2(x)  # (B, T, C)
        
        # Step 5: Apply position-wise feedforward network
        # Unlike attention which mixes information across positions,
        # the MLP processes each position (token) independently
        # 
        # This allows the model to:
        # - Transform and refine the representations
        # - Apply non-linear combinations of features
        # - Learn position-specific patterns
        # 
        # The "position-wise" means the same MLP is applied to each token,
        # but tokens don't interact with each other in this step
        mlp_output = self.mlp(normalized)  # (B, T, C)
        
        # Step 6: Add residual connection
        # Again, we add the input to the output for gradient flow and stability
        # The final output combines:
        # - Original input features
        # - Context from attention
        # - Refined features from MLP
        x = x + mlp_output  # (B, T, C)
        
        return x

# === ARCHITECTURAL DESIGN NOTES ===
#
# 1. Why decoder-only (GPT) vs encoder-decoder (BERT)?
#    - GPT is autoregressive: predicts next token given previous tokens
#    - Perfect for text generation (stories, code, completions)
#    - BERT is bidirectional: better for understanding (classification, Q&A)
#
# 2. Token + Position Embeddings:
#    - Summing (used here) vs concatenating: both work, summing is more common
#    - Alternative: Sinusoidal embeddings (fixed, not learned)
#    - Alternative: Rotary embeddings (RoPE) used in modern LLMs
#
# 3. Why LayerNorm after all blocks?
#    - Stabilizes the final representations
#    - Ensures lm_head receives consistent input distributions
#    - Improves training stability and convergence
#
# 4. Weight tying (not implemented here):
#    - Some models tie token_emb and lm_head weights
#    - Reduces parameters and can improve performance
#    - Trade-off: less flexibility in output projection
#
# 5. Scaling laws:
#    - Larger vocab_size: More tokens, but each needs embedding storage
#    - Larger embed_dim: More capacity, but quadratic attention cost
#    - More num_layers: Better performance, but slower inference
#    - Larger block_size: Longer context, but quadratic memory usage
class MiniGPT(nn.Module):
    """
    GPT-style autoregressive language model with configurable architecture.
    
    This model follows the GPT (Generative Pre-trained Transformer) architecture,
    which uses a decoder-only transformer to predict the next token in a sequence.
    
    High-level architecture:
        Input tokens -> Token embeddings + Position embeddings ->
        Stack of Transformer blocks -> Layer normalization ->
        Linear projection to vocabulary -> Output logits
    
    Key features:
        - Character-level language modeling (can be adapted for other tokenizations)
        - Causal (autoregressive) attention: can only see previous tokens
        - Configurable depth (num_layers), width (embed_dim), and attention (num_heads)
        - Generates text by iteratively predicting the next token
    
    Args:
        vocab_size (int): Size of vocabulary (number of unique tokens)
        embed_dim (int): Dimension of token embeddings (model width)
        num_heads (int): Number of attention heads in each transformer block
        num_layers (int): Number of transformer blocks to stack (model depth)
        mlp_ratio (int): MLP hidden layer expansion factor (typically 4)
        block_size (int): Maximum sequence length (context window)
    
    Example:
        model = MiniGPT(vocab_size=100, embed_dim=256, num_heads=8, 
                        num_layers=6, mlp_ratio=4, block_size=128)
        
        # Input: batch of token sequences
        idx = torch.randint(0, 100, (4, 50))  # batch_size=4, seq_len=50
        logits = model(idx)  # Output: (4, 50, 100) - predictions for each position
    """
    
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, mlp_ratio, block_size):
        super().__init__()

        # Store maximum sequence length for validation
        # This is the context window - the maximum number of previous tokens
        # the model can "see" when making predictions
        self.block_size = block_size

        # === EMBEDDING LAYERS ===
        
        # Token embedding: Maps each token ID to a learned vector representation
        # Shape: (vocab_size, embed_dim)
        # 
        # Example: If vocab_size=100 and embed_dim=256
        #   Token ID 42 -> 256-dimensional learned vector
        # 
        # These embeddings are learned during training to capture semantic meaning
        # Similar tokens (e.g., 'cat' and 'dog') will have similar embeddings
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        
        # Position embedding: Maps each position in the sequence to a learned vector
        # Shape: (block_size, embed_dim)
        # 
        # Since transformers have no inherent notion of token order (unlike RNNs),
        # we need to explicitly encode positional information
        # 
        # Example: Position 0, 1, 2, ... each get their own learned vector
        # This allows the model to learn that nearby positions have related meanings
        # and that position matters (e.g., subject usually comes before verb)
        self.pos_emb = nn.Embedding(block_size, embed_dim)

        # === TRANSFORMER BLOCKS ===
        
        # Stack of transformer blocks that process the embeddings
        # Each block contains self-attention and MLP layers
        # 
        # ModuleList ensures PyTorch properly registers these as model parameters
        # and includes them in .parameters(), .to(device), etc.
        # 
        # Depth matters: More layers = more capacity to learn complex patterns
        # - Shallow models (2-4 layers): Simple patterns, limited context
        # - Medium models (6-12 layers): Standard for many tasks
        # - Deep models (24+ layers): Complex reasoning, used in large LLMs
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, block_size)
            for _ in range(num_layers)
        ])

        # === OUTPUT LAYERS ===
        
        # Final layer normalization
        # Applied after all transformer blocks to stabilize the final representations
        # Ensures the input to the language modeling head is well-scaled
        self.ln_f = nn.LayerNorm(embed_dim)
        
        # Language modeling head: Projects from embedding space to vocabulary logits
        # Shape: (embed_dim, vocab_size)
        # 
        # For each position, this produces a score (logit) for every possible next token
        # Higher logits = more confident predictions
        # 
        # Example: embed_dim=256, vocab_size=100
        #   256-dimensional representation -> 100 logits (one per vocabulary token)
        # 
        # These logits are converted to probabilities via softmax during generation
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        """
        Forward pass: Convert token indices to next-token prediction logits.
        
        Args:
            idx: Input tensor of token indices, shape (B, T) where:
                 - B = batch size (number of sequences processed in parallel)
                 - T = sequence length (number of tokens in each sequence)
                 - Values are integers in range [0, vocab_size)
        
        Returns:
            logits: Prediction scores for next token, shape (B, T, vocab_size)
                    For each position, we get a score for every possible next token
        
        Process:
            1. Embed tokens and positions
            2. Pass through transformer blocks
            3. Normalize and project to vocabulary
        
        Example:
            Input: idx = [[5, 12, 7, 3]]  # One sequence of 4 tokens
            Output: logits of shape (1, 4, vocab_size)
                    logits[0, 0, :] = predictions after seeing token 5
                    logits[0, 1, :] = predictions after seeing tokens 5, 12
                    logits[0, 2, :] = predictions after seeing tokens 5, 12, 7
                    logits[0, 3, :] = predictions after seeing tokens 5, 12, 7, 3
        """
        
        # Extract dimensions from input
        B, T = idx.shape  # Batch size and sequence length
        
        # Validate sequence length
        # If T > block_size, the positional embeddings won't be defined
        # and the attention masks will be too small
        assert T <= self.block_size, "Input sequence too long."

        # === STEP 1: CREATE EMBEDDINGS ===
        
        # Get token embeddings for each token in the input
        # idx: (B, T) -> tok: (B, T, embed_dim)
        # Each token ID is replaced with its learned embedding vector
        tok = self.token_emb(idx)
        
        # Get position embeddings for positions 0, 1, 2, ..., T-1
        # torch.arange(T) creates [0, 1, 2, ..., T-1]
        # pos: (T, embed_dim)
        # 
        # We use the same position embeddings for all sequences in the batch
        # (broadcasting will handle this automatically when we add)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        
        # Combine token and position embeddings
        # tok: (B, T, embed_dim) + pos: (T, embed_dim) -> x: (B, T, embed_dim)
        # 
        # Broadcasting adds the position embeddings to each sequence in the batch
        # This gives each token a representation that includes:
        # 1. What the token is (from token_emb)
        # 2. Where it is in the sequence (from pos_emb)
        x = tok + pos

        # === STEP 2: PROCESS THROUGH TRANSFORMER BLOCKS ===
        
        # Pass through each transformer block sequentially
        # Each block refines the representations using attention and MLP
        # 
        # Information flows and is transformed:
        # - Early layers: Learn basic patterns and local dependencies
        # - Middle layers: Combine information from multiple positions
        # - Late layers: Make high-level predictions and abstractions
        # 
        # x shape remains (B, T, embed_dim) throughout
        for layer in self.layers:
            x = layer(x)  # Each layer returns (B, T, embed_dim)

        # === STEP 3: FINAL NORMALIZATION ===
        
        # Apply final layer normalization
        # Stabilizes the representations before the output projection
        # Ensures the lm_head receives well-scaled inputs
        x = self.ln_f(x)  # (B, T, embed_dim)

        # === STEP 4: PROJECT TO VOCABULARY ===
        
        # Project from embedding space to vocabulary logits
        # x: (B, T, embed_dim) -> logits: (B, T, vocab_size)
        # 
        # For each position t in each sequence:
        #   logits[b, t, :] = scores for what token should come NEXT
        # 
        # These are raw scores (logits), not probabilities
        # During training: Cross-entropy loss handles the conversion
        # During generation: We apply softmax to get probabilities
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        return logits

# ---------------------------------------------------------
# TRAINING
# ---------------------------------------------------------

# === WHY THIS DESIGN? ===
#
# 1. Random sampling:
#    - Exposes model to different parts of the dataset
#    - Prevents overfitting to a specific order
#    - Each epoch sees different sequence combinations
#
# 2. Multiple predictions per sequence:
#    - We get block_size predictions from each sequence
#    - Efficient use of data (one sequence -> many examples)
#    - The model learns at every position simultaneously
#
# 3. Shifting by 1 (y = x shifted):
#    - This is autoregressive prediction
#    - Model learns: "given this context, what comes next?"
#    - Foundation of language modeling and text generation
#
# 4. Why batch_size matters:
#    - Larger batches: More stable gradients, faster training
#    - Smaller batches: More noise, can help generalization
#    - Trade-off with GPU memory (larger batches need more memory)
#
# 5. Why block_size matters:
#    - Larger context: Model can use more information
#    - Smaller context: Faster training, less memory
#    - Must match the block_size used in the model architecture

# === DETAILED EXAMPLE ===
#
# Suppose we have data = [10, 20, 30, 40, 50, 60, 70, 80, 90]
# block_size = 3, batch_size = 2
#
# Step 1: Sample random starting positions
#   Possible range: [0, 9 - 3 - 1] = [0, 5]
#   Suppose we get ix = [1, 4]
#
# Step 2: Extract x (inputs)
#   For i=1: data[1:4] = [20, 30, 40]
#   For i=4: data[4:7] = [50, 60, 70]
#   x = [[20, 30, 40],
#        [50, 60, 70]]
#
# Step 3: Extract y (targets)
#   For i=1: data[2:5] = [30, 40, 50]
#   For i=4: data[5:8] = [60, 70, 80]
#   y = [[30, 40, 50],
#        [60, 70, 80]]
#
# Training examples created:
#   Batch 0:
#     - Input [20]     -> Predict 30 ✓
#     - Input [20, 30] -> Predict 40 ✓
#     - Input [20, 30, 40] -> Predict 50 ✓
#   
#   Batch 1:
#     - Input [50]     -> Predict 60 ✓
#     - Input [50, 60] -> Predict 70 ✓
#     - Input [50, 60, 70] -> Predict 80 ✓
#
# Total: 6 prediction tasks from 2 sequences!
def get_batch(data, block_size, batch_size):
    """
    Sample random batches of sequences for supervised next-token prediction.
    
    This function creates training batches by randomly sampling subsequences
    from the dataset. Each batch contains input sequences (x) and their
    corresponding targets (y), where targets are simply the inputs shifted
    by one position to the right.
    
    This is the core of self-supervised language modeling: the model learns
    to predict the next token given previous tokens, without any labeled data.
    
    Args:
        data: Tensor of token indices, shape (total_tokens,)
              The entire dataset encoded as integers
              Example: tensor([5, 12, 7, 3, 8, 19, ...])
        
        block_size: Maximum sequence length (context window)
                    How many tokens the model can see at once
                    Example: 128 means model sees up to 128 previous tokens
        
        batch_size: Number of sequences to sample
                    How many independent sequences to process in parallel
                    Example: 32 means we'll create 32 different sequences
    
    Returns:
        x: Input sequences, shape (batch_size, block_size)
           The context tokens that the model will use for prediction
        
        y: Target sequences, shape (batch_size, block_size)
           The tokens that the model should predict (x shifted by 1)
    
    Example:
        If data = [1, 2, 3, 4, 5, 6, 7, 8, 9] and block_size = 4:
        
        One possible sample starting at index 2:
            x = [3, 4, 5, 6]  (indices 2:6)
            y = [4, 5, 6, 7]  (indices 3:7)
        
        The model learns:
            - Given [3], predict 4
            - Given [3, 4], predict 5
            - Given [3, 4, 5], predict 6
            - Given [3, 4, 5, 6], predict 7
    
    Training objective:
        At each position t in the sequence, the model predicts token y[t]
        based on all previous tokens x[0:t+1]. This creates multiple training
        examples from a single sequence.
    """
    
    # === STEP 1: SAMPLE RANDOM STARTING POSITIONS ===
    
    # Generate random starting indices for each sequence in the batch
    # 
    # Range: [0, len(data) - block_size - 1]
    # - We need block_size tokens for x
    # - We need 1 more token for y (the target after the last x token)
    # - So we need at least block_size + 1 tokens available
    # 
    # Example: If data has 1000 tokens and block_size=128
    #   ix could be [42, 356, 721, 89, ...] (batch_size random numbers)
    #   Each number is a valid starting position where we can extract
    #   128 tokens for x and 128 tokens for y
    ix = torch.randint(
        0,                              # Minimum starting position
        len(data) - block_size - 1,    # Maximum starting position
        (batch_size,)                   # Number of random positions to generate
    )
    
    # === STEP 2: EXTRACT INPUT SEQUENCES (X) ===
    
    # For each starting position, extract block_size consecutive tokens
    # 
    # data[i:i+block_size] extracts tokens from position i to i+block_size-1
    # 
    # Example: If i=100 and block_size=4
    #   data[100:104] gets tokens at positions [100, 101, 102, 103]
    # 
    # List comprehension creates batch_size sequences
    # torch.stack combines them into a 2D tensor
    # 
    # Result shape: (batch_size, block_size)
    # Each row is one sequence of input tokens
    x = torch.stack([data[i:i+block_size] for i in ix])
    
    # === STEP 3: EXTRACT TARGET SEQUENCES (Y) ===
    
    # For each starting position, extract block_size tokens starting ONE position later
    # 
    # data[i+1:i+block_size+1] extracts tokens from position i+1 to i+block_size
    # 
    # Example: If i=100 and block_size=4
    #   data[101:105] gets tokens at positions [101, 102, 103, 104]
    # 
    # This creates the "next token" targets for training
    # y is exactly x shifted by one position to the right
    # 
    # Alignment:
    #   x[b, t] is the input at position t
    #   y[b, t] is the target (what should come after x[b, t])
    # 
    # Result shape: (batch_size, block_size)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    return x, y


# === UNDERSTANDING TRAINING DYNAMICS ===
#
# What happens during training?
# 
# 1. Initially (untrained model):
#    - Loss is high (around ln(vocab_size) ≈ 4-5 for typical vocabs)
#    - Predictions are essentially random
#    - Model outputs roughly uniform probabilities
#
# 2. Early training:
#    - Model learns token frequencies (common tokens get higher probabilities)
#    - Loss drops quickly as model learns basic statistics
#    - Predictions become slightly better than random
#
# 3. Mid training:
#    - Model learns local patterns (bigrams, trigrams)
#    - Begins to capture simple syntactic rules
#    - Loss decreases more slowly
#
# 4. Late training:
#    - Model learns longer-range dependencies
#    - Captures semantic relationships and context
#    - Loss plateaus as model approaches data complexity limit
#
# 5. Convergence:
#    - Loss stabilizes around some minimum value
#    - Further training gives diminishing returns
#    - Risk of overfitting if trained too long


# === MONITORING TRAINING ===
#
# Good signs:
# - Loss steadily decreases
# - No sudden spikes or instability
# - Training loss and validation loss track together
#
# Warning signs:
# - Loss increases or oscillates wildly -> learning rate too high
# - Loss plateaus immediately -> learning rate too low
# - Loss becomes NaN -> gradient explosion (need gradient clipping)
# - Training loss << validation loss -> overfitting
#
# Typical loss values (character-level):
# - Start: ~4.5 (random guessing for 100-char vocab)
# - After 100 steps: ~3.0-3.5
# - After 1000 steps: ~2.0-2.5
# - Well-trained: ~1.5-2.0
# - These depend heavily on vocab size, model size, and data complexity


# === HYPERPARAMETER GUIDANCE ===
#
# Learning rate (lr):
# - Too high: Training unstable, loss diverges
# - Too low: Training very slow, may get stuck
# - Sweet spot: 1e-4 to 1e-3 for Adam/AdamW
# - Tip: Start high, decrease if unstable
#
# Batch size:
# - Small (8-16): Noisy gradients, slower but can generalize better
# - Medium (32-64): Good balance for most tasks
# - Large (128+): Stable gradients, faster, needs more memory
# - Tip: Use the largest batch size that fits in memory
#
# Training steps:
# - Too few: Underfitting, model hasn't learned enough
# - Too many: Overfitting, model memorizes training data
# - Tip: Monitor validation loss, stop when it stops improving
#
# Block size:
# - Larger: More context, better long-range predictions
# - Smaller: Faster training, less memory
# - Tip: Match to your typical sequence lengths
def train(model, data, steps, block_size, batch_size, lr, eval_interval):
    """
    Train the language model using next-token prediction.
    
    This function implements the complete training loop for the GPT model,
    using self-supervised learning where the model learns to predict the
    next token given previous tokens.
    
    Training process:
        1. Sample a batch of sequences from the data
        2. Forward pass: compute predictions (logits)
        3. Compute loss: compare predictions to actual next tokens
        4. Backward pass: compute gradients
        5. Update weights: adjust parameters to reduce loss
        6. Repeat for specified number of steps
    
    Args:
        model: The MiniGPT model to train
        data: Tensor of encoded tokens, shape (total_tokens,)
              The complete dataset as a 1D tensor of integer token IDs
        steps: Number of training iterations (batches to process)
               More steps = more training, but takes longer
        block_size: Sequence length (context window)
                    Must match the model's block_size
        batch_size: Number of sequences per batch
                    Larger = faster but needs more memory
        lr: Learning rate for the optimizer
            Controls how much to adjust weights each step
            Typical values: 1e-4 to 1e-3
        eval_interval: How often to print loss (in steps)
                       Example: 50 means print every 50 steps
    
    Side effects:
        - Modifies model parameters in-place
        - Saves trained model to "minilm.pt" file
        - Prints training progress to console
    
    Note:
        This is a simplified training loop. Production code would add:
        - Validation set evaluation
        - Learning rate scheduling
        - Gradient clipping
        - Early stopping
        - Checkpointing best models
    """
    
    # === SETUP: INITIALIZE OPTIMIZER ===
    
    # AdamW optimizer: Adam with weight decay (L2 regularization)
    # 
    # Adam advantages:
    # - Adaptive learning rates per parameter
    # - Momentum for faster convergence
    # - Handles sparse gradients well
    # 
    # AdamW improvement:
    # - Decouples weight decay from gradient updates
    # - Better generalization than standard Adam
    # 
    # model.parameters() returns all trainable weights in the model
    # (embeddings, attention layers, MLPs, etc.)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Set model to training mode
    # This enables:
    # - Dropout (if we had any)
    # - Batch normalization training behavior
    # - Gradient computation for all parameters
    # 
    # Important: Always call model.train() before training
    # and model.eval() before inference/validation
    model.train()

    # === TRAINING LOOP ===
    
    
    for step in range(steps):
        # === STEP 1: GET TRAINING BATCH ===
        
        # Sample a random batch of sequences from the dataset
        # x: input sequences, shape (batch_size, block_size)
        # y: target sequences (x shifted by 1), shape (batch_size, block_size)
        # 
        # Example: batch_size=32, block_size=128
        #   x: (32, 128) - 32 sequences of 128 tokens each
        #   y: (32, 128) - the next token for each position
        x, y = get_batch(data, block_size, batch_size)
        
        # === STEP 2: FORWARD PASS ===
        
        # Pass input through the model to get predictions
        # x: (batch_size, block_size) -> logits: (batch_size, block_size, vocab_size)
        # 
        # Each position gets a score (logit) for every possible next token
        # Higher logits = more confident predictions
        # 
        # Example: vocab_size=100
        #   logits[0, 5, :] = 100 scores for what should come after position 5
        #   The highest score indicates the model's best guess
        logits = model(x)
        
        # === STEP 3: COMPUTE LOSS ===
        
        # Cross-entropy loss measures how wrong the predictions are
        # 
        # Reshaping explained:
        # - logits.view(-1, logits.size(-1)):
        #   (batch_size, block_size, vocab_size) -> (batch_size*block_size, vocab_size)
        #   Flatten all positions into one long list of predictions
        #   Example: (32, 128, 100) -> (4096, 100)
        # 
        # - y.view(-1):
        #   (batch_size, block_size) -> (batch_size*block_size,)
        #   Flatten all targets into one long list
        #   Example: (32, 128) -> (4096,)
        # 
        # Why flatten?
        # - Cross-entropy expects 2D predictions: (num_samples, num_classes)
        # - We have batch_size * block_size independent predictions
        # - Each position is treated as a separate classification problem
        # 
        # Cross-entropy does:
        # 1. Apply softmax to logits to get probabilities
        # 2. Take the negative log probability of the correct token
        # 3. Average over all predictions
        # 
        # Lower loss = better predictions
        # Perfect predictions would give loss ≈ 0
        # Random guessing gives loss ≈ ln(vocab_size)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),  # Predictions: (total_positions, vocab_size)
            y.view(-1)                          # Targets: (total_positions,)
        )
        
        # === STEP 4: BACKWARD PASS ===
        
        # Zero out gradients from the previous step
        # PyTorch accumulates gradients by default, so we must clear them
        # 
        # Without this, gradients would pile up from multiple batches,
        # causing incorrect weight updates
        opt.zero_grad()
        
        # Compute gradients via backpropagation
        # This calculates ∂loss/∂param for every parameter in the model
        # 
        # The chain rule automatically propagates gradients backward
        # through all layers: output -> transformer blocks -> embeddings
        # 
        # After this call, every parameter has a .grad attribute
        # containing its gradient
        loss.backward()
        
        # === STEP 5: UPDATE WEIGHTS ===
        
        # Update model parameters using computed gradients
        # 
        # For each parameter θ:
        #   θ_new = θ_old - lr * gradient
        # (AdamW actually uses a more sophisticated update rule)
        # 
        # This is where the model actually "learns"
        # Parameters are adjusted to reduce the loss
        opt.step()
        
        # === STEP 6: LOGGING ===
        
        # Periodically print training progress
        # eval_interval controls how often we report
        # 
        # Example: eval_interval=50 means print every 50 steps
        # 
        # .item() extracts the loss value as a Python float
        # (without this, loss is a tensor which retains computation graph)
        if step % eval_interval == 0:
            print(f"[step {step}] loss = {loss.item():.4f}")
    
    # === SAVE TRAINED MODEL ===
    
    # Save the model's learned parameters to disk
    # 
    # state_dict() returns a dictionary containing all parameters:
    # {
    #   'token_emb.weight': tensor(...),
    #   'layers.0.attn.query.weight': tensor(...),
    #   ...
    # }
    # 
    # This saves only the weights, not the model architecture
    # To load later, you need to:
    # 1. Create a model with the same architecture
    # 2. Load the weights: model.load_state_dict(torch.load("minilm.pt"))
    torch.save(model.state_dict(), "minilm.pt")
    print("\n[INFO] Model saved to minilm.pt\n")


# ---------------------------------------------------------
# GENERATION
# ---------------------------------------------------------



# === UNDERSTANDING GENERATION PARAMETERS ===
#
# Temperature effects:
# 
# Temperature = 0.1 (very low):
#   "The cat sat on the mat. The cat sat on the mat. The cat..."
#   - Very repetitive
#   - Always picks most likely token
#   - Boring but grammatically correct
#
# Temperature = 0.7 (low):
#   "The cat sat on the mat and looked around curiously."
#   - Focused and coherent
#   - Some variety but stays on topic
#   - Good for factual or structured text
#
# Temperature = 1.0 (neutral):
#   "The cat explored the garden, discovering new scents."
#   - Balanced creativity and coherence
#   - Default for most applications
#
# Temperature = 1.5 (high):
#   "The cat suddenly started dancing under the moonlight!"
#   - Creative and unexpected
#   - More errors and inconsistencies
#   - Good for creative writing
#
# Temperature = 2.0 (very high):
#   "Cat? Moon! Garden spaghetti tornado yes!"
#   - Often incoherent
#   - Too random to be useful
#   - May break grammar entirely


# === TOP-K SAMPLING EFFECTS ===
#
# Top-k = 1:
#   Always picks the most likely token (greedy decoding)
#   Equivalent to temperature = 0
#   Most deterministic, most repetitive
#
# Top-k = 10:
#   Only considers 10 most likely tokens
#   Conservative, safe, coherent
#   Good for consistent output
#
# Top-k = 50 (default):
#   Good balance for most use cases
#   Enough diversity without nonsense
#
# Top-k = 200:
#   Considers many possibilities
#   More creative but riskier
#   May include unlikely words
#
# Top-k = None:
#   All tokens considered (even very unlikely ones)
#   Can produce rare words or typos
#   Often too unpredictable


# === COMBINING TEMPERATURE AND TOP-K ===
#
# Conservative (safe, coherent):
#   temperature=0.7, top_k=20
#   Use for: Technical writing, documentation
#
# Balanced (default):
#   temperature=1.0, top_k=50
#   Use for: General text generation
#
# Creative (diverse, surprising):
#   temperature=1.2, top_k=100
#   Use for: Stories, poetry, brainstorming
#
# Experimental (chaotic):
#   temperature=1.5, top_k=None
#   Use for: Exploring model capabilities


# === COMPUTATIONAL CONSIDERATIONS ===
#
# Generation is slower than training per token because:
# 1. Sequential process (can't parallelize across tokens)
# 2. Each token requires a full forward pass
# 3. Context window grows with each token (until it hits block_size)
#
# Optimization strategies:
# 1. Use smaller models for faster generation
# 2. Reduce block_size if you don't need long context
# 3. Use caching (KV-cache) to avoid recomputing attention for old tokens
# 4. Batch multiple prompts together
# 5. Use quantization or pruning for inference
#
# Speed example (typical laptop):
# - Small model (2 layers, 128 dim): ~50 tokens/second
# - Medium model (6 layers, 256 dim): ~20 tokens/second
# - Large model (12 layers, 512 dim): ~5 tokens/second
def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=1.0, top_k=50):
    """
    Generate text autoregressively by predicting one token at a time.
    
    This function implements autoregressive sampling: starting from a prompt,
    the model repeatedly predicts the next token, adds it to the sequence,
    and uses the extended sequence to predict the next token, and so on.
    
    Generation process:
        1. Encode the prompt into tokens
        2. Feed tokens through the model to get next-token predictions
        3. Sample a token from the probability distribution
        4. Append the sampled token to the sequence
        5. Repeat steps 2-4 until we've generated max_new_tokens
        6. Decode the token sequence back to text
    
    Args:
        model: The trained MiniGPT model
        tokenizer: CharTokenizer for encoding/decoding text
        prompt: Starting text to condition generation
                Example: "Once upon a time"
        max_new_tokens: Number of tokens to generate (default: 200)
                        Longer = more text but slower
        temperature: Sampling temperature (default: 1.0)
                     - Higher (e.g., 1.5): More random, creative, diverse
                     - Lower (e.g., 0.5): More deterministic, focused, repetitive
                     - 0.0: Greedy decoding (always pick most likely token)
        top_k: Number of top tokens to consider (default: 50)
               Restricts sampling to the k most likely tokens
               - Higher: More diversity
               - Lower: More conservative
               - None: Consider all tokens (can produce nonsense)
    
    Returns:
        Generated text as a string (prompt + generated tokens)
    
    Example:
        model = MiniGPT(...)
        tokenizer = CharTokenizer(data)
        
        # Conservative generation
        text = generate(model, tokenizer, "Hello", temperature=0.7, top_k=20)
        
        # Creative generation
        text = generate(model, tokenizer, "Hello", temperature=1.2, top_k=100)
    
    Note:
        This is a simple sampling strategy. Advanced methods include:
        - Nucleus (top-p) sampling
        - Beam search
        - Contrastive search
        - Repetition penalties
    """
    
    # === SETUP: PREPARE FOR GENERATION ===
    
    # Set model to evaluation mode
    # This disables:
    # - Dropout (if we had any)
    # - Batch normalization training behavior
    # - Gradient computation (saves memory)
    # 
    # Always use model.eval() during inference!
    model.eval()
    
    # Encode the prompt into token indices
    # 
    # Example: prompt = "Hello"
    # If tokenizer maps: H->5, e->12, l->7, o->3
    # tokenizer.encode("Hello") -> [5, 12, 7, 7, 3]
    # 
    # Wrap in list and convert to tensor: [[5, 12, 7, 7, 3]]
    # Shape: (1, prompt_length) - batch_size=1, sequence_length=len(prompt)
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    
    # === AUTOREGRESSIVE GENERATION LOOP ===
    
    # Generate one token at a time
    # Each iteration:
    # 1. Model predicts next token distribution
    # 2. We sample one token from that distribution
    # 3. Append it to our sequence
    # 4. Use extended sequence for next prediction
    for _ in range(max_new_tokens):
        
        # === STEP 1: PREPARE INPUT (CONTEXT WINDOW) ===
        
        # Only use the last block_size tokens as context
        # This is necessary because:
        # 1. Model's positional embeddings only go up to block_size
        # 2. Attention mask is only defined for block_size positions
        # 
        # Example: block_size=128
        # - If idx has 50 tokens: use all 50
        # - If idx has 200 tokens: use only the last 128
        # 
        # This creates a "sliding window" - as we generate, we keep
        # only the most recent context
        idx_cond = idx[:, -model.block_size:]  # (1, min(seq_len, block_size))
        
        # === STEP 2: GET MODEL PREDICTIONS ===
        
        # Forward pass through the model
        # idx_cond: (1, context_len) -> logits: (1, context_len, vocab_size)
        # 
        # We only need predictions for the LAST position (the next token)
        # [:, -1, :] extracts the last position's logits
        # Result: (1, vocab_size) - scores for each possible next token
        # 
        # Temperature scaling:
        # - Dividing by temperature adjusts the distribution's "sharpness"
        # - temp > 1: Flatter distribution (more random sampling)
        # - temp < 1: Sharper distribution (more deterministic sampling)
        # - temp = 1: Use original distribution
        # 
        # Example: Original logits = [2.0, 1.0, 0.5]
        #   temp=0.5: [4.0, 2.0, 1.0] -> sharper (picks first more often)
        #   temp=2.0: [1.0, 0.5, 0.25] -> flatter (more random)
        logits = model(idx_cond)[:, -1, :] / temperature  # (1, vocab_size)
        
        # === STEP 3: APPLY TOP-K FILTERING ===
        
        # Restrict sampling to only the k most likely tokens
        # This prevents the model from sampling very unlikely tokens
        # (which often produce nonsense or break the flow)
        if top_k:
            # Get the top k logits and their values
            # v: (1, top_k) - the k highest logit values
            # _: indices (we don't need them)
            v, _ = torch.topk(logits, top_k)
            
            # Set all logits below the k-th highest to -infinity
            # v[:, [-1]] gets the smallest value among the top k
            # 
            # Example: top_k=3, logits = [2.0, 1.5, 1.0, 0.5, 0.1]
            #   Top 3 values: [2.0, 1.5, 1.0]
            #   Threshold: 1.0
            #   After filtering: [2.0, 1.5, 1.0, -inf, -inf]
            # 
            # Tokens with -inf logits get probability 0 after softmax
            # Effectively removing them from consideration
            logits[logits < v[:, [-1]]] = -float("inf")
        
        # === STEP 4: CONVERT TO PROBABILITIES ===
        
        # Apply softmax to convert logits to a probability distribution
        # 
        # Softmax formula: prob_i = exp(logit_i) / sum(exp(logit_j))
        # 
        # Properties:
        # - All probabilities sum to 1.0
        # - Higher logits -> higher probabilities
        # - -inf logits -> 0 probability
        # 
        # Example: logits = [2.0, 1.0, -inf]
        #   probs = [0.73, 0.27, 0.00]
        probs = F.softmax(logits, dim=-1)  # (1, vocab_size)
        
        # === STEP 5: SAMPLE NEXT TOKEN ===
        
        # Sample one token from the probability distribution
        # torch.multinomial samples proportionally to probabilities
        # 
        # This is random sampling, not greedy (picking the max)
        # 
        # Example: probs = [0.6, 0.3, 0.1] for tokens [A, B, C]
        # - 60% chance of sampling A
        # - 30% chance of sampling B
        # - 10% chance of sampling C
        # 
        # Random sampling creates diversity - running generation multiple times
        # with the same prompt will produce different outputs
        # 
        # Result: (1, 1) - the index of the sampled token
        next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)
        
        # === STEP 6: APPEND TO SEQUENCE ===
        
        # Concatenate the new token to our sequence
        # idx: (1, current_length) + next_id: (1, 1) -> (1, current_length + 1)
        # 
        # Example: If idx = [[5, 12, 7]] and next_id = [[3]]
        #   Result: [[5, 12, 7, 3]]
        # 
        # This extended sequence becomes the context for the next iteration
        idx = torch.cat([idx, next_id], dim=1)
    
    # === DECODE BACK TO TEXT ===
    
    # After generating all tokens, convert indices back to text
    # 
    # idx[0]: Extract the sequence from batch dimension (1, seq_len) -> (seq_len,)
    # .tolist(): Convert tensor to Python list [5, 12, 7, 7, 3, ...]
    # tokenizer.decode(): Convert indices back to characters and join
    # 
    # Example: [5, 12, 7, 7, 3] -> "Hello"
    # 
    # The returned string includes both the original prompt and all generated text
    return tokenizer.decode(idx[0].tolist())


# ---------------------------------------------------------
# MAIN CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Task choice
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--prompt", type=str, default="Hello:")

    # Training hyperparameters (with defaults)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--mlp_ratio", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval_interval", type=int, default=50)

    args = parser.parse_args()

    # Load dataset
    text = load_wikitext("train")
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # Initialize model with configurable hyperparameters
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_ratio=args.mlp_ratio,
        block_size=args.block_size,
    )

    # TRAIN
    if args.train:
        train(
            model,
            data,
            steps=args.steps,
            block_size=args.block_size,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            eval_interval=args.eval_interval,
        )

    # GENERATE
    if args.generate:
        try:
            # Robust checkpoint loader:
            # - Strips `module.` prefix saved by DataParallel
            # - Filters tensors to only those whose shapes match the current model
            # - Prints diagnostics so you can detect hyperparameter/vocab mismatches
            # This prevents size mismatch errors when loading checkpoints created
            # with a different architecture or training wrapper.
            # Use this for safe partial loads; for full loads, recreate the model
            # with the exact hyperparameters used at save time.
            # Load checkpoint and normalize keys (handle DataParallel prefix)
            state = torch.load("minilm.pt", map_location="cpu")
            if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
                state = {k.replace("module.", ""): v for k, v in state.items()}

            model_state = model.state_dict()

            # Keep only tensors that both exist in the model and match shape
            filtered_state = {}
            for k, v in state.items():
                if k in model_state and v.size() == model_state[k].size():
                    filtered_state[k] = v

            loaded_keys = set(filtered_state.keys())
            ckpt_keys = set(state.keys())
            model_keys = set(model_state.keys())

            print(f"[INFO] checkpoint keys: {len(ckpt_keys)}")
            print(f"[INFO] model keys: {len(model_keys)}")
            print(f"[INFO] loaded matching keys: {len(loaded_keys)}")

            missing_keys = model_keys - loaded_keys
            unexpected_keys = ckpt_keys - model_keys
            if missing_keys:
                print(f"[WARN] {len(missing_keys)} model parameters were not found in the checkpoint; they will keep their initialized values.")
            if unexpected_keys:
                print(f"[WARN] {len(unexpected_keys)} unexpected keys were present in the checkpoint.")

            # Update model state with matching tensors and load
            model_state.update(filtered_state)
            model.load_state_dict(model_state)
            print(f"[INFO] Loaded {len(loaded_keys)} tensors from minilm.pt (partial load allowed).")

        except Exception as e:
            print(f"[ERROR] Unable to load model: {e}")

        out = generate(model, tokenizer, args.prompt)
        print("\n=== GENERATED TEXT ===\n")
        print(out)
