# MiniLM: A Character-Level GPT Implementation

## Overview

MiniLM is an educational implementation of a GPT-style transformer language model that operates at the character level. This project demonstrates the core concepts of modern large language models in a simplified, transparent codebase that's perfect for learning and experimentation.

### Project Goals

- **Educational**: Provide a clear, readable implementation of transformer architecture
- **Configurable**: Enable experimentation with hyperparameters (layers, heads, dimensions)
- **Self-contained**: Minimal dependencies, runs on CPU or GPU
- **Practical**: Train on real data (WikiText-2) and generate coherent text

## Getting Started

### Prerequisites

- Python 3.7 or higher
- PyTorch
- HuggingFace Datasets library

### Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install torch datasets
```

### Running the Project

#### Training a Model

Train a model with default hyperparameters:

```bash
python main.py --train --steps 500
```

Train with custom architecture:

```bash
python main.py --train --steps 1000 --embed_dim 256 --num_layers 4 --num_heads 8 --batch_size 64
```

#### Generating Text

After training, generate text from a prompt:

```bash
python main.py --generate --prompt "The quick brown fox"
```

### Available Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_size` | 32 | Number of sequences per training batch |
| `--block_size` | 128 | Maximum sequence length (context window) |
| `--embed_dim` | 128 | Embedding dimension for tokens |
| `--num_heads` | 4 | Number of attention heads |
| `--num_layers` | 2 | Number of transformer blocks |
| `--mlp_ratio` | 4 | MLP hidden size multiplier |
| `--learning_rate` | 3e-4 | Adam optimizer learning rate |
| `--steps` | 200 | Number of training steps |
| `--eval_interval` | 50 | Steps between loss reporting |

## Code Architecture

### 1. Data Loading (`load_wikitext`)

The script uses the WikiText-2 dataset from HuggingFace, a collection of Wikipedia articles commonly used for language modeling benchmarks. The data is concatenated into a single text string for processing. A fallback mechanism provides sample text if the dataset is unavailable.

### 2. Character Tokenizer (`CharTokenizer`)

Instead of using subword tokenization (like BPE), this implementation uses character-level tokenization:

- **Vocabulary**: Built from unique characters in the training data
- **Encoding**: Maps characters to integer indices
- **Decoding**: Converts indices back to readable text
- **Advantage**: No out-of-vocabulary tokens, completely transparent

### 3. Transformer Components

#### Self-Attention (`SelfAttention`)

Implements multi-head causal self-attention:

- Splits embedding dimension across multiple attention heads
- Computes queries, keys, and values for each position
- Uses scaled dot-product attention with causal masking
- Prevents the model from "looking ahead" during training

#### Transformer Block (`TransformerBlock`)

Combines attention and feedforward layers:

- **Pre-normalization**: LayerNorm before attention and MLP
- **Residual connections**: Adds input to output of each sublayer
- **MLP**: Two-layer feedforward network with ReLU activation

#### MiniGPT Model (`MiniGPT`)

The complete language model:

- **Token embeddings**: Learned vectors for each character
- **Position embeddings**: Learned vectors encoding sequence position
- **Stacked transformer blocks**: Configurable depth
- **Output head**: Projects to vocabulary logits

### 4. Training Loop (`train`)

Implements supervised next-token prediction:

- Samples random sequences from the dataset
- Computes cross-entropy loss between predictions and targets
- Updates model parameters using AdamW optimizer
- Periodically reports loss and saves the trained model

### 5. Text Generation (`generate`)

Autoregressive sampling:

- Starts with a prompt encoded as token indices
- Iteratively predicts the next token
- Applies temperature scaling for randomness control
- Uses top-k sampling to improve quality

## Machine Learning Theory

### Transformer Architecture

The transformer, introduced in "Attention Is All You Need" (Vaswani et al., 2017), revolutionized sequence modeling by replacing recurrence with attention mechanisms.

#### Self-Attention Mechanism

Self-attention allows each position in a sequence to attend to all other positions, computing representations based on relevance:

**Q·K^T / √d_k** produces attention scores, where:
- **Q** (queries): "what am I looking for?"
- **K** (keys): "what do I offer?"
- **V** (values): "what information do I contain?"

The scaling factor √d_k prevents gradients from becoming too small in high dimensions.

#### Multi-Head Attention

Instead of single attention, we run multiple attention operations in parallel:

- Each head learns different relationships (syntax, semantics, long-range dependencies)
- Outputs are concatenated and projected back to the embedding dimension
- Increases model capacity without dramatically increasing parameters

#### Causal Masking

For language modeling, we prevent positions from attending to future tokens:

- A triangular mask sets future positions to -∞ before softmax
- Ensures the model learns to predict based only on past context
- Critical for autoregressive generation

### Language Modeling Objective

The model learns by predicting the next token in a sequence:

**Loss = -∑ log P(token_t | token_1, ..., token_{t-1})**

This self-supervised objective requires no labeled data—the text itself provides supervision.

### Position Embeddings

Transformers have no inherent notion of sequence order, so we add positional information:

- Learned embeddings (used here) are simply looked up for each position
- Alternative: sinusoidal encodings use mathematical functions
- Added to token embeddings before processing

### Residual Connections and Layer Normalization

These architectural choices enable training deep networks:

- **Residual connections**: Allow gradients to flow directly through the network
- **Layer normalization**: Stabilizes training by normalizing activations
- **Pre-norm architecture**: Applies normalization before each sublayer (more stable)

### Character-Level Modeling

This implementation uses characters rather than subwords:

**Advantages**:
- No vocabulary limits or unknown tokens
- Perfect for educational purposes
- Works for any text without preprocessing

**Disadvantages**:
- Longer sequences needed to represent the same information
- Harder to capture long-range semantic relationships
- Requires more compute for the same "conceptual" context

### Sampling Strategies

During generation, we convert logits to a probability distribution:

- **Temperature**: Controls randomness (low = conservative, high = creative)
- **Top-k sampling**: Only considers the k most probable tokens
- Balances between deterministic (greedy) and fully random sampling

## Further Exploration

### Experiment Ideas

1. **Scaling up**: Increase `embed_dim`, `num_layers`, or `num_heads` to see how model capacity affects quality
2. **Context length**: Modify `block_size` to give the model longer memory
3. **Training duration**: Run more `steps` for better convergence
4. **Generation tuning**: Adjust temperature and top_k during generation

### Theoretical Extensions

- Replace learned positional embeddings with rotary (RoPE) or ALiBi
- Implement attention dropout or MLP dropout for regularization
- Add gradient clipping to prevent instability
- Experiment with different activation functions (GELU, SwiGLU)

## References

- Vaswani et al. (2017): "Attention Is All You Need"
- Radford et al. (2019): "Language Models are Unsupervised Multitask Learners" (GPT-2)
- Merity et al. (2016): "Pointer Sentinel Mixture Models" (WikiText dataset)

## License

This is an educational project. Feel free to use, modify, and learn from the code.