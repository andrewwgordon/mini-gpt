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
"""README for mini-gpt (educational mini GPT implementation)

This file provides an overview, quick start instructions, and a concise
explanation of the deep learning concepts used by the project.
"""

# mini-gpt — Miniature GPT-style Language Model (Educational)

mini-gpt is a compact, educational implementation of a GPT-style (decoder-only) transformer that operates at the character level. The code is intentionally small and readable so you can study the core components of modern neural language models: tokenization, embeddings, causal self-attention, transformer blocks, training loop, and autoregressive generation.

**Repository layout**
- `minillm.py` — Main script: tokenizer, model, training loop, generation, and the CLI.
- `minilm.pt` / `tinygpt.pt` — Example model checkpoints (if present).
- `tests/` — Minimal tests to validate basic script behaviour.

**Goals**
- Explain a working GPT-style model in a few hundred lines of PyTorch.
- Provide a configurable experiment harness to change `embed_dim`, `num_layers`, `num_heads`, and other hyperparameters.
- Be runnable on CPU (small models) or GPU (if available) with minimal dependencies.

**What this project is not**
- Not a production LLM; it is designed for learning and experimentation.
- Not optimized for speed or memory (no KV-cache, quantization, or distributed training helpers).

**Quick Start**

Requirements
- Python 3.8+
- PyTorch
- HuggingFace `datasets` (optional; script falls back to small sample text if offline)

Install dependencies

```bash
pip install -r requirements.txt
# or, at minimum
pip install torch datasets
```

Train a small model (default configuration)

```bash
python minillm.py --train --steps 500
```

Train with custom architecture

```bash
python minillm.py --train --steps 1000 --embed_dim 256 --num_layers 4 --num_heads 8 --batch_size 64
```

Generate text from a checkpoint

```bash
python minillm.py --generate --prompt "The quick brown fox"
```

If you saved a model to `minilm.pt` with different hyperparameters, create the same model configuration on load (same `embed_dim`, `num_layers`, `num_heads`, `block_size`, `vocab_size`), otherwise you will see size mismatch errors. The script contains a robust loader that helps detect and partially load matching tensors — see "Troubleshooting" below.

CLI flags (important ones)
- `--train`: run training loop and save `minilm.pt` when finished
- `--generate`: load `minilm.pt` (if present) and generate text from `--prompt`
- `--prompt`: text prompt for generation (default: `"Hello:"`)
- `--batch_size` (default 32)
- `--block_size` (default 128)
- `--embed_dim` (default 128)
- `--num_heads` (default 4)
- `--num_layers` (default 2)
- `--mlp_ratio` (default 4)
- `--learning_rate` (default 3e-4)
- `--steps` (default 200)
- `--eval_interval` (default 50)

Design and Implementation Notes

- Tokenization: character-level `CharTokenizer` built from the training text (no BPE).
- Model: `MiniGPT` — token + position embeddings, stack of `TransformerBlock` modules with pre-layernorm, and final LM head.
- Attention: causal multi-head self-attention with a lower-triangular mask.
- Training: next-token prediction using cross-entropy loss, AdamW optimizer.

Deep Learning Theory (concise)

Transformers
- Replaced recurrence with attention mechanisms (Vaswani et al., 2017).
- Self-attention computes pairwise relations between tokens enabling long-range dependencies.

Scaled Dot-Product Attention
- Compute Q, K, V projections; attention weights = softmax(Q K^T / sqrt(d_k)).
- `d_k` scaling keeps gradients stable for larger embedding sizes.

Multi-Head Attention
- Multiple attention heads allow the model to attend to different subspaces and types of relationships in parallel.

Causal Masking
- Ensures autoregressive prediction: each position only attends to itself and previous positions (no peeking into the future).

Position Embeddings
- Adds positional information to token embeddings so the model can utilize order.
- This code uses learned position embeddings; alternatives include sinusoidal encodings or Rotary embeddings (RoPE).

Training Objective
- The model is trained to minimize negative log-likelihood / cross-entropy of next-token predictions:

	Loss = -∑ log P(token_t | token_<t)

This is self-supervised; the training data (text) provides the supervision.

Why character-level?
- Pros: no OOV tokens, simpler tokenizer, great for teaching.
- Cons: longer sequences per semantic unit, slower to learn some patterns, larger effective context required.

Troubleshooting: model size / shape mismatches

Common cause: saved checkpoint parameters do not match the model you instantiated (different `embed_dim`, `num_layers`, `num_heads`, `block_size`, or `vocab_size`).

The repo's loader in `minillm.py` includes a robust loading routine that:
- Strips `module.` prefixes from keys (if saved from `DataParallel`).
- Filters checkpoint tensors to only those whose names and shapes match the model's state dict.
- Prints diagnostics showing checkpoint keys count, model keys count, loaded keys count, and warnings for missing/unexpected keys.

Recommended workflow to avoid errors:
1. Save hyperparameters alongside checkpoints (not implemented here by default). Keep a note of the model configuration used to train each checkpoint.
2. Recreate the exact model when loading: same `embed_dim`, `num_layers`, `num_heads`, `block_size`, and `vocab_size`.
3. If you intentionally change architecture and want to reuse parts of a checkpoint (e.g., token embeddings), use the robust loader or a custom script to selectively copy matching tensors.

Example diagnostic snippet (already integrated in `minillm.py`):

```python
# Loads checkpoint, strips `module.` prefix, and prints counts for keys/shapes
state = torch.load("minilm.pt", map_location="cpu")
# ... filter / compare ...
```

Extending and experiments

- Try increasing `embed_dim`, `num_layers`, and `num_heads` to see quality gains (watch memory).
- Implement KV-cache to speed up generation (necessary for larger models and longer contexts).
- Replace character tokenizer with a subword tokenizer (e.g., SentencePiece) to compare training speed and quality.
- Add checkpointing / resume training and validation evaluation.

Tests

Run the minimal tests in `tests/` (if present):

```bash
pytest -q
```

(if `pytest` is not installed: `pip install pytest`)

License & Attribution

This is an educational project. Code is free to use for learning and experimentation. See `LICENSE` for details.

Acknowledgements

- Core ideas: Vaswani et al. "Attention Is All You Need" (2017)
- Dataset example: WikiText-2
