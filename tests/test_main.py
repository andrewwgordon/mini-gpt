import pytest
import torch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import minillm


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
@pytest.fixture
def tiny_text():
    return "hello world\nthis is a test dataset\n"


@pytest.fixture
def tokenizer(tiny_text):
    return minillm.CharTokenizer(tiny_text)


@pytest.fixture
def tiny_model(tokenizer):
    # Very small model for testing
    return minillm.MiniGPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=32,
        num_heads=4,
        num_layers=1,
        mlp_ratio=2,
        block_size=16,
    )


@pytest.fixture
def tiny_data(tokenizer, tiny_text):
    return torch.tensor(tokenizer.encode(tiny_text), dtype=torch.long)


# ------------------------------------------------------------
# Test: tokenizer
# ------------------------------------------------------------
def test_tokenizer_roundtrip(tokenizer):
    s = "hello"
    encoded = tokenizer.encode(s)
    decoded = tokenizer.decode(encoded)
    assert decoded == s, "Tokenizer encode/decode must round-trip."


# ------------------------------------------------------------
# Test: data batching
# ------------------------------------------------------------
def test_get_batch(tiny_data):
    block_size = 8
    batch_size = 4

    x, y = minillm.get_batch(tiny_data, block_size, batch_size)

    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)

    # y should be x shifted by 1 in the original data
    for i in range(batch_size):
        assert torch.all(y[i, :-1] == x[i, 1:])


# ------------------------------------------------------------
# Test: model forward pass
# ------------------------------------------------------------
def test_model_forward_pass(tiny_model):
    batch = torch.randint(0, tiny_model.lm_head.out_features, (2, 10))
    logits = tiny_model(batch)
    assert logits.shape == (2, 10, tiny_model.lm_head.out_features)


# ------------------------------------------------------------
# Test: training loop (very small)
# ------------------------------------------------------------
def test_training_step(monkeypatch, tiny_model, tiny_data):
    # Patch eval_interval to 1 for faster output
    def fake_print(*args, **kwargs):
        pass
    monkeypatch.setattr("builtins.print", fake_print)

    # Run with 1 tiny step to ensure training loop executes
    minillm.train(
        model=tiny_model,
        data=tiny_data,
        steps=1,
        block_size=tiny_model.block_size,
        batch_size=4,
        lr=1e-3,
        eval_interval=1,
    )

    # After training, model should have a saved file
    p = minillm.Path("minilm.pt")
    assert p.exists(), "Training should save minilm.pt"


# ------------------------------------------------------------
# Test: generation
# ------------------------------------------------------------
def test_generation_smoke(monkeypatch, tiny_model, tokenizer):
    # Disable randomness by patching torch.multinomial
    def mock_multinomial(probs, num_samples):
        return torch.zeros((probs.size(0), num_samples), dtype=torch.long)
    monkeypatch.setattr(torch, "multinomial", mock_multinomial)

    # Also disable prints
    monkeypatch.setattr("builtins.print", lambda *args, **kwargs: None)

    text = minillm.generate(
        tiny_model,
        tokenizer,
        prompt="hi",
        max_new_tokens=10,
        temperature=1.0,
        top_k=5,
    )

    assert isinstance(text, str)
    assert len(text) > 0
