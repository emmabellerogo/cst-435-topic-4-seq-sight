"""Numerical test: the model recovers the trigger-token classes AND its attention
localizes on the class-marker position.

Every synthetic sequence's label is decided by a single class marker placed right
after a randomly-positioned trigger token. A model that solves the task must
learn to *attend* to that position, so we assert both (a) held-out accuracy
clears a floor and (b) the per-timestep attention weight peaks on (or next to)
the marker. That is what makes the attention viz genuinely explanatory.
"""
from __future__ import annotations

import numpy as np

from api.training import predict_seq, train_attention
from shared.data import SEQ_LEN, generate_one


def test_attention_recovers_classes_and_localizes_unit():
    metrics, model_b64, loss = train_attention(
        n_rows=800, seq_len=SEQ_LEN, seed=42,
        lr=0.01, batch_size=32, epochs=12,
    )
    assert metrics["accuracy"] > 0.85
    assert metrics["macro_f1"] > 0.85
    assert loss[-1] < loss[0]  # training loss decreased
    assert isinstance(model_b64, str) and len(model_b64) > 0

    # Attention should peak on the marker (trigger_pos + 1) for most samples.
    hits = 0
    n = 40
    for seed in range(n):
        seq, _label, pos = generate_one(seq_len=SEQ_LEN, seed=1000 + seed)
        result = predict_seq(model_b64, [int(t) for t in seq])
        attn = np.asarray(result["attention"])
        assert abs(float(attn.sum()) - 1.0) < 1e-4  # weights are a distribution
        if int(attn.argmax()) in (pos, pos + 1):
            hits += 1
    assert hits / n > 0.8
