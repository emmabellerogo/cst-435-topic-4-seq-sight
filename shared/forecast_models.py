"""Seq Sight forecasting models: an LSTM baseline and a self-attention model.

Both map a standardized history ``[batch, 72, 1]`` (float32) to a standardized
24 h forecast ``[batch, 24]`` and share one return contract::

    forecast, attention = model(x)
    # LSTMForecaster:      attention is None (it has no attention weights)
    # AttentionForecaster: attention is [batch, 72, 72]

Only the 72 known historical hours enter either model; no target value is an
input. Full (non-causal) attention over the history is therefore legitimate.

IMPORTANT - what the attention matrix means. ``attention[b, i, j]`` says how much
the contextualized representation of input hour *i* draws on input hour *j*
(rows sum to 1 over j). It describes INPUT-TO-INPUT relationships inside the
encoder. It is NOT an attribution of any one of the 24 output hours to input
hours: the forecast head reads pooled representations, so no output hour has its
own attention row. Describe it as "where the encoder's attention weights were",
not as "why hour t+k was predicted".

Architectures (kept small for local CPU training)
-------------------------------------------------
LSTMForecaster: LSTM(1 -> hidden, 1 layer) -> final hidden state -> ForecastHead.

AttentionForecaster:
    Linear(1 -> d_model)                      input projection
    + sinusoidal positional encoding          fixed, not learned
    single-head scaled dot-product self-attention (hand-written):
        Q = X Wq, K = X Wk, V = X Wv
        A = softmax(Q K^T / sqrt(d_k), over keys);  C = A V
    H = LayerNorm(X + C)                      residual keeps each hour's own signal
    pooling = concat[ H[:, -1], mean_t H ]    -> [batch, 2 * d_model]
    ForecastHead

Pooling: after attention, every row H[:, t] is a contextualized representation of
hour t that has mixed in information from all 72 hours. The last row (most recent
hour, which the next 24 hours directly follow) is the natural summary of "where
the series is now in the context of its history"; the mean row is an order-free
summary of the whole window. Concatenating both and mapping them with the head to
all 24 outputs at once (direct multi-horizon, no autoregressive feedback) gives
the head both the current state and the window-wide context. Pooling discards
which input hour contributed what, which is why the attention matrix cannot be
read as per-output attribution.

ForecastHead (identical in both models): Linear(in -> head_hidden) -> ReLU ->
Linear(head_hidden -> 24). Only the encoder differs, as the data design requires.

Equal parameter counts do NOT mean equal compute: attention costs O(T^2) in the
sequence length and parallelizes over time, while the LSTM is O(T) but sequential.
Compare measured training time, not just parameter counts.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

INPUT_HOURS = 72
TARGET_HOURS = 24
MODEL_TYPES = ("lstm", "attention")  # matches runs.model_type in the database


def _check_input(x: torch.Tensor) -> None:
    if x.ndim != 3 or tuple(x.shape[1:]) != (INPUT_HOURS, 1):
        raise ValueError(f"expected input [batch, {INPUT_HOURS}, 1], got {tuple(x.shape)}")
    if x.dtype != torch.float32:
        raise ValueError(f"expected float32 input, got {x.dtype}")


def sinusoidal_positional_encoding(length: int, d_model: int) -> torch.Tensor:
    """``[length, d_model]``: PE[p, 2i] = sin(p / 10000^(2i/d)), PE[p, 2i+1] = cos(...)."""
    if d_model % 2:
        raise ValueError("d_model must be even for sinusoidal positional encoding")
    pos = torch.arange(length, dtype=torch.float64).unsqueeze(1)
    freq = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float64) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(length, d_model, dtype=torch.float64)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe.float()


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """softmax(q k^T / sqrt(d_k)) v for ``[..., T, d]`` tensors; returns (output, weights)."""
    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # [..., T_q, T_k]
    weights = torch.softmax(scores, dim=-1)                          # over keys
    return torch.matmul(weights, v), weights


class ForecastHead(nn.Module):
    def __init__(self, in_features: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, TARGET_HOURS))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class LSTMForecaster(nn.Module):
    """Single-layer LSTM baseline. Returns ``(forecast [B, 24], None)``."""

    def __init__(self, hidden_size: int = 32, head_hidden: int = 64):
        super().__init__()
        self.hidden_size, self.head_hidden = hidden_size, head_hidden
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.head = ForecastHead(hidden_size, head_hidden)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        _check_input(x)
        _, (h_n, _) = self.lstm(x)           # h_n: [1, B, hidden] = state after the 72nd hour
        return self.head(h_n[-1]), None

    def config(self) -> Dict[str, Any]:
        return {"model_type": "lstm", "hidden_size": self.hidden_size, "head_hidden": self.head_hidden}


class AttentionForecaster(nn.Module):
    """Single-head self-attention encoder. Returns ``(forecast [B, 24], weights [B, 72, 72])``."""

    def __init__(self, d_model: int = 32, head_hidden: int = 64):
        super().__init__()
        self.d_model, self.head_hidden = d_model, head_hidden
        self.input_proj = nn.Linear(1, d_model)
        self.register_buffer("pos_enc", sinusoidal_positional_encoding(INPUT_HOURS, d_model), persistent=False)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = ForecastHead(2 * d_model, head_hidden)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _check_input(x)
        h = self.input_proj(x) + self.pos_enc                        # [B, 72, d]
        context, weights = scaled_dot_product_attention(self.w_q(h), self.w_k(h), self.w_v(h))
        h = self.norm(h + context)                                   # [B, 72, d] contextualized
        pooled = torch.cat([h[:, -1], h.mean(dim=1)], dim=-1)        # [B, 2d]
        return self.head(pooled), weights

    def config(self) -> Dict[str, Any]:
        return {"model_type": "attention", "d_model": self.d_model, "head_hidden": self.head_hidden}


def build_model(model_type: str, **kwargs) -> nn.Module:
    if model_type == "lstm":
        return LSTMForecaster(**kwargs)
    if model_type == "attention":
        return AttentionForecaster(**kwargs)
    raise ValueError(f"unknown model_type {model_type!r}; expected one of {MODEL_TYPES}")


def count_parameters(model: nn.Module) -> int:
    """Trainable parameters (the fixed positional encoding is a buffer, not counted)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
