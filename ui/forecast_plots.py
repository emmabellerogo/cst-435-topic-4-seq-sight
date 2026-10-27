"""Matplotlib figures for the UI (object-oriented API; no pyplot global state)."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from matplotlib.figure import Figure

INPUT_HOURS, FORECAST_HOURS = 72, 24


def forecast_figure(history_c: Optional[Sequence[float]], forecast_c: Sequence[float],
                    actual_c: Optional[Sequence[float]] = None, title: str = "") -> Figure:
    """History at hours -71..0, forecast (and optionally the actual values) at +1..+24."""
    fig = Figure(figsize=(8, 3.4), layout="constrained")
    ax = fig.subplots()
    if history_c is not None:
        ax.plot(range(-INPUT_HOURS + 1, 1), history_c, color="0.35", label="input history")
    hours = range(1, FORECAST_HOURS + 1)
    if actual_c is not None:
        ax.plot(hours, actual_c, color="tab:green", marker="o", ms=3, label="actual")
    ax.plot(hours, forecast_c, color="tab:orange", marker="o", ms=3, label="forecast")
    ax.axvline(0.5, color="0.7", ls=":")
    ax.set_xlabel("hours relative to the last input hour")
    ax.set_ylabel("temperature (°C)")
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    return fig


def attention_figure(att: np.ndarray, title: str = "Encoder self-attention (input hour → input hour)") -> Figure:
    """Heatmap of a 72x72 matrix: row i = the hour doing the attending, column j = the hour attended to."""
    att = np.asarray(att)
    if att.shape != (INPUT_HOURS, INPUT_HOURS):
        raise ValueError(f"expected a 72x72 matrix, got {att.shape}")
    fig = Figure(figsize=(6.2, 5.2), layout="constrained")
    ax = fig.subplots()
    im = ax.imshow(att, cmap="viridis", aspect="equal", origin="upper")
    ticks = [0, 12, 24, 36, 48, 60, 71]
    ax.set_xticks(ticks, [str(t - 71) for t in ticks])
    ax.set_yticks(ticks, [str(t - 71) for t in ticks])
    ax.set_xlabel("attended-to input hour j (hours before the last input hour)")
    ax.set_ylabel("attending input hour i")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="weight (each row sums to 1)")
    return fig


def mean_attention_profile_figure(att: np.ndarray) -> Figure:
    """How much attention, averaged over rows, each input hour receives."""
    fig = Figure(figsize=(8, 2.4), layout="constrained")
    ax = fig.subplots()
    ax.bar(range(-INPUT_HOURS + 1, 1), np.asarray(att).mean(axis=0), color="tab:blue")
    ax.set_xlabel("input hour j (hours before the last input hour)")
    ax.set_ylabel("mean weight received")
    return fig


def comparison_figure(history_c, actual_c, lstm_c, attention_c) -> Figure:
    """Same window, both models' forecasts against the actual values."""
    fig = forecast_figure(history_c, lstm_c, actual_c)
    ax = fig.axes[0]
    ax.lines[-1].set_label("LSTM forecast")
    ax.plot(range(1, FORECAST_HOURS + 1), attention_c, color="tab:blue", marker="s", ms=3,
            label="attention forecast")
    ax.legend(loc="best", fontsize=8)
    return fig
