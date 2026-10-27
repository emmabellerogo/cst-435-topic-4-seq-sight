"""Training and evaluation workflow for the Seq Sight forecasting models.

One code path trains both ``lstm`` and ``attention`` so that only the model
differs between runs. Held constant by construction: data version, scaler,
training windows, batch size, the batch order of every epoch, optimizer, loss,
learning rate and gradient-clipping policy.

Initial choices (documented defaults, not tuned): Adam, lr 1e-3, batch size 128,
L1 loss on standardized values (the same quantity MAE measures, up to the
constant std), gradient-norm clipping at 1.0, no dropout, no early stopping and no
learning-rate schedule. The epoch cap is configurable and is a cap, not evidence
of convergence.

Reproducibility. ``seed_everything`` seeds Python, NumPy and torch before the
model is built (so initialization is determined by the seed). Batch order comes
from a SEPARATE ``torch.Generator`` created fresh for every epoch from
``(seed, epoch)`` alone, so it cannot be changed by how many random numbers the
model's initialization or dropout consumed. Both models therefore see identical
batches in identical order; a SHA-256 of each epoch's index sequence is recorded
in the history to prove it.

Test discipline. ``train_model`` is given only training and validation windows;
the test split is not reachable from it. The checkpoint is selected by validation
MAE in degrees Celsius alone. Test MAE is computed only by the explicit final
``evaluate_test`` call.

Artifacts are saved with ``torch.save`` of plain tensors/numbers/strings and loaded
with ``weights_only=True``. Even so, load only artifacts you produced locally.
"""
from __future__ import annotations

import copy
import hashlib
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch import nn

from shared import forecast_models as fm
from shared.forecasting import ForecastDataError, Standardizer, WindowSet

ARTIFACT_FORMAT = 1
EVAL_BATCH = 1024


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    max_epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    grad_clip_norm: float = 1.0
    optimizer: str = "adam"
    loss: str = "l1_standardized"
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.max_epochs < 1 or self.batch_size < 1 or self.lr <= 0 or self.grad_clip_norm <= 0:
            raise ValueError("max_epochs, batch_size, lr and grad_clip_norm must be positive")
        if self.optimizer != "adam" or self.loss != "l1_standardized" or self.dropout != 0.0:
            raise ValueError("only adam / l1_standardized / dropout 0.0 are implemented")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvalResult:
    mae_c: float                       # over ALL examples and ALL 24 forecast hours
    mae_by_horizon_c: List[float]      # [24], hour t+1 .. t+24
    n_examples: int


@dataclass
class TrainResult:
    model: nn.Module                   # best-validation weights loaded, eval mode
    model_type: str
    model_config: Dict[str, Any]
    train_config: Dict[str, Any]
    best_epoch: int
    best_val_mae_c: float
    history: List[Dict[str, Any]]
    optimizer_steps: int
    parameter_count: int
    train_seconds: float               # wall clock of the whole call, incl. validation
    state_dict: Dict[str, torch.Tensor] = field(repr=False, default_factory=dict)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def epoch_permutation(n: int, seed: int, epoch: int) -> torch.Tensor:
    """Shuffled indices for one epoch, from a dedicated generator keyed by (seed, epoch)."""
    g = torch.Generator()
    g.manual_seed(seed * 1_000_003 + epoch)
    return torch.randperm(n, generator=g)


def _require_windows(w: WindowSet, name: str) -> None:
    if len(w) == 0:
        raise ForecastDataError(f"{name} has no windows")


# ------------------------------------------------------------------ evaluation

def _accumulate(predict: Callable[[torch.Tensor], torch.Tensor], w: WindowSet, scaler: Standardizer,
                batch_size: int) -> EvalResult:
    """Exact MAE in degC: sum of absolute errors over all examples/hours, divided once.

    Never a mean of per-batch means, so a short final batch is weighted correctly.
    |pred - target| in degC equals |pred_z - target_z| * std (the mean cancels).
    """
    x_all, y_all = torch.from_numpy(w.inputs), torch.from_numpy(w.targets)
    n = len(w)
    abs_err_by_hour = np.zeros(fm.TARGET_HOURS, dtype=np.float64)
    for i in range(0, n, batch_size):
        pred = predict(x_all[i:i + batch_size])
        err = (pred.double() - y_all[i:i + batch_size].double()).abs().sum(dim=0).numpy()
        abs_err_by_hour += err
    by_hour = abs_err_by_hour / n * scaler.std
    if not np.isfinite(by_hour).all():
        raise ForecastDataError("non-finite predictions during evaluation")
    return EvalResult(float(by_hour.mean()), by_hour.tolist(), n)


def evaluate(model: nn.Module, w: WindowSet, scaler: Standardizer, batch_size: int = EVAL_BATCH) -> EvalResult:
    """Batched eval-mode MAE in degC; attention matrices are discarded batch by batch."""
    _require_windows(w, "evaluation set")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            return _accumulate(lambda xb: model(xb)[0], w, scaler, batch_size)
    finally:
        model.train(was_training)


def persistence_forecast(inputs: torch.Tensor) -> torch.Tensor:
    """Repeat the last observed value over the next 24 hours: ``[B,72,1] -> [B,24]``.

    The standardizer is affine, so repeating the last standardized value is the
    same as repeating the last temperature in degC.
    """
    return inputs[:, -1, :].expand(-1, fm.TARGET_HOURS).clone()


def evaluate_persistence(w: WindowSet, scaler: Standardizer, batch_size: int = EVAL_BATCH) -> EvalResult:
    _require_windows(w, "evaluation set")
    return _accumulate(persistence_forecast, w, scaler, batch_size)


# -------------------------------------------------------------------- training

def train_model(model_type: str, train: WindowSet, val: WindowSet, scaler: Standardizer, config: TrainConfig,
                model_kwargs: Optional[Dict[str, Any]] = None,
                log: Callable[[str], None] = print) -> TrainResult:
    """Train one model for up to ``config.max_epochs``; keep the best-validation weights."""
    _require_windows(train, "training set")
    _require_windows(val, "validation set")
    seed_everything(config.seed)
    model = fm.build_model(model_type, **(model_kwargs or {}))
    model_config = model.config()
    n_params = fm.count_parameters(model)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    x_all, y_all = torch.from_numpy(train.inputs), torch.from_numpy(train.targets)
    n = len(train)
    log(f"[{model_type}] params={n_params:,} windows train={n:,} val={len(val):,} config={config.to_dict()}")

    history: List[Dict[str, Any]] = []
    best_val, best_epoch, best_state = float("inf"), 0, None
    steps = 0
    t_start = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        t_epoch = time.perf_counter()
        model.train()
        order = epoch_permutation(n, config.seed, epoch)
        digest = hashlib.sha256()
        loss_sum = 0.0
        for i in range(0, n, config.batch_size):          # final batch may be shorter
            idx = order[i:i + config.batch_size]
            digest.update(idx.numpy().tobytes())
            pred, _ = model(x_all[idx])
            loss = torch.nn.functional.l1_loss(pred, y_all[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            opt.step()
            steps += 1
            loss_sum += float(loss) * len(idx)            # weight by examples, not batches
        train_s = time.perf_counter() - t_epoch
        t_val = time.perf_counter()
        val_res = evaluate(model, val, scaler)
        record = {
            "epoch": epoch, "train_loss_standardized_l1": loss_sum / n, "val_mae_c": val_res.mae_c,
            "train_seconds": train_s, "val_seconds": time.perf_counter() - t_val,
            "elapsed_seconds": time.perf_counter() - t_start, "optimizer_steps": steps,
            "batch_order_sha256": digest.hexdigest(),
        }
        improved = val_res.mae_c < best_val               # strict: ties keep the earlier epoch
        if improved:
            best_val, best_epoch = val_res.mae_c, epoch
            best_state = copy.deepcopy(model.state_dict())
        record["best_so_far"] = improved
        history.append(record)
        log(f"[{model_type}] epoch {epoch}/{config.max_epochs} train_l1={record['train_loss_standardized_l1']:.4f} "
            f"val_mae={val_res.mae_c:.4f} degC steps={steps} elapsed={record['elapsed_seconds']:.1f}s"
            + ("  *best*" if improved else ""))
    if best_state is None:
        raise ForecastDataError("no finite validation MAE was produced; check the data and learning rate")
    model.load_state_dict(best_state)
    model.eval()
    return TrainResult(model, model_type, model_config, config.to_dict(), best_epoch, best_val, history, steps,
                       n_params, time.perf_counter() - t_start, best_state)


# ------------------------------------------------------------------- artifacts

def save_artifact(path, result: TrainResult, scaler: Standardizer, data_version: str,
                  extra: Optional[Dict[str, Any]] = None) -> None:
    """One file holding everything needed to reproduce predictions and the run record."""
    payload = {
        "format_version": ARTIFACT_FORMAT,
        "model_type": result.model_type,
        "model_config": result.model_config,
        "state_dict": result.state_dict,
        "scaler": scaler.to_dict(),
        "data_version": data_version,
        "seed": result.train_config["seed"],
        "train_config": result.train_config,
        "history": result.history,
        "best_epoch": result.best_epoch,
        "best_val_mae_c": result.best_val_mae_c,
        "optimizer_steps": result.optimizer_steps,
        "parameter_count": result.parameter_count,
        "train_seconds": result.train_seconds,
        "torch_version": str(torch.__version__),
        "extra": extra or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


_REQUIRED = ("format_version", "model_type", "model_config", "state_dict", "scaler", "data_version", "seed",
             "train_config", "history", "best_epoch", "best_val_mae_c")


def load_artifact(path) -> tuple[nn.Module, Standardizer, Dict[str, Any]]:
    """Rebuild ``(model in eval mode, scaler, metadata)``. Trusted local files only."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    missing = [k for k in _REQUIRED if k not in payload]
    if missing:
        raise ForecastDataError(f"artifact is missing {missing}")
    if payload["format_version"] != ARTIFACT_FORMAT:
        raise ForecastDataError(f"unsupported artifact format {payload['format_version']}")
    cfg = dict(payload["model_config"])
    model = fm.build_model(cfg.pop("model_type"), **cfg)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    meta = {k: v for k, v in payload.items() if k != "state_dict"}
    return model, Standardizer.from_dict(payload["scaler"]), meta


def evaluate_test(model: nn.Module, test: WindowSet, scaler: Standardizer) -> Dict[str, Any]:
    """The explicit FINAL operation: call once, after all configurations are fixed."""
    res = evaluate(model, test, scaler)
    base = evaluate_persistence(test, scaler)
    return {"test_mae_c": res.mae_c, "test_mae_by_horizon_c": res.mae_by_horizon_c,
            "persistence_test_mae_c": base.mae_c, "n_examples": res.n_examples}
