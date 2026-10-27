"""Evaluate FIXED forecasting checkpoints on the test split (evaluation only).

THIS IS THE ONE-TIME FINAL TEST EVALUATION. It never trains, never updates a
weight, and never chooses between checkpoints: all six artifacts are scored and
reported, and nothing in the output ranks or selects them. Which models to serve
must be decided from validation results BEFORE looking at this output.

DRY-RUN BY DEFAULT: with no flags it loads the local artifacts, cross-checks their
metadata and prints the plan. It reads no corpus, needs no credentials and makes no
network request.

    python scripts/evaluate_forecast_artifacts.py                 # dry-run
    python scripts/evaluate_forecast_artifacts.py --execute       # reads Supabase, scores test

``--execute`` reads SUPABASE_URL / SUPABASE_SERVICE_KEY (via ``db.load.load_config``,
never printed), loads the corpus for the artifacts' data version through
``shared.forecast_store``, and refuses unless that corpus reproduces each
artifact's data version and scaler exactly. The result file is created exclusively
(``--output``, default ``<artifact-dir>/test_eval.json``); an existing file is never
overwritten, and there is no force flag. Move it away deliberately to re-run.

The output is accepted directly by ``scripts/register_forecast_runs.py
--test-mae-file``, which also re-checks each artifact's SHA-256 against it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from shared import forecast_training as ft  # noqa: E402
from shared.forecast_registration import RESULT_FORMAT, default_keys, file_sha256  # noqa: E402
from shared.forecasting import ForecastDataError  # noqa: E402

DEFAULT_DIR = REPO / "artifacts" / "epoch50" / "final"
SCALER_TOL = 1e-9


class EvalError(RuntimeError):
    pass


def weights_fingerprint(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_fixed(artifact_dir: Path, keys):
    """Load trusted checkpoints and cross-check their metadata. No data is touched."""
    items = []
    for key in keys:
        path = Path(artifact_dir) / f"{key}.pt"
        if not path.is_file():
            raise EvalError(f"artifact not found: {key}.pt")
        kind, _, seed = key.rpartition("_seed")
        model, scaler, meta = ft.load_artifact(path)
        if meta["model_type"] != kind or str(meta["seed"]) != seed or meta["train_config"].get("seed") != meta["seed"]:
            raise EvalError(f"{key}: file name disagrees with artifact model_type/seed metadata")
        items.append({"key": key, "path": path, "model": model, "scaler": scaler, "meta": meta,
                      "sha256": file_sha256(path)})
    for field in ("data_version",):
        if len({i["meta"][field] for i in items}) != 1:
            raise EvalError(f"artifacts disagree on {field}")
    if len({(i["scaler"].mean, i["scaler"].std) for i in items}) != 1:
        raise EvalError("artifacts disagree on the scaler")
    return items


def check_corpus_matches(items, data) -> None:
    ref = items[0]
    if data.data_version != ref["meta"]["data_version"]:
        raise EvalError("corpus data_version does not match the artifacts")
    s, a = data.scaler, ref["scaler"]
    if abs(s.mean - a.mean) > SCALER_TOL or abs(s.std - a.std) > SCALER_TOL:
        raise EvalError("scaler refit from the corpus does not match the artifacts' scaler")


def evaluate_fixed(items, data, batch_size: int = ft.EVAL_BATCH):
    """Score every checkpoint on the test split. Weights are fingerprinted before and after."""
    test = data.splits["test"]
    if len(test) == 0:
        raise EvalError("test split has no windows")
    persistence = ft.evaluate_persistence(test, data.scaler, batch_size)
    models = {}
    for it in items:
        before = weights_fingerprint(it["model"])
        res = ft.evaluate(it["model"], test, it["scaler"], batch_size)
        if weights_fingerprint(it["model"]) != before:
            raise EvalError(f"{it['key']}: weights changed during evaluation")
        models[it["key"]] = {
            "model_type": it["meta"]["model_type"], "seed": it["meta"]["seed"],
            "artifact_sha256": it["sha256"], "weights_sha256": before,
            "best_epoch": it["meta"]["best_epoch"], "val_mae_c": it["meta"]["best_val_mae_c"],
            "test_mae_c": res.mae_c, "test_mae_by_horizon_c": res.mae_by_horizon_c,
            "persistence_test_mae_c": persistence.mae_c, "n_examples": res.n_examples}
    return persistence, models


def build_result(items, data, persistence, models, batch_size: int) -> dict:
    return {
        "format": RESULT_FORMAT,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_version": data.data_version,
        "scaler": items[0]["scaler"].to_dict(),
        "evaluation_config": {"split": "test", "batch_size": batch_size, "mode": "eval, torch.no_grad",
                              "metric": "mean absolute error over all windows and all 24 hours, degC",
                              "n_test_windows": len(data.splits["test"]),
                              "training": "none", "checkpoint_selection": "none (all artifacts reported)",
                              "torch_version": str(torch.__version__)},
        "persistence_test_mae_c": persistence.mae_c,
        "test_mae_c": {k: m["test_mae_c"] for k, m in models.items()},   # what the registration command reads
        "models": models,
    }


def write_exclusive(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "x", encoding="utf-8") as f:     # "x": fail if it exists; never overwrite
            json.dump(result, f, indent=2)
            f.write("\n")
    except FileExistsError:
        raise EvalError("output file already exists; refusing to overwrite") from None


def default_loader(data_version: str):
    from db import load
    from shared.forecast_store import load_forecasting_data
    from supabase import create_client
    cfg = load.load_config()
    return load_forecasting_data(create_client(cfg.url, cfg.key), data_version)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact-dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--keys", nargs="+", default=None, help="artifact stems (default: lstm/attention seeds 0-2)")
    ap.add_argument("--output", type=Path, default=None, help="default: <artifact-dir>/test_eval.json")
    ap.add_argument("--batch-size", type=int, default=ft.EVAL_BATCH)
    ap.add_argument("--execute", action="store_true", help="read the corpus and score the test split")
    return ap


def main(argv=None, data_loader=default_loader) -> int:
    args = build_parser().parse_args(argv)
    output = args.output or args.artifact_dir / "test_eval.json"
    try:
        if args.batch_size < 1:
            raise EvalError("--batch-size must be >= 1")
        items = load_fixed(args.artifact_dir, args.keys or default_keys())
        version = items[0]["meta"]["data_version"]
        print(f"{'EXECUTE' if args.execute else 'DRY-RUN'}: {len(items)} fixed checkpoint(s), data_version={version}")
        for it in items:
            print(f"  {it['key']}: sha256={it['sha256'][:12]} val_mae_c={it['meta']['best_val_mae_c']:.4f}")
        print(f"  output: {output.name} ({'exists' if output.exists() else 'will be created'})")
        if not args.execute:
            print("\nDry-run only: no corpus read, no test data touched, nothing written.")
            return 0
        if output.exists():
            raise EvalError("output file already exists; refusing to overwrite")
        data = data_loader(version)
        check_corpus_matches(items, data)
        persistence, models = evaluate_fixed(items, data, args.batch_size)
        result = build_result(items, data, persistence, models, args.batch_size)
        write_exclusive(output, result)
    except (EvalError, ForecastDataError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 - config/supabase errors may mention the URL; print type only
        print(f"ERROR: evaluation failed: {type(e).__name__}", file=sys.stderr)
        return 1
    print(f"test persistence reference: {persistence.mae_c:.4f} degC")
    for k, m in models.items():
        print(f"  {k}: test_mae_c={m['test_mae_c']:.4f}")
    print(f"wrote {output.name}. These figures are for REPORTING; do not use them to select or tune models.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
