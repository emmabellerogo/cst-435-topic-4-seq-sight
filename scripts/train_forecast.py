"""Train and compare the LSTM and attention forecasters (explicit, local CLI).

Nothing here runs on import. Two data sources:

  --source local     prepare the local Jena archives with db.load.prepare()
                     (no network, no credentials)
  --source supabase  read the stored corpus for --data-version from Supabase via
                     shared.forecast_store (reads SUPABASE_URL / SUPABASE_SERVICE_KEY
                     from .env or the environment; needs the `supabase` package)

Run kinds:

  --run-kind smoke   timing / sanity run. Artifacts go to artifacts/smoke/ and are
                     EXCLUDED from any comparison. Test evaluation is refused.
  --run-kind final   artifacts go to artifacts/final/. Add --evaluate-test ONLY
                     after every configuration is fixed from validation results.

Both models get the same data, seed, batch size, batch order, optimizer, loss,
learning rate and clipping. Validation MAE in degC selects each checkpoint; the
test split is touched only with --evaluate-test, once, at the end.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from shared import forecast_models as fm  # noqa: E402
from shared import forecast_training as ft  # noqa: E402
from shared.forecasting import prepare_forecasting_data  # noqa: E402

DEFAULT_VERSION = "jena-mpi-roof-h1-60138aac7d898b8c"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("local", "supabase"), required=True)
    ap.add_argument("--run-kind", choices=("smoke", "final"), required=True)
    ap.add_argument("--data-version", default=DEFAULT_VERSION, help="required to match when --source supabase")
    ap.add_argument("--models", nargs="+", choices=fm.MODEL_TYPES, default=list(fm.MODEL_TYPES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, required=True, help="epoch CAP, not a convergence claim")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-train-windows", type=int, default=None,
                    help="smoke only: use the first K training windows to shorten timing runs")
    ap.add_argument("--out-dir", type=Path, default=REPO / "artifacts")
    ap.add_argument("--evaluate-test", action="store_true", help="final operation; refused for smoke runs")
    return ap


def load_data(args):
    if args.source == "local":
        from db import load
        prep = load.prepare()
        return prepare_forecasting_data(prep.rows, expected_data_version=prep.data_version)
    from db import load  # config parsing only; never prints values
    from shared.forecast_store import load_forecasting_data
    from supabase import create_client
    cfgs = load.load_config()
    return load_forecasting_data(create_client(cfgs.url, cfgs.key), args.data_version)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.run_kind == "smoke" and args.evaluate_test:
        print("ERROR: --evaluate-test is refused for smoke runs", file=sys.stderr)
        return 2
    if args.max_train_windows is not None and args.run_kind != "smoke":
        print("ERROR: --max-train-windows is only allowed for smoke runs", file=sys.stderr)
        return 2
    data = load_data(args)
    if args.source == "local" and data.data_version != args.data_version:
        print(f"WARNING: local data version {data.data_version} != {args.data_version}")
    train = data.splits["train"]
    if args.max_train_windows:
        from shared.forecasting import WindowSet
        k = args.max_train_windows
        train = WindowSet(train.inputs[:k], train.targets[:k], train.origins[:k])
    print(f"data_version={data.data_version} windows: "
          + ", ".join(f"{k}={len(v):,}" for k, v in data.splits.items()))
    if args.run_kind == "smoke":
        print("*** SMOKE RUN: timing only, excluded from final comparison ***")

    val = data.splits["validation"]
    base = ft.evaluate_persistence(val, data.scaler)
    print(f"persistence reference, validation MAE: {base.mae_c:.4f} degC")

    config = ft.TrainConfig(seed=args.seed, max_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    out_dir = args.out_dir / args.run_kind
    summary = {"data_version": data.data_version, "run_kind": args.run_kind, "persistence_val_mae_c": base.mae_c,
               "models": {}}
    results = {}
    for kind in args.models:   # each call reseeds and rebuilds the batch-order generator from the seed
        res = ft.train_model(kind, train, val, data.scaler, config)
        path = out_dir / f"{kind}_seed{args.seed}.pt"
        ft.save_artifact(path, res, data.scaler, data.data_version,
                         extra={"run_kind": args.run_kind, "source": args.source,
                                "train_windows_used": len(train)})
        results[kind] = res
        summary["models"][kind] = {
            "artifact": str(path), "best_epoch": res.best_epoch, "best_val_mae_c": res.best_val_mae_c,
            "epochs_run": len(res.history), "optimizer_steps": res.optimizer_steps,
            "parameter_count": res.parameter_count, "train_seconds": res.train_seconds,
            "seconds_per_epoch": res.train_seconds / len(res.history)}
    if args.evaluate_test:
        summary["test"] = {k: ft.evaluate_test(r.model, data.splits["test"], data.scaler) for k, r in results.items()}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
