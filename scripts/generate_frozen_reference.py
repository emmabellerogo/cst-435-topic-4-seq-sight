"""Generate a frozen-reference fixture ONCE (tests/fixtures/*.json). Never overwrites.

    python -m scripts.generate_frozen_reference --kind synthetic
    python -m scripts.generate_frozen_reference --kind trained     # needs the git-ignored seed-0 artifacts

``trained`` loads artifacts/epoch50/final/{lstm,attention}_seed0.pt (or --artifact-dir), checks their
SHA-256 against the values pinned in scripts/curate_forecast_examples.py, and runs the fixed input
through the serving path (api.forecast_service.run_forecasts). ``synthetic`` does the same for
untrained models built with torch.manual_seed(0) and a fixed scaler. Neither reads a corpus, trains,
evaluates a split, or touches the network or credentials. An existing fixture is never replaced:
delete it deliberately (and review the diff) to re-freeze.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from api.forecast_service import file_sha256, load_model, run_forecasts  # noqa: E402
from shared.forecast_models import MODEL_TYPES  # noqa: E402
from tests import frozen_reference as fr  # noqa: E402

DEFAULT_DIR = REPO / "artifacts" / "epoch50" / "final"


def _forecasts(models):
    hist = fr.reference_input()
    out = {}
    for kind, loaded in models.items():
        f = run_forecasts(loaded, [hist])[0]
        exp = {"forecast_c": f.forecast_c}
        if f.attention is not None:
            exp["attention_probe"] = fr.attention_probe(f.attention)
        out[kind] = exp
    return out


def _doc(kind, models, extra):
    return {
        "format": fr.FORMAT, "kind": kind,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generation": ("scripts/generate_frozen_reference.py --kind " + kind + "; input run through "
                       "api.forecast_service.run_forecasts (the /predict path), eval mode, CPU, float32 model "
                       "with float64 scaling"),
        "torch_version": str(torch.__version__),
        "input": {"units": "degC", "definition": fr.INPUT_DEFINITION, "history_c": fr.reference_input()},
        "tolerances": {"forecast_atol_c": fr.FORECAST_ATOL_C, "attention_atol": fr.ATTENTION_ATOL,
                       "note": "absolute; allows float32 and platform/torch-build differences, far below any "
                               "change in behavior. A larger difference means the model, scaler or "
                               "preprocessing changed."},
        "expected": _forecasts(models), **extra,
    }


def trained(artifact_dir: Path) -> dict:
    from scripts.curate_forecast_examples import FIXED, DATA_VERSION
    models, hashes, scalers = {}, {}, set()
    for k in MODEL_TYPES:
        path = Path(artifact_dir) / FIXED[k]["file"]
        digest = file_sha256(path)
        if digest != FIXED[k]["sha256"]:
            raise SystemExit(f"{path.name}: SHA-256 differs from the pinned seed-0 artifact")
        models[k] = load_model(k, path)
        hashes[k] = digest
        scalers.add((models[k].scaler.mean, models[k].scaler.std))
    if len(scalers) != 1:
        raise SystemExit("artifacts disagree on the scaler")
    scaler = models["lstm"].scaler.to_dict()
    return _doc("trained_seed0", models, {
        "artifacts": {k: {"file": FIXED[k]["file"], "sha256": hashes[k], "run_id_reported_by_user": FIXED[k]["run_id"]}
                      for k in MODEL_TYPES},
        "scaler": scaler, "data_version": DATA_VERSION})


def synthetic() -> dict:
    from tests.test_forecast_api import make_artifact
    from shared.forecasting import Standardizer
    models = {}
    with tempfile.TemporaryDirectory() as d:
        for k in MODEL_TYPES:
            p = Path(d) / f"{k}.pt"
            make_artifact(p, k, seed=fr.SYNTHETIC_SEED, scaler=Standardizer(fr.SYNTHETIC_SCALER["mean"],
                                                                             fr.SYNTHETIC_SCALER["std"]))
            models[k] = load_model(k, p)
    return _doc("synthetic_untrained", models, {
        "model_recipe": {"seed": fr.SYNTHETIC_SEED, "construction": "torch.manual_seed(seed); "
                         "shared.forecast_models.build_model(kind) with default sizes; untrained",
                         "disclaimer": "NOT the trained checkpoints; guards the serving code path only"},
        "scaler": fr.SYNTHETIC_SCALER, "data_version": "synthetic-v1"})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=("trained", "synthetic"), required=True)
    ap.add_argument("--artifact-dir", type=Path, default=DEFAULT_DIR)
    args = ap.parse_args(argv)
    out = fr.TRAINED_FIXTURE if args.kind == "trained" else fr.SYNTHETIC_FIXTURE
    if out.exists():
        print(f"ERROR: {out.relative_to(REPO)} exists; refusing to overwrite", file=sys.stderr)
        return 2
    torch.set_num_threads(1)
    doc = trained(args.artifact_dir) if args.kind == "trained" else synthetic()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print(f"wrote {out.relative_to(REPO)} ({out.stat().st_size / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
