"""Register trained forecasting artifacts as `runs` + `run_artifacts` rows.

DRY-RUN BY DEFAULT: with no flags this reads only local artifact files, prints
what would be written, and exits. It needs no credentials and no network.

    python scripts/register_forecast_runs.py                       # dry-run, six epoch50 artifacts
    python scripts/register_forecast_runs.py --execute --test-mae-file test_mae.json

Writing requires BOTH ``--execute`` and ``--test-mae-file``. ``runs.test_mae_c``
is NOT NULL and the final test evaluation has not been run, so until measured
values exist this command cannot write anything -- by design it never inserts a
zero, a validation metric or any other placeholder.

``--test-mae-file`` is a JSON object mapping artifact key to measured test MAE in
degC, e.g. ``{"lstm_seed0": 1.23, "attention_seed0": 1.31, ...}`` (all selected
keys required). Execution reads SUPABASE_URL / SUPABASE_SERVICE_KEY from .env or
the environment via ``db.load.load_config`` and never prints them.

After a successful run, set the printed ids as FORECAST_<TYPE>_RUN_ID for the API.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from shared.forecast_registration import (RegistrationError, attach_test_mae,  # noqa: E402
                                          default_keys, execute_registration, plan_registration)

DEFAULT_DIR = REPO / "artifacts" / "epoch50" / "final"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact-dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--keys", nargs="+", default=None,
                    help="artifact stems to register (default: lstm/attention seeds 0-2)")
    ap.add_argument("--execute", action="store_true", help="actually write to Supabase (default: dry-run)")
    ap.add_argument("--test-mae-file", type=Path, default=None,
                    help="JSON {key: measured test MAE degC}; required with --execute")
    return ap


def main(argv=None, client_factory=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plans = plan_registration(args.artifact_dir, args.keys or default_keys())
    except RegistrationError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if args.test_mae_file:
        try:
            attach_test_mae(plans, json.loads(args.test_mae_file.read_text()))
        except (OSError, ValueError, RegistrationError) as e:
            print(f"ERROR: test MAE file rejected: {e}", file=sys.stderr)
            return 2

    print(f"{'EXECUTE' if args.execute else 'DRY-RUN'}: {len(plans)} artifact(s) from {args.artifact_dir.name}/")
    for p in plans:
        r = p.run
        print(f"  {p.key}: model_type={r['model_type']} seed={r['seed']} data_version={r['data_version']} "
              f"val_mae_c={r['val_mae_c']:.4f} artifact_sha256={p.artifact_sha256[:12]} "
              f"model_b64_chars={p.model_b64_chars:,} "
              f"test_mae_c={'MISSING' if p.test_mae_c is None else p.test_mae_c}")
    pending = [p.key for p in plans if p.test_mae_c is None]
    if pending:
        print("\nBLOCKED: test_mae_c is NOT NULL in `runs`, and the final test evaluation has not been run.\n"
              "No test MAE is available for: " + ", ".join(pending) + ".\n"
              "Registration will not insert zero, a validation metric or any placeholder.")

    if not args.execute:
        print("\nDry-run only; nothing was written.")
        return 0
    if pending:
        print("ERROR: refusing to execute without measured test MAE for every artifact "
              "(pass --test-mae-file).", file=sys.stderr)
        return 2
    try:
        if client_factory is None:
            from db import load
            from supabase import create_client
            cfg = load.load_config()
            client_factory = lambda: create_client(cfg.url, cfg.key)  # noqa: E731
        results = execute_registration(client_factory(), plans)
    except Exception as e:  # noqa: BLE001 - config/supabase errors may mention the URL; print type only
        msg = str(e) if isinstance(e, RegistrationError) else type(e).__name__
        print(f"ERROR: registration failed: {msg}", file=sys.stderr)
        return 1
    for r in results:
        print(f"  {r['key']}: {r['action']} run_id={r['run_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
