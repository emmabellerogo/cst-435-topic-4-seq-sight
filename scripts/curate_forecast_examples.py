"""Select and measure the curated comparison examples and failure cases (validation only).

Writes ``ui/content/examples.json``. Local and read-only with respect to everything
else: it reads the local Jena ZIP files (verified against the checksum manifest by
``db.load.prepare``) and two FIXED checkpoints, and makes no network request, reads no
credentials, trains nothing and never touches the test split.

    python -m scripts.curate_forecast_examples --measure-only   # print the selection, write nothing
    python -m scripts.curate_forecast_examples                  # write ui/content/examples.json
    python -m scripts.curate_forecast_examples --check          # re-derive and compare with the file

Fixed inputs (not chosen from any result): seed-0 LSTM (run 1) and seed-0 attention
(run 4) from ``artifacts/epoch50/final``; their SHA-256 values are pinned below and
checked. Run ids are the ids the user reported for the registered runs; this script
cannot verify them (it does not contact Supabase).

SELECTION RULES (written before the gallery was produced; thresholds are not tuned)
-----------------------------------------------------------------------------------
Pool: windows of the VALIDATION split only. Origin = last input hour. A selected
window occupies 96 hourly values (72 input + 24 target).

Spacing, applied to all six selected windows together: origins at least
``MIN_SEPARATION_H`` = 168 h (7 days) apart, so the 96 h spans cannot overlap and the
cases are on different weather days.

Comparison examples (3). Chosen from the ACTUAL TEMPERATURES ONLY; no forecast is
looked at. Candidates have origin hour 00:00 (targets then cover the next calendar
day). Three patterns, tested in this order, a window belonging to the first it meets:
  diurnal  range(actual) >= 10 degC and range(history) >= 10 degC and
           |mean(actual) - mean(history)| <= 3 degC          (strong daily cycle, no trend)
  flat     range(history + actual) <= 4 degC                  (nearly constant temperature)
  shift    |mean(actual) - mean(last 24 h of history)| >= 6 degC  (large level change)
Within each pattern the window whose pattern statistic (range(actual), range(span),
|shift|) is closest to the median over that pattern's candidates is taken (ties: earliest).

Failure cases (3). Candidates are validation windows where the ATTENTION model's MAE
over the 24 forecast hours is >= ``FAILURE_MAE_THRESHOLD_C`` = 5.0 degC (about 2.6x
its aggregate validation MAE, and above the persistence reference). Taken greedily
by descending attention MAE, skipping any window within the spacing rule of one
already taken (comparison examples included). This INTENTIONALLY selects difficult
cases; it says nothing about average performance. The LSTM's error on the same
windows is reported but never used to select.

Stored numbers are rounded to 4 decimals (degC) and attention weights to 6 decimals;
every MAE in the file is computed from the stored, rounded values.

Hand-written interpretation lives in ``ui/content/example_notes.json`` (keyed by case
key); the script appends it to the auto-generated measured paragraph.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from db import load  # noqa: E402
from shared import forecast_training as ft  # noqa: E402
from shared.forecast_registration import file_sha256  # noqa: E402
from shared.forecasting import (HOUR, INPUT_HOURS, TARGET_HOURS, prepare_forecasting_data,  # noqa: E402
                                prepare_inference_input)

ARTIFACT_DIR = REPO / "artifacts" / "epoch50" / "final"
OUTPUT = REPO / "ui" / "content" / "examples.json"
NOTES = REPO / "ui" / "content" / "example_notes.json"

# Fixed, pre-selected artifacts (seed 0). Hashes as recorded by the evaluation/registration step.
FIXED = {
    "lstm": {"file": "lstm_seed0.pt", "run_id": 1,
             "sha256": "de0af50530a4a4d9eddf33fa3ccfd492dd467268fd95e62816a8277b5115aa69"},
    "attention": {"file": "attention_seed0.pt", "run_id": 4,
                  "sha256": "82c51854a590124317c76dc973d03bdd68a0ebe799a347abebd4669ad7048234"},
}
DATA_VERSION = "jena-mpi-roof-h1-60138aac7d898b8c"

FAILURE_MAE_THRESHOLD_C = 5.0
MIN_SEPARATION_H = 168
N_EXAMPLES = 3
TEMP_DECIMALS, ATT_DECIMALS = 4, 6
SCALER_TOL = 1e-9
CHECK_TOL_C = 2e-4

PATTERNS = ("diurnal", "flat", "shift")


class CurationError(RuntimeError):
    pass


# ------------------------------------------------------------------ loading

def load_models():
    out = {}
    for kind, spec in FIXED.items():
        path = ARTIFACT_DIR / spec["file"]
        if not path.is_file():
            raise CurationError(f"missing artifact {path.name}")
        digest = file_sha256(path)
        if digest != spec["sha256"]:
            raise CurationError(f"{path.name}: SHA-256 {digest[:12]} differs from the pinned {spec['sha256'][:12]}")
        model, scaler, meta = ft.load_artifact(path)
        if meta["model_type"] != kind or meta["seed"] != 0 or meta["data_version"] != DATA_VERSION:
            raise CurationError(f"{path.name}: unexpected model_type/seed/data_version")
        out[kind] = (model, scaler, meta)
    a, b = out["lstm"][1], out["attention"][1]
    if (a.mean, a.std) != (b.mean, b.std):
        raise CurationError("the two artifacts carry different scalers")
    return out


def load_corpus(scaler):
    """Local, manifest-verified corpus -> validated windows. Scaler must equal the frozen one."""
    prepared = load.prepare()
    if prepared.data_version != DATA_VERSION:
        raise CurationError(f"local corpus is {prepared.data_version}, artifacts expect {DATA_VERSION}")
    data = prepare_forecasting_data(prepared.rows, expected_data_version=DATA_VERSION)
    if abs(data.scaler.mean - scaler.mean) > SCALER_TOL or abs(data.scaler.std - scaler.std) > SCALER_TOL:
        raise CurationError("scaler refit from the local corpus does not match the artifacts' frozen scaler")
    temps = {datetime.fromisoformat(r["station_ts"]): r["temperature_c"] for r in prepared.rows}
    return data, temps


def predict_all(model, windows, batch=1024):
    out = np.empty((len(windows), TARGET_HOURS), dtype=np.float64)
    x = torch.from_numpy(windows.inputs)
    with torch.no_grad():
        for i in range(0, len(windows), batch):
            out[i:i + batch] = model(x[i:i + batch])[0].double().numpy()
    return out


# ----------------------------------------------------------------- selection

def _spaced(origin, taken):
    return all(abs((origin - t) / HOUR) >= MIN_SEPARATION_H for t in taken)


def pattern_of(h, a):
    span = np.concatenate([h, a])
    if np.ptp(a) >= 10 and np.ptp(h) >= 10 and abs(a.mean() - h.mean()) <= 3:
        return "diurnal", float(np.ptp(a))
    if np.ptp(span) <= 4:
        return "flat", float(np.ptp(span))
    shift = float(a.mean() - h[-24:].mean())
    if abs(shift) >= 6:
        return "shift", abs(shift)
    return None, 0.0


def select_comparison(windows, hist_c, act_c):
    pools = {p: [] for p in PATTERNS}
    for i, o in enumerate(windows.origins):
        if o.hour != 0:
            continue
        p, stat = pattern_of(hist_c[i], act_c[i])
        if p:
            pools[p].append((stat, i))
    chosen, counts = [], {}
    for p in PATTERNS:
        counts[p] = len(pools[p])
        if not pools[p]:
            raise CurationError(f"no validation window matches the {p!r} pattern")
        med = statistics.median(s for s, _ in pools[p])
        for stat, i in sorted(pools[p], key=lambda t: (abs(t[0] - med), windows.origins[t[1]])):
            if _spaced(windows.origins[i], [windows.origins[j] for _, j in chosen]):
                chosen.append((p, i))
                break
        else:
            raise CurationError(f"no spaced window for pattern {p!r}")
    return chosen, counts


def select_failures(windows, att_mae, taken_origins):
    cand = [int(i) for i in np.argsort(-att_mae, kind="stable") if att_mae[i] >= FAILURE_MAE_THRESHOLD_C]
    chosen = []
    for i in cand:
        if _spaced(windows.origins[i], taken_origins + [windows.origins[j] for j in chosen]):
            chosen.append(i)
            if len(chosen) == N_EXAMPLES:
                break
    if len(chosen) < N_EXAMPLES:
        raise CurationError(f"only {len(chosen)} spaced windows reach MAE >= {FAILURE_MAE_THRESHOLD_C} degC")
    return chosen, len(cand)


# ---------------------------------------------------------------- measurement

def r4(x):
    return [round(float(v), TEMP_DECIMALS) for v in x]


def mae(p, a):
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(a))))


def serve_path_forecast(model, scaler, history_c):
    """Forecast through the same preprocessing the API uses (stored scaler, one 72 h history)."""
    x = torch.from_numpy(prepare_inference_input(history_c, scaler))
    with torch.no_grad():
        f, att = model(x)
    return scaler.inverse_transform(f[0].double().numpy()), (None if att is None else att[0].double().numpy())


def raw_window(temps, origin):
    hist = [temps[origin - (INPUT_HOURS - 1 - k) * HOUR] for k in range(INPUT_HOURS)]
    act = [temps[origin + (k + 1) * HOUR] for k in range(TARGET_HOURS)]
    return hist, act


def attention_stats(att):
    last = att[-1]
    recv = att.mean(axis=0)
    ent = -(att * np.log(np.clip(att, 1e-12, None))).sum(axis=1)
    return {
        "last_row_argmax_hours_before_last_input": int(INPUT_HOURS - 1 - last.argmax()),
        "last_row_max_weight": round(float(last.max()), 4),
        "last_row_weight_on_last_24h": round(float(last[-24:].sum()), 4),
        "mean_weight_received_by_last_24h": round(float(recv[-24:].sum()), 4),
        "mean_row_entropy_nats": round(float(ent.mean()), 4),
        "uniform_row_entropy_nats": round(float(np.log(INPUT_HOURS)), 4),
        "most_attended_column_on_average_hours_before_last_input": int(INPUT_HOURS - 1 - recv.argmax()),
    }


def measured_text(c):
    a = np.array(c["actual_c"]); p_l = np.array(c["lstm_predicted_c"]); p_a = np.array(c["attention_predicted_c"])
    hist = np.array(c["history_c"])
    return " ".join([
        f"Measured (validation window, origin {c['origin']}): last observed {hist[-1]:.2f} °C; actual next 24 h ranged "
        f"{a.min():.2f} to {a.max():.2f} °C (mean {a.mean():.2f}).",
        f"LSTM run {c['lstm_run_id']}: MAE {c['lstm_mae_c']:.2f} °C, mean signed error {np.mean(p_l - a):+.2f} °C, forecast "
        f"range {p_l.min():.2f} to {p_l.max():.2f} °C.",
        f"Attention run {c['attention_run_id']}: MAE {c['attention_mae_c']:.2f} °C, mean signed error {np.mean(p_a - a):+.2f} °C, "
        f"forecast range {p_a.min():.2f} to {p_a.max():.2f} °C.",
        f"Persistence (repeat the last value) MAE for reference: {c['persistence_mae_c']:.2f} °C."])


def measured_text_failure(c):
    a = np.array(c["actual_c"]); p = np.array(c["predicted_c"]); hist = np.array(c["history_c"])
    err = p - a
    s = c["attention_summary"]
    return (f"Measured (validation window, origin {c['origin']}): last observed {hist[-1]:.2f} °C; actual next 24 h ranged "
            f"{a.min():.2f} to {a.max():.2f} °C. Attention run {c['run_id']} forecast {p.min():.2f} to {p.max():.2f} °C: "
            f"MAE {c['mae_c']:.2f} °C, mean signed error {err.mean():+.2f} °C, largest hourly error "
            f"{np.abs(err).max():.2f} °C at +{int(np.abs(err).argmax()) + 1} h. The LSTM (run {c['lstm_run_id']}) on the same "
            f"window: MAE {c['lstm_mae_c']:.2f} °C; persistence: {c['persistence_mae_c']:.2f} °C. Encoder attention: the "
            f"final-hour row puts {s['last_row_weight_on_last_24h']:.2f} of its weight on the last 24 input hours "
            f"(max single weight {s['last_row_max_weight']:.3f}); mean row entropy {s['mean_row_entropy_nats']:.2f} nats "
            f"versus {s['uniform_row_entropy_nats']:.2f} for uniform.")


def build(data, temps, models):
    w = data.splits["validation"]
    scaler = models["lstm"][1]
    n = len(w)
    hist_c = scaler.inverse_transform(w.inputs[:, :, 0].astype(np.float64))
    act_c = scaler.inverse_transform(w.targets.astype(np.float64))
    pred = {k: scaler.inverse_transform(predict_all(models[k][0], w)) for k in models}
    att_mae = np.abs(pred["attention"] - act_c).mean(axis=1)
    pers_mae = np.abs(hist_c[:, -1:] - act_c).mean(axis=1)

    comp_sel, counts = select_comparison(w, hist_c, act_c)
    fail_sel, n_cand = select_failures(w, att_mae, [w.origins[i] for _, i in comp_sel])

    def common(i):
        origin = w.origins[i]
        hist, act = raw_window(temps, origin)               # from the corpus rows, not the standardized array
        if np.abs(np.array(hist) - hist_c[i]).max() > 1e-4 or np.abs(np.array(act) - act_c[i]).max() > 1e-4:
            raise CurationError(f"window {origin}: corpus values disagree with the prepared window")
        out = {}
        for k in models:
            f, att = serve_path_forecast(models[k][0], scaler, hist)
            if np.abs(f - pred[k][i]).max() > 1e-3:
                raise CurationError(f"window {origin}: serving-path forecast disagrees with the batch scan ({k})")
            out[k] = (f, att)
        return origin, hist, act, out

    def prov(origin):
        return {"split": "validation", "data_version": DATA_VERSION,
                "station_time": "naive hour-end labels, never converted",
                "history_first_hour": (origin - (INPUT_HOURS - 1) * HOUR).isoformat(),
                "history_last_hour": origin.isoformat(),
                "target_first_hour": (origin + HOUR).isoformat(),
                "target_last_hour": (origin + TARGET_HOURS * HOUR).isoformat()}

    comparisons, failures = [], []
    for p, i in comp_sel:
        origin, hist, act, out = common(i)
        hist, act = r4(hist), r4(act)
        lp, ap = r4(out["lstm"][0]), r4(out["attention"][0])
        att = np.round(out["attention"][1], ATT_DECIMALS)
        comparisons.append({
            "key": f"comparison:{p}", "pattern": p, "origin": origin.isoformat(), "history_c": hist, "actual_c": act,
            "lstm_predicted_c": lp, "attention_predicted_c": ap,
            "lstm_run_id": FIXED["lstm"]["run_id"], "attention_run_id": FIXED["attention"]["run_id"],
            "lstm_artifact_sha256": FIXED["lstm"]["sha256"], "attention_artifact_sha256": FIXED["attention"]["sha256"],
            "lstm_mae_c": round(mae(lp, act), 4), "attention_mae_c": round(mae(ap, act), 4),
            "persistence_mae_c": round(mae([hist[-1]] * TARGET_HOURS, act), 4),
            "attention": att.tolist(), "attention_summary": attention_stats(att),
            "input_provenance": prov(origin)})
    for i in fail_sel:
        origin, hist, act, out = common(i)
        hist, act = r4(hist), r4(act)
        lp, ap = r4(out["lstm"][0]), r4(out["attention"][0])
        att = np.round(out["attention"][1], ATT_DECIMALS)
        failures.append({
            "key": f"failure:{origin.date().isoformat()}", "origin": origin.isoformat(), "model_type": "attention",
            "run_id": FIXED["attention"]["run_id"], "artifact_sha256": FIXED["attention"]["sha256"],
            "history_c": hist, "actual_c": act, "predicted_c": ap,
            "mae_c": round(mae(ap, act), 4), "max_abs_error_c": round(float(np.abs(np.array(ap) - act).max()), 4),
            "lstm_run_id": FIXED["lstm"]["run_id"], "lstm_predicted_c": lp, "lstm_mae_c": round(mae(lp, act), 4),
            "persistence_mae_c": round(mae([hist[-1]] * TARGET_HOURS, act), 4),
            "attention": att.tolist(), "attention_summary": attention_stats(att),
            "input_provenance": prov(origin)})

    notes = json.loads(NOTES.read_text(encoding="utf-8")) if NOTES.exists() else {}
    for c in comparisons + failures:
        mt = measured_text(c) if c["key"].startswith("comparison") else measured_text_failure(c)
        note = notes.get(c["key"])
        c["title"] = (note or {}).get("title") or c["key"]
        c["discussion"] = mt + ("\n\n" + note["text"] if note else "")
    agg = {k: round(float(np.abs(pred[k] - act_c).mean()), 4) for k in pred}
    meta = {
        "purpose": "Measured, curated examples. Failure cases are DELIBERATELY the hardest validation windows for the "
                   "attention model and do not represent average performance. Neither model reports uncertainty or "
                   "calibrated confidence.",
        "generator": "scripts/curate_forecast_examples.py (rules in its docstring; docs/forecasting-examples.md)",
        "split": "validation (the test split is not used)", "data_version": DATA_VERSION,
        "scaler": scaler.to_dict(),
        "corpus": "local ZIPs verified against docs/jena-mpi-roof-checksums.txt; Supabase was not contacted",
        "checkpoints": {k: {"run_id": FIXED[k]["run_id"], "artifact": f"artifacts/epoch50/final/{FIXED[k]['file']}",
                            "artifact_sha256": FIXED[k]["sha256"]} for k in FIXED},
        "run_id_note": "run ids are as reported by the user for the registered runs; not verified by this script",
        "rules": {"failure_mae_threshold_c": FAILURE_MAE_THRESHOLD_C, "min_origin_separation_h": MIN_SEPARATION_H,
                  "comparison_patterns": list(PATTERNS),
                  "rounding": f"{TEMP_DECIMALS} dp degC, {ATT_DECIMALS} dp attention"},
        "selection_audit": {"validation_windows": n, "comparison_candidates_by_pattern": counts,
                            "failure_candidates_at_or_above_threshold": n_cand,
                            "aggregate_validation_mae_c_this_scan": agg,
                            "persistence_validation_mae_c_this_scan": round(float(pers_mae.mean()), 4)},
    }
    return {"_readme": "Measured examples only; see _meta and ui/content/README.md.", "_meta": meta,
            "comparison_examples": comparisons, "failure_cases": failures}


def summarize(doc):
    print(json.dumps(doc["_meta"]["selection_audit"], indent=2))
    for c in doc["comparison_examples"]:
        print(f"{c['key']:<24} origin {c['origin']}  LSTM {c['lstm_mae_c']:.3f}  attention {c['attention_mae_c']:.3f}  "
              f"persistence {c['persistence_mae_c']:.3f}")
    for c in doc["failure_cases"]:
        print(f"{c['key']:<24} origin {c['origin']}  attention {c['mae_c']:.3f} (max {c['max_abs_error_c']:.2f})  "
              f"LSTM {c['lstm_mae_c']:.3f}  persistence {c['persistence_mae_c']:.3f}")


def check(doc, path):
    old = json.loads(path.read_text(encoding="utf-8"))
    bad = []
    for sect, fields in (("comparison_examples", ("history_c", "actual_c", "lstm_predicted_c", "attention_predicted_c")),
                         ("failure_cases", ("history_c", "actual_c", "predicted_c", "lstm_predicted_c"))):
        if [c["origin"] for c in old[sect]] != [c["origin"] for c in doc[sect]]:
            bad.append(f"{sect}: different windows selected")
            continue
        for a, b in zip(old[sect], doc[sect]):
            for f in fields:
                if np.abs(np.array(a[f]) - np.array(b[f])).max() > CHECK_TOL_C:
                    bad.append(f"{a['origin']} {f} differs")
            if np.abs(np.array(a["attention"]) - np.array(b["attention"])).max() > 1e-4:
                bad.append(f"{a['origin']} attention differs")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measure-only", action="store_true", help="print the selection and write nothing")
    ap.add_argument("--check", action="store_true", help="re-derive and compare with the existing file")
    ap.add_argument("--output", type=Path, default=OUTPUT)
    args = ap.parse_args(argv)
    try:
        torch.set_num_threads(1)
        models = load_models()
        data, temps = load_corpus(models["lstm"][1])
        doc = build(data, temps, models)
    except CurationError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    summarize(doc)
    if args.check:
        bad = check(doc, args.output)
        print("CHECK FAILED:\n  " + "\n  ".join(bad) if bad else "CHECK OK: selection and measured values reproduce the file")
        return 1 if bad else 0
    if args.measure_only:
        return 0
    args.output.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output.relative_to(REPO)} ({args.output.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
