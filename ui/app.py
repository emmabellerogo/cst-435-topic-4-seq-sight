"""Streamlit UI -- Cloud #1 (deployed on Streamlit Community Cloud).

Seq Sight: 24 h temperature forecasts from a 72 h history. The three-cloud
structure (Streamlit UI -> FastAPI -> Supabase) and this file's scaffolding were
adapted from the professor-provided Attend-It template; the tabs, client logic
and content here are Seq Sight's own.

This is a THIN client:
  * forecasts are HTTPS calls to the FastAPI service (no torch, no model code);
  * the ONLY database access is read-only, with the PUBLIC anon key
    (tables `runs` and `predictions`). The service-role key never belongs here.

Configuration comes from st.secrets (see .streamlit/secrets.toml.example at the repository root):
    API_URL, SUPABASE_URL, SUPABASE_ANON_KEY
Logic lives in forecast_client.py / forecast_plots.py so it can be unit-tested.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import forecast_client as fc
import forecast_plots as fp

HERE = Path(__file__).resolve().parent

st.set_page_config(page_title="Seq Sight", page_icon="🌡️", layout="wide")
st.title("🌡️ Seq Sight — 24 h Temperature Forecasts with Attention")
st.caption("Streamlit (this UI) → FastAPI (LSTM and self-attention) → Supabase (runs and audit log). "
           "Historical Jena data; not a validated forecast for any other place.")


# ------------------------------------------------------------------ connections
def secret(name: str) -> str:
    try:
        return str(st.secrets[name]).strip()
    except Exception:  # noqa: BLE001 - missing secrets file or key
        return ""


def api() -> fc.ForecastApi | None:
    url = secret("API_URL")
    return fc.ForecastApi(url) if url.startswith("http") and "your-service" not in url else None


@st.cache_resource
def supabase_anon():
    """Anon-key client. Returns (client, None) or (None, reason)."""
    url, key = secret("SUPABASE_URL"), secret("SUPABASE_ANON_KEY")
    if not url.startswith("https://") or "your-project-ref" in url or not key or key == "your-anon-key":
        return None, "SUPABASE_URL / SUPABASE_ANON_KEY are not configured in Streamlit secrets."
    try:
        fc.check_anon_key(key)
        from supabase import create_client
        return create_client(url, key), None
    except ValueError as e:
        return None, str(e)
    except Exception:  # noqa: BLE001
        return None, "Could not create the Supabase client; check the URL and anon key."


def db_call(fn, *args, **kw):
    """Run a Supabase read; show a friendly error instead of a traceback."""
    client, why = supabase_anon()
    if client is None:
        st.warning(why)
        return None
    try:
        return fn(client, *args, **kw)
    except Exception:  # noqa: BLE001 - never echo library errors (may contain the URL)
        st.error("The database read failed. Check the Supabase settings and that the tables exist.")
        return None


@st.cache_data(ttl=60, show_spinner=False)
def cached_runs(_token: str):
    client, _why = supabase_anon()
    return None if client is None else fc.fetch_runs(client)


_examples_file = HERE / "content" / "examples.json"
examples = fc.load_examples(_examples_file.read_text(encoding="utf-8") if _examples_file.exists() else None)

concepts, try_tab, compare_tab, recent_tab, failure_tab, card_tab = st.tabs(
    ["Concepts", "Try It", "Baseline vs. Attention", "Recent Predictions", "Failure Gallery", "Model Card"])

# ---------------------------------------------------------------------------
# 1. Concepts
# ---------------------------------------------------------------------------
with concepts:
    st.header("Recurrence versus self-attention")
    st.markdown(
        "Both models read the same 72 standardized hourly temperatures and output 24 numbers at once "
        "(hours +1 … +24). They differ in how the 72 hours are combined.")
    c1, c2 = st.columns(2)
    c1.subheader("LSTM (baseline)")
    c1.markdown(
        "Reads hour 1, then hour 2, … hour 72, carrying a state $h_t$ forward: "
        "$h_t = \\mathrm{LSTM}(x_t, h_{t-1})$. The forecast is computed from the final state $h_{72}$, so "
        "everything the model knows about hour 1 must survive 71 state updates.")
    c2.subheader("Self-attention")
    c2.markdown(
        "Projects each hour to a vector, adds position information, and lets **every hour look at every "
        "other hour directly** in one step. The forecast is computed from the last hour's updated vector "
        "plus the mean over all hours.")

    st.subheader("Where the formula comes from")
    st.markdown(
        "Each hour $i$ gets a **query** $q_i = x_iW_Q$, a **key** $k_i = x_iW_K$ and a **value** "
        "$v_i = x_iW_V$, each of dimension $d_k$. The compatibility of hour $i$ with hour $j$ is the dot "
        "product $q_i\\cdot k_j$. Stacking all hours gives the score matrix $QK^{\\top}$ (72×72).")
    st.markdown("**Why divide by $\\sqrt{d_k}$.** Suppose the components of $q$ and $k$ are independent with "
                "mean 0 and variance 1. Then")
    st.latex(r"q\cdot k=\sum_{m=1}^{d_k} q_m k_m,\qquad "
             r"\mathbb{E}[q\cdot k]=0,\qquad \mathrm{Var}(q\cdot k)=\sum_{m=1}^{d_k}\mathrm{Var}(q_mk_m)=d_k .")
    st.markdown("so the typical size of a score grows like $\\sqrt{d_k}$. Large scores push the softmax toward "
                "a one-hot vector, where its gradient is almost zero. Dividing by $\\sqrt{d_k}$ restores "
                "unit variance:")
    st.latex(r"\mathrm{Var}\!\left(\frac{q\cdot k}{\sqrt{d_k}}\right)=1 .")
    st.markdown("Turning scores into weights (softmax over the keys $j$, so each row sums to 1) and mixing the values:")
    st.latex(r"A_{ij}=\frac{\exp\!\left(q_i\cdot k_j/\sqrt{d_k}\right)}{\sum_{l}\exp\!\left(q_i\cdot k_l/\sqrt{d_k}\right)},"
             r"\qquad c_i=\sum_j A_{ij}v_j")
    st.latex(r"\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V")

    st.subheader("Worked example: the effect of scaling")
    ex = fc.scaling_example()
    st.markdown(
        f"One query attends to three hours. With $d_k={ex['d_k']}$, raw scores a few times "
        f"$\\sqrt{{d_k}}={int(ex['d_k'] ** 0.5)}$ in size are, say, $(16, 8, 0)$, and the values are "
        "$v=(10, 20, 30)$ (one number per hour, for readability).")
    table = pd.DataFrame({
        "": ["scores", "softmax weights", "output Σ A·v", "largest softmax slope  p(1−p)"],
        "without scaling": [", ".join(f"{s:.1f}" for s in ex["raw"]),
                            ", ".join(f"{w:.4f}" for w in ex["weights_raw"]),
                            f"{ex['output_raw']:.3f}", f"{ex['max_slope_raw']:.5f}"],
        f"with 1/√{ex['d_k']} = 1/8": [", ".join(f"{s:.1f}" for s in ex["scaled"]),
                                       ", ".join(f"{w:.4f}" for w in ex["weights_scaled"]),
                                       f"{ex['output_scaled']:.3f}", f"{ex['max_slope_scaled']:.5f}"],
    })
    st.table(table.set_index(""))
    st.markdown(
        "Without scaling, nearly all weight lands on the first hour (output ≈ 10) and the softmax is "
        "saturated, so training barely sees a gradient through the other hours. With scaling the weights "
        "stay spread (≈ 0.67 / 0.24 / 0.09), so every hour can still be learned from.")

    st.subheader("Recurrent versus attention processing")
    st.table(pd.DataFrame({
        "": ["Order of work", "Path between hour 1 and hour 72", "Pairwise comparisons per window",
             "Parallel over time", "Built-in inspectable weights", "Parameters here"],
        "LSTM": ["72 sequential steps", "71 state updates", "none", "no", "no", "8,152"],
        "Self-attention": ["one pass over all hours", "1 attention step", "72 × 72 = 5,184", "yes",
                           "yes: the 72×72 matrix", "9,016"]}).set_index(""))
    st.info("Equal-looking parameter counts do not mean equal cost: attention scales with the square of the "
            "window length, the LSTM is sequential. Which one forecasts better here is an empirical "
            "question, answered on the *Baseline vs. Attention* tab from persisted runs.")

# ---------------------------------------------------------------------------
# 2. Try It
# ---------------------------------------------------------------------------
with try_tab:
    st.header("Try it")
    st.caption("Forecasts come from the registered models served by the API. A free-tier API that has been idle "
               "can take about a minute to wake; use *Check API status* if the first request times out.")
    client_api = api()
    if client_api is None:
        st.error("API_URL is not configured in Streamlit secrets (copy .streamlit/secrets.toml.example at the repository root to .streamlit/secrets.toml).")

    c1, c2 = st.columns([2, 1])
    with c2:
        model_type = st.radio("Model", fc.MODEL_TYPES, format_func=fc.MODEL_LABELS.get, key="model_type")
        st.caption("**Demonstration inputs** (synthetic, not measurements):")
        for kind, label in fc.DEMO_LABELS.items():
            if st.button(label, key=f"demo_{kind}"):
                st.session_state["history_text"] = fc.format_history(fc.demo_history(kind))
        if st.button("Check API status", disabled=client_api is None):
            try:
                with st.spinner("Contacting the API (a sleeping free-tier service can take ~1 minute)..."):
                    h = client_api.wait_until_up()
                st.markdown("\n".join(fc.health_summary(h)))
            except fc.ApiError as e:
                st.error(fc.describe_error(e))
    with c1:
        text = st.text_area("72 hourly temperatures in °C, oldest first (comma, space or newline separated)",
                            key="history_text", height=160)

    if st.button("Forecast the next 24 hours", type="primary", disabled=client_api is None):
        try:
            history = fc.parse_history(text)
        except ValueError as e:
            st.error(str(e))
            history = None
        if history is not None:
            try:
                with st.spinner("Asking the API..."):
                    pred = client_api.predict(model_type, history)
                st.session_state["last_prediction"] = (history, pred)
            except fc.ApiError as e:
                st.session_state.pop("last_prediction", None)
                (st.info if e.kind == "pending" else st.error)(fc.describe_error(e))

    if "last_prediction" in st.session_state:
        history, pred = st.session_state["last_prediction"]
        st.subheader(f"{fc.MODEL_LABELS[pred.model_type]} forecast")
        st.pyplot(fp.forecast_figure(history, pred.forecast_c), clear_figure=True)
        st.caption(f"Served by registered run {pred.served_by_run_id}; logged as prediction "
                   f"{pred.prediction_id} (input hash {pred.input_hash[:16]}…). A single value per hour: "
                   "there is no confidence estimate or prediction interval.")
        with st.expander("Forecast values"):
            st.dataframe(pd.DataFrame({"hour": range(1, 25), "forecast (°C)": pred.forecast_c}),
                         hide_index=True)
        if pred.attention is not None:
            st.subheader("Encoder self-attention")
            st.info("This matrix shows **input-to-input** relationships: how much the encoder's representation "
                    "of input hour *i* (row) draws on input hour *j* (column). It is **not** causal and **not** "
                    "an attribution of any forecast hour to input hours. Concentrated attention does not mean "
                    "a more reliable forecast.")
            st.pyplot(fp.attention_figure(pred.attention), clear_figure=True)
            with st.expander("Average attention each input hour receives"):
                st.pyplot(fp.mean_attention_profile_figure(pred.attention), clear_figure=True)
        else:
            st.caption("The LSTM has no attention weights.")

# ---------------------------------------------------------------------------
# 3. Baseline vs. Attention
# ---------------------------------------------------------------------------
with compare_tab:
    st.header("Baseline vs. self-attention")
    st.caption("Persisted runs, read from Supabase with the public anon key.")
    _client, _why = supabase_anon()
    runs = cached_runs("runs") if _client is not None else None
    if _client is None:
        st.warning(_why)
    elif not runs:
        st.info("No runs are registered yet. Registration needs the final test evaluation first, because "
                "`runs.test_mae_c` is required; nothing is shown here until real values exist.")
    else:
        st.markdown(
            "**Validation MAE** was used to pick each run's checkpoint, so it is optimistic for choosing "
            "between runs. **Test MAE** was computed once, after training, on data never used for training or "
            "selection; it is the figure to compare. Lower is better; both are in °C over all 24 hours.")
        st.dataframe(pd.DataFrame([{
            "run": r["id"], "model": r["model_type"], "seed": r["seed"], "data version": r["data_version"],
            "validation MAE (°C)": r["val_mae_c"], "test MAE (°C)": r["test_mae_c"],
            "epochs": (r.get("compute_budget") or {}).get("epochs_run"),
            "optimizer steps": (r.get("compute_budget") or {}).get("optimizer_steps"),
            "parameters": (r.get("compute_budget") or {}).get("parameter_count"),
            "created": r["created_at"]} for r in runs]), hide_index=True, use_container_width=True)
        st.subheader("Per-model summary over the runs shown")
        st.dataframe(pd.DataFrame(fc.summarize_runs(runs)), hide_index=True, use_container_width=True)
        st.caption("Mean and sample SD across the seeds present; with few seeds the SD is itself uncertain. "
                   "These tables show the numbers only. Whether a difference matters is for the written "
                   "analysis, not this page.")

    st.subheader("Illustrative examples")
    good = examples["comparison_examples"]
    if len(good) < 3:
        st.info(f"{len(good)} of 3 measured comparison examples are available. They will appear here once "
                "real examples are added to `ui/content/examples.json` (schema in `ui/content/README.md`); "
                "none have been invented.")
    for ex in good[:3]:
        st.markdown(f"**{ex['title']}** — origin {ex['origin']} (runs {ex['lstm_run_id']} / {ex['attention_run_id']})")
        st.pyplot(fp.comparison_figure(ex["history_c"], ex["actual_c"], ex["lstm_predicted_c"],
                                       ex["attention_predicted_c"]), clear_figure=True)
        if ex.get("attention") is not None:
            st.pyplot(fp.attention_figure(fc.parse_attention(ex["attention"]),
                                          title="Attention model (run %s): encoder self-attention for this input" % ex["attention_run_id"]),
                      clear_figure=True)
            st.caption("Input-to-input attention weights of the attention model for this 72 h history (row i = the hour "
                       "attending, column j = the hour attended to). They are not an attribution of any forecast hour to "
                       "input hours, not an explanation of the forecast, and not a confidence measure.")
            reading = fc.describe_attention_summary(ex.get("attention_summary"))
            if reading:
                st.markdown(reading)
        if ex.get("lstm_mae_c") is not None and ex.get("attention_mae_c") is not None:
            st.caption(f"MAE over these 24 hours: LSTM {ex['lstm_mae_c']:.2f} °C, attention {ex['attention_mae_c']:.2f} °C. "
                       "One validation window, not a ranking of the models. Artifact SHA-256 (prefix): "
                       f"LSTM {str(ex.get('lstm_artifact_sha256', ''))[:12]}, attention {str(ex.get('attention_artifact_sha256', ''))[:12]}.")
        st.markdown(ex["discussion"])
    for p in examples["problems"]:
        st.caption(f"⚠️ {p}")

# ---------------------------------------------------------------------------
# 4. Recent Predictions
# ---------------------------------------------------------------------------
with recent_tab:
    st.header("Recent predictions")
    st.caption("Read directly from Supabase with the public anon key. Inputs are not stored, only a hash. "
               "This table is publicly readable.")
    c1, c2, c3, c4 = st.columns(4)
    sort_by = c1.selectbox("Sort by", list(fc.SORTABLE), format_func=fc.SORTABLE.get)
    desc = c2.radio("Order", ["Descending", "Ascending"], horizontal=True) == "Descending"
    page_size = c3.selectbox("Rows per page", [10, 25, 50, 100], index=1)
    page = int(c4.number_input("Page", min_value=1, value=1, step=1))
    rows = db_call(fc.fetch_predictions, sort_by, desc, page_size, (page - 1) * page_size)
    if rows is not None:
        if not rows:
            st.info("No predictions to show on this page. Predictions appear after the API has served and "
                    "logged forecasts.")
        else:
            st.dataframe(pd.DataFrame([{
                "id": r["id"], "time": r["created_at"], "model": r["model_type"], "run": r["served_by_run_id"],
                "input hash": r["input_hash"][:12] + "…",
                "forecast +1 h (°C)": r["forecast"][0], "forecast +24 h (°C)": r["forecast"][-1]} for r in rows]),
                hide_index=True, use_container_width=True)
            by_id = {r["id"]: r for r in rows}
            chosen = st.selectbox("Inspect a prediction", list(by_id),
                                  format_func=lambda i: f"#{i} — {by_id[i]['model_type']} — {by_id[i]['created_at']}")
            sel = by_id[chosen]
            st.pyplot(fp.forecast_figure(None, sel["forecast"], title=f"Stored forecast #{chosen}"),
                      clear_figure=True)
            if sel["model_type"] == "attention":
                if st.button("Load stored attention matrix"):      # fetched for this one row only
                    att = db_call(fc.fetch_attention, chosen)
                    if att is not None:
                        st.info("Input-to-input encoder attention (not causal, not per-output attribution).")
                        st.pyplot(fp.attention_figure(att, f"Stored attention, prediction #{chosen}"),
                                  clear_figure=True)
            else:
                st.caption("LSTM predictions have no attention matrix.")

# ---------------------------------------------------------------------------
# 5. Failure Gallery
# ---------------------------------------------------------------------------
with failure_tab:
    st.header("Failure gallery")
    st.markdown("Cases where a model's forecast was far from what actually happened, with discussion. "
                "These models give **no calibrated confidence**; a forecast being wrong says nothing about how "
                "sure the model looked, and concentrated attention is not confidence.")
    st.warning("This gallery deliberately shows the attention model's largest validation errors (selection rule in "
               "`docs/forecasting-examples.md`). It does not represent average performance, and the examples were "
               "selected by error size, not drawn at random. Because neither model displays uncertainty, these are "
               "large errors from a model that shows no uncertainty, not measured overconfidence.")
    cases = examples["failure_cases"]
    if len(cases) < 3:
        st.info(f"Pending: {len(cases)} of 3 documented failures. Cases are added only after they are measured "
                "from a persisted run's forecasts against the actual temperatures (schema in "
                "`ui/content/README.md`). None have been invented.")
    for c in cases[:3]:
        st.subheader(c["title"])
        st.caption(f"{fc.MODEL_LABELS[c['model_type']]}, run {c['run_id']}, forecast origin {c['origin']}")
        st.pyplot(fp.forecast_figure(c["history_c"], c["predicted_c"], c["actual_c"]), clear_figure=True)
        errs = [abs(a - p) for a, p in zip(c["actual_c"], c["predicted_c"])]
        st.caption(f"Mean absolute error for this window: {sum(errs) / len(errs):.2f} °C; "
                   f"largest hourly error: {max(errs):.2f} °C.")
        if c.get("lstm_mae_c") is not None:
            st.caption(f"LSTM (run {c.get('lstm_run_id')}) on the same window: {c['lstm_mae_c']:.2f} °C. "
                       f"Validation window; artifact SHA-256 (prefix) {str(c.get('artifact_sha256', ''))[:12]}.")
        if c.get("attention") is not None:
            st.pyplot(fp.attention_figure(fc.parse_attention(c["attention"])), clear_figure=True)
            st.caption("Input-to-input attention; not an explanation of the error.")
        st.markdown(c["discussion"])
    for p in examples["problems"]:
        st.caption(f"⚠️ {p}")

# ---------------------------------------------------------------------------
# 6. Model Card
# ---------------------------------------------------------------------------
with card_tab:
    st.header("Model card")
    card = HERE.parent / "MODEL_CARD.md"
    if card.exists():
        st.markdown(card.read_text(encoding="utf-8"))
    else:
        st.warning("MODEL_CARD.md not found.")
