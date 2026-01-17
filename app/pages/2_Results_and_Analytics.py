from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ------------------------------------------------------------------------------
# Path bootstrap (do this early)
# ------------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parents[1]  # app/
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from vero_engine import VEROEngine  # noqa: E402
from pdf_report import PDFInputs, build_vero_pdf  # noqa: E402
from ui_theme import apply_bioergotech_theme  # noqa: E402

# ------------------------------------------------------------------------------
# Streamlit config (ONLY ONCE)
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="VERO - Results & Visual Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

ASSETS = Path(__file__).resolve().parents[1] / "assets"
apply_bioergotech_theme(
    assets_dir=ASSETS,
    sidebar_title="VERO Risk Calculator",
    sidebar_subtitle="Results & Visual Analytics",
)

# ------------------------------------------------------------------------------
# Page-specific CSS (light, non-conflicting)
# ------------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .section-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: rgba(0,0,0,0.75);
        margin-top: 0.6rem;
      }
      .stButton > button {
        border-radius: 12px;
        height: 46px;
        font-weight: 800;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# Paths to artifacts
# ------------------------------------------------------------------------------
HERE = Path(__file__).resolve().parents[1]
BASE_MODEL = HERE / "vero_base_model_prefit.joblib"
CALIBRATOR = HERE / "vero_calibrator_prefit.joblib"
META = HERE / "vero_metadata.json"

# ------------------------------------------------------------------------------
# Timeline columns (STRICT, per your instruction)
# ------------------------------------------------------------------------------
TIMELINE_COLS = [
    "observation_start_date",
    "observation_end_date",
    "tumor_diagnosis_date",
    "Oncology Unit Intake Date",
    "surgery_date",
    "radiotherapy_start_date",
    "radiotherapy_end_date",
]

TIMELINE_LABELS = {
    "observation_start_date": "Observation start",
    "observation_end_date": "Observation end",
    "tumor_diagnosis_date": "Tumor diagnosis",
    "Oncology Unit Intake Date": "Oncology unit intake",
    "surgery_date": "Surgery",
    "radiotherapy_start_date": "Radiotherapy start",
    "radiotherapy_end_date": "Radiotherapy end",
}

# ------------------------------------------------------------------------------
# Engine loader
# ------------------------------------------------------------------------------
@st.cache_resource
def load_engine() -> VEROEngine:
    return VEROEngine(BASE_MODEL, CALIBRATOR, META)

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def pretty_label(code: str) -> str:
    return str(code).replace("_", " ").strip().title()

def derive_age_group(age_value: Any) -> Optional[str]:
    if age_value is None or age_value == "":
        return None
    try:
        a = float(age_value)
    except Exception:
        return None
    return "<= 65 years" if a <= 65 else "> 65 years"

def clip_for_display(prob: float) -> float:
    eps = 1e-6
    p = float(prob)
    return float(min(max(p, eps), 1 - eps))

def fmt_prob(p: float) -> str:
    try:
        p = float(p)
    except Exception:
        return "-"
    if np.isnan(p):
        return "-"
    if p == 0.0:
        return "≈ 0 (very small)"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.6f}"

def risk_badge(stratum: str) -> str:
    color = {"Low": "#2e7d32", "Medium": "#ed6c02", "High": "#d32f2f"}.get(stratum, "#444")
    return f"""
    <div style="display:inline-block;padding:8px 14px;border-radius:999px;
                background:{color};color:white;font-weight:900;">
        {stratum} Risk
    </div>
    """

def safe_to_date(x: Any) -> Optional[date]:
    """
    Convert input to python date safely.
    Prevents numeric values being interpreted as epoch dates.
    """
    if x is None or x is pd.NaT:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    if isinstance(x, pd.Timestamp):
        return None if pd.isna(x) else x.date()

    if isinstance(x, date):
        return x

    if isinstance(x, (int, float, np.integer, np.floating)):
        return None

    s = str(x).strip()
    if s == "":
        return None

    dt = pd.to_datetime(s, errors="coerce")
    if pd.isna(dt):
        return None
    return dt.date()

def build_timeline_events(
    timeline_record: Dict[str, Any],
    patient_record: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    STRICT timeline builder:
    - only uses TIMELINE_COLS
    - pulls from timeline_record first, then patient_record
    """
    primary = timeline_record or {}
    fallback = patient_record or {}

    events: List[Dict[str, Any]] = []
    for col in TIMELINE_COLS:
        v = primary.get(col, None)
        if v is None:
            v = fallback.get(col, None)
        d = safe_to_date(v)
        if d is None:
            continue
        events.append(
            {"event": TIMELINE_LABELS.get(col, pretty_label(col)), "date": d}
        )

    events.sort(key=lambda x: x["date"])
    return events

def make_probability_gauge(prob_display: float, title: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(prob_display),
            number={"valueformat": ".4f"},
            gauge={"axis": {"range": [0, 1]}},
            title={"text": title},
        )
    )
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=50, b=10))
    return fig

def make_timeline_figure(events: List[Dict[str, Any]]) -> go.Figure:
    ev_df = pd.DataFrame(events).dropna(subset=["date"]).sort_values("date")
    if ev_df.empty:
        return go.Figure()

    lanes = min(4, max(1, len(ev_df)))
    y = [i % lanes for i in range(len(ev_df))]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ev_df["date"],
            y=y,
            mode="markers+text",
            text=ev_df["event"],
            textposition="top center",
            marker=dict(size=10),
            hovertemplate="<b>%{text}</b><br>%{x|%Y-%m-%d}<extra></extra>",
        )
    )

    fig.add_shape(
        type="line",
        x0=min(ev_df["date"]),
        x1=max(ev_df["date"]),
        y0=(lanes - 1) / 2,
        y1=(lanes - 1) / 2,
        line=dict(width=2),
    )

    fig.update_yaxes(visible=False, range=[-0.8, lanes - 0.2])
    fig.update_xaxes(tickformat="%b %Y", tickangle=-25, showgrid=True, title="Date")
    fig.update_layout(height=340, margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
    return fig

# ------------------------------------------------------------------------------
# Page header + sidebar utilities
# ------------------------------------------------------------------------------
st.title("Results & Visual Analytics")

with st.sidebar:
    st.markdown("### Utilities")
    if st.button("Reload engine (clear cache)", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

# ------------------------------------------------------------------------------
# Load engine + session input
# ------------------------------------------------------------------------------
engine = load_engine()
FEATURE_COLS = engine.feature_cols

if "patient_record" not in st.session_state:
    st.warning("No inputs found yet. Go to Patient Input first and save inputs.")
    st.stop()

patient: Dict[str, Any] = dict(st.session_state["patient_record"])
selected_id = st.session_state.get("selected_patient_id", None)
timeline_record = st.session_state.get("timeline_record", {}) or {}

if "age_group" in FEATURE_COLS:
    patient["age_group"] = derive_age_group(patient.get("age"))

age_val = patient.get("age")
age_group_val = derive_age_group(age_val)
gender_val = patient.get("gender")
eth_val = patient.get("ethnicity")
edu_val = patient.get("education_level")
emp_val = patient.get("employment_status")

# ------------------------------------------------------------------------------
# Patient summary
# ------------------------------------------------------------------------------
with st.expander("Patient summary (display only)", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patient ID", selected_id or "(manual)")
    c2.metric("Gender", gender_val or "-")
    c3.metric("Age", "-" if age_val in [None, ""] else str(age_val))
    c4.metric("Age group", age_group_val or "-")

    demo_table = pd.DataFrame(
        [
            ("Ethnicity", eth_val or "-"),
            ("Education level", edu_val or "-"),
            ("Employment status", emp_val or "-"),
        ],
        columns=["Field", "Value"],
    )
    st.dataframe(demo_table, use_container_width=True, hide_index=True)

st.divider()

# ------------------------------------------------------------------------------
# Compute
# ------------------------------------------------------------------------------
compute = st.button("Compute VERO", type="primary", use_container_width=True)
if not compute:
    st.info("Click Compute VERO to generate score, contributors, timeline, and PDF.")
    st.stop()

# ------------------------------------------------------------------------------
# Predict
# ------------------------------------------------------------------------------
try:
    res = engine.predict_single(patient)
except Exception as e:
    st.error(f"Scoring failed: {e}")
    st.stop()

p_base = float(res.base_probability)
p_cal = float(res.calibrated_probability)
p_score = float(res.probability_used_for_score)

score = int(res.vero_score)
stratum = str(res.risk_stratum)

# Display decisions
threshold = 0.5
screen_positive = p_score >= threshold
membership_label = "Accelerated aging / frailty" if screen_positive else "Non-accelerated / lower frailty"

p_score_disp = clip_for_display(p_score)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Probability used for score", fmt_prob(p_score))
m2.metric("Calibrated probability", fmt_prob(p_cal))
m3.metric("Base probability", fmt_prob(p_base))
m4.markdown(risk_badge(stratum), unsafe_allow_html=True)

st.divider()

# ------------------------------------------------------------------------------
# Membership panel
# ------------------------------------------------------------------------------
left, right = st.columns([0.55, 0.45], gap="large")
with left:
    st.subheader("Phenotype membership")
    st.write(f"**Decision rule:** positive if probability ≥ {threshold:.2f}")
    st.write(f"**Predicted membership:** {membership_label}")
    st.progress(p_score_disp)
    st.caption("Progress bar uses probability used for scoring (clipped only for UI).")

with right:
    gauge_fig = make_probability_gauge(p_score_disp, title="Membership probability")
    st.plotly_chart(gauge_fig, use_container_width=True)

st.divider()

# ------------------------------------------------------------------------------
# Timeline
# ------------------------------------------------------------------------------
st.subheader("Patient timeline")

events = build_timeline_events(timeline_record=timeline_record, patient_record=patient)

timeline_fig: Optional[go.Figure] = None
if not events:
    st.info("No usable timeline dates found for this patient.")
    with st.expander("Debug: timeline raw values", expanded=False):
        st.write({k: timeline_record.get(k, None) for k in TIMELINE_COLS})
else:
    timeline_fig = make_timeline_figure(events)
    st.plotly_chart(timeline_fig, use_container_width=True)
    st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)

st.divider()

# ------------------------------------------------------------------------------
# Contributors
# ------------------------------------------------------------------------------
TOP_K = 10
st.subheader(f"Top {TOP_K} contributors")

topk = engine.top_contributors_single(patient, top_k=TOP_K).copy()
topk["feature_code"] = topk["base_feature"]
topk["feature_label"] = topk["base_feature"].apply(pretty_label)
topk = topk[["feature_code", "feature_label", "total_contribution"]].reset_index(drop=True)

st.dataframe(topk, use_container_width=True, hide_index=True)

contrib_fig = px.bar(
    topk.sort_values("total_contribution"),
    x="total_contribution",
    y="feature_label",
    orientation="h",
    title=f"Top {TOP_K} contributors (signed contribution)",
)
contrib_fig.update_layout(height=max(420, 70 + 32 * len(topk)))
st.plotly_chart(contrib_fig, use_container_width=True)

st.divider()

# ------------------------------------------------------------------------------
# Export PDF
# ------------------------------------------------------------------------------
st.subheader("Export")
notes = st.text_area("Clinical notes (optional, for PDF)", value="", height=95)

# Convert charts to PNG for PDF (requires kaleido)
try:
    contrib_png = pio.to_image(contrib_fig, format="png", scale=2)
    gauge_png = pio.to_image(gauge_fig, format="png", scale=2)
    timeline_png = pio.to_image(timeline_fig, format="png", scale=2) if timeline_fig is not None else b""
except Exception:
    st.error("Chart rendering for PDF failed. Add kaleido to requirements.txt and redeploy.")
    contrib_png, gauge_png, timeline_png = b"", b"", b""

fields_provided = int(sum(1 for k in FEATURE_COLS if patient.get(k) not in [None, ""]))
fields_total = int(len(FEATURE_COLS))

pdf_inputs = PDFInputs(
    patient_id=selected_id or None,
    notes=notes.strip() or None,
    age_group=age_group_val,
    gender=gender_val,
    ethnicity=eth_val,
    education_level=edu_val,
    employment_status=emp_val,
    vero_probability=float(p_score),
    vero_probability_display=float(p_score_disp),
    vero_score=int(score),
    risk_stratum=stratum,
    fields_provided=fields_provided,
    fields_total=fields_total,
    phenotype_label=membership_label,
    phenotype_threshold=float(threshold),
)

pdf_bytes = build_vero_pdf(
    summary=pdf_inputs,
    top_contributors=topk,
    contrib_bar_png_bytes=contrib_png,
    gauge_png_bytes=gauge_png,
    timeline_png_bytes=timeline_png,
    timeline_events=events,
)

st.download_button(
    label="Download patient summary (PDF)",
    data=pdf_bytes,
    file_name=f"vero_summary_{selected_id or 'patient'}.pdf",
    mime="application/pdf",
    use_container_width=True,
)


pdf_bytes = build_vero_pdf(
    summary=pdf_inputs,
    top_contributors=topk,
    contrib_bar_png_bytes=contrib_png,
    gauge_png_bytes=gauge_png,
    timeline_png_bytes=timeline_png,
    timeline_events=events,
    assets_dir=ASSETS,   
)
