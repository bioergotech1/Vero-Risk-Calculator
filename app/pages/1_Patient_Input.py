from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------------------
# Path bootstrap (do this early)
# ------------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parents[1]  # app/
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from vero_engine import VEROEngine  # noqa: E402
from ui_theme import apply_bioergotech_theme  # noqa: E402

# ------------------------------------------------------------------------------
# Streamlit config (ONLY ONCE)
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="VERO - Patient Input",
    layout="wide",
    initial_sidebar_state="expanded",
)

ASSETS = Path(__file__).resolve().parents[1] / "assets"
apply_bioergotech_theme(
    assets_dir=ASSETS,
    sidebar_title="VERO Risk Calculator",
    sidebar_subtitle="Patient Input",
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
      .card {
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 16px;
        padding: 16px 16px 10px 16px;
        background: rgba(255,255,255,0.9);
        margin-bottom: 0.8rem;
      }
      .card h3 {
        margin: 0 0 10px 0;
        font-size: 1.05rem;
      }
      .small-note {
        color: rgba(0,0,0,0.62);
        font-size: 0.95rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# Constants & helpers
# ------------------------------------------------------------------------------
LEAVE_BLANK_LABEL = "(leave blank)"
MISSING_LABELS = {"Not Known / Missing", "Missing / Not Noted"}

# Timeline columns (STRICT, per instruction)
TIMELINE_COLS = [
    "observation_start_date",
    "observation_end_date",
    "tumor_diagnosis_date",
    "Oncology Unit Intake Date",
    "surgery_date",
    "radiotherapy_start_date",
    "radiotherapy_end_date",
]


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except Exception:
        return False


def _clean_scalar(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    return v


def _as_text(v: Any) -> str:
    if _is_missing(v):
        return ""
    return str(_clean_scalar(v))


def pretty_label(code: str) -> str:
    return str(code).replace("_", " ").strip().title()


def parse_numeric(raw: str) -> Optional[float]:
    raw = raw.strip()
    if raw == "":
        return None
    try:
        return float(raw) if "." in raw else int(raw)
    except Exception:
        return None


def derive_age_group(age_value: Any) -> Optional[str]:
    if _is_missing(age_value):
        return None
    try:
        a = float(age_value)
    except Exception:
        return None
    return "<= 65 years" if a <= 65 else "> 65 years"


# ------------------------------------------------------------------------------
# Engine & data loading
# ------------------------------------------------------------------------------
HERE = Path(__file__).resolve().parents[1]
BASE_MODEL = HERE / "vero_base_model_prefit.joblib"
CALIBRATOR = HERE / "vero_calibrator_prefit.joblib"
META = HERE / "vero_metadata.json"
DEFAULT_DATA_PATH = HERE / "data" / "codige_master_clean__v2.xlsx"


@st.cache_resource
def load_engine() -> VEROEngine:
    return VEROEngine(BASE_MODEL, CALIBRATOR, META)


@st.cache_data
def load_master_df(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ------------------------------------------------------------------------------
# Widget key strategy
# ------------------------------------------------------------------------------
def wkey(col: str) -> str:
    return f"w__{col}"


def seed_widgets_from_patient(
    base_patient: Dict[str, Any],
    feature_cols: List[str],
    numeric_cols: set,
    levels: Dict[str, List[str]],
) -> None:
    for col in feature_cols:
        if col == "age_group":
            continue

        wk = wkey(col)

        if col in numeric_cols:
            st.session_state[wk] = _as_text(base_patient.get(col))
        elif col in levels and levels[col]:
            opts = [LEAVE_BLANK_LABEL] + levels[col]
            v = base_patient.get(col)
            st.session_state[wk] = (
                LEAVE_BLANK_LABEL if _is_missing(v) or str(v) not in opts else str(v)
            )
        else:
            st.session_state[wk] = _as_text(base_patient.get(col))


def input_widget(col: str, numeric_cols: set, levels: Dict[str, List[str]]) -> Any:
    label = pretty_label(col)
    wk = wkey(col)

    if col in numeric_cols:
        raw = st.text_input(label, key=wk, placeholder="Leave blank if unknown")
        v = parse_numeric(raw)
        if raw and v is None:
            st.warning(f"{label} expects a number.")
        return v

    if col in levels and levels[col]:
        pick = st.selectbox(label, [LEAVE_BLANK_LABEL] + levels[col], key=wk)
        return None if pick == LEAVE_BLANK_LABEL else pick

    raw = st.text_input(label, key=wk)
    return None if raw.strip() == "" else raw.strip()


# ------------------------------------------------------------------------------
# Layout definitions
# ------------------------------------------------------------------------------
CATEGORIES: Dict[str, List[str]] = {
    "Demographics & Socio-economic": [
        "age", "gender", "ethnicity", "education_level",
        "employment_status", "alcohol_consumption", "smoking_status_detail",
    ],
    "Tumor & Molecular Context": [
        "tumor_type", "molecular_alterations", "mutations_present", "genotipo_DPYD_type",
    ],
    "Treatment Exposure (Baseline)": [
        "surgical_intervention", "radiotherapy_status", "received_chemo",
        "received_targeted_therapy", "oncology_treatment_lines_n",
        "n_treatment_lines", "max_combo_regimen_size",
        "total_chemo_cycles", "treatment_duration_days",
    ],
    "Comorbidities & Clinical Conditions": [
        "hypertension", "dyslipidemia", "ischemic_heart_disease",
        "atrial_fibrillation", "diabete_tipo_II", "obesity_comorbidity",
        "copd", "asthma", "renal_insufficiency", "anemia_comorbidity",
        "psychiatric_disorders", "cardiovascular_disorders",
    ],
    "Frailty & Burden Indices": [
        "cci_score", "IPB", "farmaci_cat_n", "total_unique_active_drugs",
    ],
    "Laboratory Ranges": [
        "white_blood_cells_range", "hemoglobin_range",
        "neutrophils_percent_range", "platelet_count_range",
        "creatinine_range", "ast_got_range", "alt_gpt_range",
    ],
    "ADR Summary": ["adr_n_tot"],
}

# ------------------------------------------------------------------------------
# Page content
# ------------------------------------------------------------------------------
st.title("Patient Input")

engine = load_engine()
FEATURE_COLS = engine.feature_cols
NUM_COLS = set(engine.meta.get("numeric_cols", []))
CAT_COLS = list(engine.meta.get("categorical_cols", []))

st.markdown('<div class="section-title">Data source</div>', unsafe_allow_html=True)
left, right = st.columns([0.65, 0.35])

with left:
    upload = st.file_uploader("Upload master Excel (optional)", type=["xlsx"])

with right:
    st.caption("Default dataset:")
    st.code(str(DEFAULT_DATA_PATH), language="text")

df_master: Optional[pd.DataFrame] = None
if upload:
    df_master = pd.read_excel(BytesIO(upload.getvalue()))
else:
    df_master = load_master_df(DEFAULT_DATA_PATH)

if "patient_id" not in df_master.columns:
    st.error("patient_id column missing in dataset.")
    st.stop()

levels = {
    c: sorted(df_master[c].dropna().astype(str).unique().tolist())
    for c in CAT_COLS if c in df_master.columns
}

st.divider()

st.markdown('<div class="section-title">Select patient (auto-fill)</div>', unsafe_allow_html=True)
ids = sorted(df_master["patient_id"].astype(str).unique().tolist())
sel = st.selectbox("patient_id", ["(none)"] + ids)

base_patient = {c: None for c in FEATURE_COLS}
timeline_record: Dict[str, Any] = {}

if sel != "(none)":
    row = df_master.loc[df_master["patient_id"].astype(str) == sel].iloc[0]
    base_patient = {c: row.get(c, None) for c in FEATURE_COLS}
    timeline_record = {c: row.get(c, None) for c in TIMELINE_COLS}

    if st.session_state.get("_active_patient_id") != sel:
        st.session_state["_active_patient_id"] = sel
        seed_widgets_from_patient(base_patient, FEATURE_COLS, NUM_COLS, levels)

    st.success(f"Loaded patient {sel}")

st.session_state["patient_record"] = dict(base_patient)
st.session_state["timeline_record"] = dict(timeline_record)
st.session_state["selected_patient_id"] = None if sel == "(none)" else sel

st.divider()

st.markdown('<div class="section-title">Inputs (grouped)</div>', unsafe_allow_html=True)
st.caption("Leave any field blank if unknown.")

with st.form("patient_form"):
    updated: Dict[str, Any] = {}

    for cat, cols in CATEGORIES.items():
        st.markdown(f'<div class="card"><h3>{cat}</h3>', unsafe_allow_html=True)
        cols_ui = st.columns(3)
        for i, col in enumerate(cols):
            if col in FEATURE_COLS:
                with cols_ui[i % 3]:
                    updated[col] = input_widget(col, NUM_COLS, levels)
        st.markdown("</div>", unsafe_allow_html=True)

    submitted = st.form_submit_button("Save inputs for scoring", use_container_width=True)

if submitted:
    merged = dict(base_patient)
    merged.update(updated)
    merged["age_group"] = derive_age_group(merged.get("age"))

    st.session_state["patient_record"] = merged
    st.success("Saved. Proceed to Results & Visual Analytics.")
