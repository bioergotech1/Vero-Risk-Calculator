from __future__ import annotations

from pathlib import Path
import base64
import streamlit as st


# -----------------------------
# Page config (must be first)
# -----------------------------
st.set_page_config(
    page_title="VERO Risk Calculator",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Paths / assets
# -----------------------------
HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
HERO_IMG = ASSETS / "vero_banner.png"
LOGO_IMG = ASSETS / "bioergotech_logo.png"


def img_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


# -----------------------------
# Theme tokens
# -----------------------------
PRIMARY = "#13d6b0"
PRIMARY_DARK = "#0eb093"
TEXT = "#1f2937"
MUTED = "rgba(31,41,55,0.75)"
BORDER = "rgba(15,23,42,0.12)"


# -----------------------------
# Global CSS
# - hides footer
# - styles sidebar
# - hides the MAIN page ("app") from the sidebar nav
# -----------------------------
st.markdown(
    f"""
    <style>
    :root {{
        --primary: {PRIMARY};
        --primary-dark: {PRIMARY_DARK};
        --text: {TEXT};
        --muted: {MUTED};
        --border: {BORDER};
    }}

    html, body, [class*="css"] {{
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
                     Roboto, Helvetica, Arial, sans-serif;
        font-size: 18px !important;
        color: var(--text);
    }}

    .block-container {{
        padding-top: 0.9rem !important;
        padding-bottom: 1.8rem !important;
        max-width: 100% !important;
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(
            180deg,
            rgba(19,214,176,0.12),
            rgba(255,255,255,1) 55%
        );
        border-right: 1px solid var(--border);
    }}

    section[data-testid="stSidebar"] .block-container {{
        padding-top: 1.2rem !important;
    }}

    section[data-testid="stSidebar"] * {{
        font-size: 19px !important;
        line-height: 1.55 !important;
    }}

    section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] a {{
        font-size: 20px !important;
        font-weight: 650 !important;
    }}

    section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] a[aria-current="page"] {{
        color: var(--primary-dark) !important;
    }}

    section[data-testid="stSidebar"] .stCaption {{
        font-size: 18px !important;
        color: var(--muted) !important;
    }}

    .sidebar-banner {{
        background: rgba(19,214,176,0.14);
        border: 1px solid rgba(19,214,176,0.28);
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 16px;
    }}

    .sidebar-banner h3 {{
        margin: 0 0 6px 0;
        font-size: 1.35rem;
        font-weight: 800;
    }}

    .sidebar-banner p {{
        margin: 0;
        font-size: 1.05rem;
        color: var(--muted);
    }}

    /* Hide the first nav item (this is the main script: app.py -> "app") */
    section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] ul li:first-child {{
        display: none !important;
    }}

    footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Sidebar content
# -----------------------------
with st.sidebar:
    if LOGO_IMG.exists():
        st.image(str(LOGO_IMG), use_container_width=True)

    st.markdown(
        """
        <div class="sidebar-banner">
            <h3>VERO Risk Calculator</h3>
            <p>
                Use the pages below:
                <br>• <b>Patient Input</b>
                <br>• <b>Results & Visual Analytics</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Start with Patient Input, then review Results & Visual Analytics.")


# -----------------------------
# Immediately redirect users to Patient Input
# (Correct relative path: pages/... NOT app/pages/...)
# -----------------------------
st.switch_page("pages/1_Patient_Input.py")
