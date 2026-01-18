from __future__ import annotations

from pathlib import Path
import base64
import streamlit as st


# =========================================================
# Page config (MUST be the first Streamlit command)
# =========================================================
st.set_page_config(
    page_title="VERO Risk Calculator",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# Paths / assets
# =========================================================
HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
LOGO_IMG = ASSETS / "bioergotech_logo.png"


# =========================================================
# Helpers
# =========================================================
def _read_as_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _render_logo(path: Path) -> None:
    if not path.exists():
        return
    # Base64 embed avoids occasional relative-path issues on Streamlit Cloud
    b64 = _read_as_base64(path)
    st.markdown(
        f"""
        <div style="padding: 6px 0 14px 0;">
            <img src="data:image/png;base64,{b64}" style="width:100%; max-width: 240px; height:auto;" />
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# Global CSS: strict black & white theme + hide Streamlit footer
# =========================================================
st.markdown(
    """
    <style>
    /* Hard force black/white across app */
    html, body, .stApp {
        background: #ffffff !important;
        color: #000000 !important;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
                     Roboto, Helvetica, Arial, sans-serif;
        font-size: 18px !important;
    }

    /* Main content container */
    .block-container {
        padding-top: 0.9rem !important;
        padding-bottom: 1.8rem !important;
        max-width: 100% !important;
        background: #ffffff !important;
    }

    /* Sidebar: white background, no gradient */
    section[data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid rgba(0,0,0,0.12) !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.2rem !important;
        background: #ffffff !important;
    }

    /* Sidebar font sizing */
    section[data-testid="stSidebar"] * {
        font-size: 19px !important;
        line-height: 1.55 !important;
        color: #000000 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] a {
        font-size: 20px !important;
        font-weight: 650 !important;
        color: #000000 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] a[aria-current="page"] {
        text-decoration: underline !important;
    }

    /* Simple sidebar banner (black/white) */
    .sidebar-banner {
        background: #ffffff !important;
        border: 1px solid rgba(0,0,0,0.15) !important;
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 16px;
    }
    .sidebar-banner h3 {
        margin: 0 0 6px 0;
        font-size: 1.35rem;
        font-weight: 800;
        color: #000000 !important;
    }
    .sidebar-banner p {
        margin: 0;
        font-size: 1.05rem;
        color: rgba(0,0,0,0.70) !important;
    }

    /* Hide the first nav item (main script entry) */
    section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] ul li:first-child {
        display: none !important;
    }

    /* Hide Streamlit footer */
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Sidebar content
# =========================================================
with st.sidebar:
    _render_logo(LOGO_IMG)

    st.markdown(
        """
        <div class="sidebar-banner">
            <h3>VERO Risk Calculator</h3>
            <p>
                Use the pages below:
                <br>• <b>Patient Input</b>
                <br>• <b>Results &amp; Visual Analytics</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Start with Patient Input, then review Results & Visual Analytics.")


# =========================================================
# Redirect to Patient Input (removes need for a visible Home page)
# =========================================================
st.switch_page("pages/1_Patient_Input.py")
