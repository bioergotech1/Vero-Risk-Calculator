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
    b64 = _read_as_base64(path)
    st.markdown(
        f"""
        <div style="padding: 6px 0 14px 0;">
            <img src="data:image/png;base64,{b64}"
                 style="width:100%; max-width: 240px; height:auto;" />
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# Global CSS: hard black/white + force sidebar nav styling
# =========================================================
st.markdown(
    """
    <style>
    /* Base */
    html, body, .stApp {
        background: #ffffff !important;
        color: #000000 !important;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
                     Roboto, Helvetica, Arial, sans-serif;
        font-size: 18px !important;
    }

    /* Main container */
    .block-container {
        padding-top: 0.9rem !important;
        padding-bottom: 1.8rem !important;
        max-width: 100% !important;
        background: #ffffff !important;
    }

    /* Sidebar wrapper */
    section[data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid rgba(0,0,0,0.20) !important;
    }

    section[data-testid="stSidebar"] .block-container {
        background: #ffffff !important;
        padding-top: 1.2rem !important;
    }

    /* Force all sidebar text black */
    section[data-testid="stSidebar"] * {
        color: #000000 !important;
        font-size: 18px !important;
        line-height: 1.55 !important;
        background: transparent !important;
    }

    /* Sidebar navigation container */
    section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] {
        background: #ffffff !important;
    }

    /* Nav list items */
    section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] ul,
    section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] li {
        background: #ffffff !important;
    }

    /* Nav links (normal) */
    section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] a {
        color: #000000 !important;
        background: #ffffff !important;
        border-radius: 10px !important;
        font-weight: 650 !important;
        padding: 10px 12px !important;
        margin: 2px 0 !important;
        text-decoration: none !important;
        border: 1px solid transparent !important;
    }

    /* Nav links hover */
    section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] a:hover {
        background: #f3f4f6 !important;
        border: 1px solid rgba(0,0,0,0.15) !important;
    }

    /* Active page (selected) */
    section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] a[aria-current="page"] {
        background: #000000 !important;
        color: #ffffff !important;
        border: 1px solid #000000 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] a[aria-current="page"] * {
        color: #ffffff !important;
    }

    /* Hide the first nav item ("app") */
    section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] ul li:first-child {
        display: none !important;
    }

    /* Simple sidebar banner */
    .sidebar-banner {
        background: #ffffff !important;
        border: 1px solid rgba(0,0,0,0.20) !important;
        border-radius: 16px !important;
        padding: 18px !important;
        margin-bottom: 16px !important;
    }
    .sidebar-banner h3 {
        margin: 0 0 6px 0 !important;
        font-size: 1.25rem !important;
        font-weight: 800 !important;
        color: #000000 !important;
    }
    .sidebar-banner p {
        margin: 0 !important;
        font-size: 1.0rem !important;
        color: rgba(0,0,0,0.75) !important;
    }

    /* Hide Streamlit footer */
    footer { visibility: hidden !important; }
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
                <br>• <b>Results &amp; Analytics</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Start with Patient Input, then review Results & Analytics.")

# =========================================================
# Redirect straight to Patient Input
# =========================================================
st.switch_page("pages/1_Patient_Input.py")
