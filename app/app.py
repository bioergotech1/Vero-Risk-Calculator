from __future__ import annotations

from pathlib import Path
import base64
import streamlit as st


# =========================================================
# Page config (MUST be first Streamlit command)
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
HERO_IMG = ASSETS / "vero_banner.png"
LOGO_IMG = ASSETS / "bioergotech_logo.png"  # optional


def _img_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


# =========================================================
# BioERGOtech theme tokens
# =========================================================
PRIMARY = "#13d6b0"
PRIMARY_DARK = "#0eb093"
TEXT = "#1f2937"
MUTED = "rgba(31,41,55,0.75)"
BORDER = "rgba(15,23,42,0.12)"


# =========================================================
# Global CSS (bigger sidebar fonts + no lower banners)
# =========================================================
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

    /* Base font */
    html, body, [class*="css"] {{
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
                     Roboto, Helvetica, Arial, sans-serif;
        font-size: 18px !important;
        color: var(--text);
    }}

    /* Main container spacing */
    .block-container {{
        padding-top: 0.9rem !important;
        padding-bottom: 1.8rem !important;
        max-width: 100% !important;
    }}

    /* Sidebar styling + BIGGER LEFT PANE FONTS */
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

    /* Increase sidebar text size everywhere */
    section[data-testid="stSidebar"] * {{
        font-size: 19px !important;
           line-height: 1.55 !important;
    }}

    /* Sidebar page list (navigation) - make it even bigger */
    section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] span,
    section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] a {{
        font-size: 20px !important;
        font-weight: 650 !important;
    }}

    /* Optional: make the active page feel clearer */
    section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] a[aria-current="page"] {{
        color: var(--primary-dark) !important;
    }}

    /* Small helper text in sidebar */
    section[data-testid="stSidebar"] .stCaption {{
        font-size: 18px !important;
        color: var(--muted) !important;
    }}

    /* Sidebar banner (kept, but readable and clean) */
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

    /* Hero section (fills main area nicely) */
    .vero-hero {{
        width: 100%;
        height: calc(100vh - 4.5rem);
        background-size: cover;
        background-repeat: no-repeat;
        background-position: top center;
        border-radius: 20px;
        border: 1px solid var(--border);
        overflow: hidden;
    }}

    @media (max-width: 900px) {{
        .vero-hero {{ height: 55vh; }}
        section[data-testid="stSidebar"] * {{
            font-size: 18px !important;
        }}
    }}

    footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Sidebar
# =========================================================
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


# =========================================================
# Main (Home) - hero only, NO banners/cards underneath
# =========================================================
if HERO_IMG.exists():
    hero_b64 = _img_to_base64(HERO_IMG)
    st.markdown(
        f"""
        <div class="vero-hero"
             style="background-image:url('data:image/png;base64,{hero_b64}');">
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.warning("Hero image not found. Place it at app/assets/vero_banner.png")
