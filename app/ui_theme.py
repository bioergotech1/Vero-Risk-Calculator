from __future__ import annotations

from pathlib import Path
import base64
import streamlit as st


PRIMARY = "#13d6b0"
PRIMARY_DARK = "#0eb093"
TEXT = "#1f2937"
MUTED = "rgba(31,41,55,0.78)"
BORDER = "rgba(15,23,42,0.12)"


def _img_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def apply_bioergotech_theme(
    *,
    page_title: str = "VERO Risk Calculator",
    assets_dir: Path | None = None,
    logo_filename: str = "bioergotech_logo.png",
    sidebar_title: str = "VERO Risk Calculator",
    sidebar_subtitle: str = "Navigate using the pages below.",
    default_sidebar_caption: str = "Start with Patient Input, then review Results & Visual Analytics.",
) -> None:
    """
    Injects a consistent BioERGOtech theme and a styled sidebar on every page.
    Call this once near the top of EACH page file (after st.set_page_config).
    """

    if assets_dir is None:
        # assumes this file sits in app/ and assets is app/assets
        assets_dir = Path(__file__).resolve().parent / "assets"

    logo_path = assets_dir / logo_filename

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

        /* Main layout spacing */
        .block-container {{
            padding-top: 1rem !important;
            padding-bottom: 1.8rem !important;
            max-width: 1200px !important;
        }}

        /* Sidebar background + border */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(
                180deg,
                rgba(19,214,176,0.12),
                rgba(255,255,255,1) 55%
            );
            border-right: 1px solid var(--border);
        }}

        section[data-testid="stSidebar"] .block-container {{
            padding-top: 1.1rem !important;
        }}

        /* Increase ALL sidebar text slightly */
        section[data-testid="stSidebar"] * {{
            font-size: 19px !important;
            line-height: 1.55 !important;
        }}

        /* Sidebar page list (navigation) bigger + clearer */
        section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] span,
        section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] a {{
            font-size: 20px !important;
            font-weight: 700 !important;
        }}

        section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] a[aria-current="page"] {{
            color: var(--primary-dark) !important;
        }}

        /* Buttons (global) */
        .stButton>button {{
            border-radius: 12px !important;
            font-weight: 800 !important;
            border: 1px solid rgba(19,214,176,0.28) !important;
        }}

        /* Inputs (global) */
        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea {{
            border-radius: 10px !important;
        }}

        /* Sidebar banner */
        .sidebar-banner {{
            background: rgba(19,214,176,0.14);
            border: 1px solid rgba(19,214,176,0.28);
            border-radius: 16px;
            padding: 16px 16px;
            margin-bottom: 14px;
        }}

        .sidebar-banner h3 {{
            margin: 0 0 6px 0;
            font-size: 1.25rem;
            font-weight: 900;
        }}

        .sidebar-banner p {{
            margin: 0;
            color: var(--muted);
            font-size: 1.02rem;
        }}

        /* Slightly reduce top whitespace on pages that start with title */
        h1, h2, h3 {{
            letter-spacing: 0.2px;
        }}

        footer {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar UI (same everywhere)
    with st.sidebar:
        if logo_path.exists():
            st.image(str(logo_path), use_container_width=True)

        st.markdown(
            f"""
            <div class="sidebar-banner">
                <h3>{sidebar_title}</h3>
                <p>{sidebar_subtitle}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(default_sidebar_caption)
