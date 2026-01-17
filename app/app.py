from __future__ import annotations

import streamlit as st

# =========================================================
# Page config (must be first Streamlit command)
# =========================================================
st.set_page_config(
    page_title="VERO Risk Calculator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# IMMEDIATE ROUTING
# This file should NOT behave like a page.
# It simply redirects users to Patient Input.
# =========================================================
st.switch_page("app/pages/1_Patient_Input.py")
