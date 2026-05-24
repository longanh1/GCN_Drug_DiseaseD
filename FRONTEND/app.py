"""
PharmaLink GCN -- Navigation Entry Point
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

st.set_page_config(
    page_title="PharmaLink GCN",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.auth import render_auth_page, is_authenticated

# ── Register all pages FIRST so st.switch_page() works everywhere ───
pg = st.navigation(
    [
        st.Page("home.py",               title="Trang chu",        icon="🏠", default=True),
        st.Page("pages/1_prediction.py", title="Du doan & Phan tich", icon="🔬"),
        st.Page("pages/2_history.py",    title="Lich su",           icon="📋"),
        st.Page("pages/3_model_stages.py", title="Cac giai doan mo hinh goc", icon="🧬"),
        st.Page("pages/4_ablation.py",   title="Ablation Study",    icon="📊"),
        st.Page("pages/5_drug_generation.py", title="Sinh Thuoc Moi (VGAE)", icon="🧪"),
        st.Page("pages/6_admin_users.py", title="Quan ly tai khoan", icon="👥"),
    ],
    position="hidden",
)

# ── Guard: require login ────────────────────────────────────────────
if not is_authenticated():
    render_auth_page()
    st.stop()

pg.run()
