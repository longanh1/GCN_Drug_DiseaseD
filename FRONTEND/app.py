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

pg = st.navigation(
    [
        st.Page("home.py",               title="Trang chu",        icon="🏠", default=True),
        st.Page("pages/1_prediction.py", title="Du doan & Phan tich", icon="🔬"),
        st.Page("pages/2_history.py",    title="Lich su",           icon="📋"),
    ],
    position="hidden",
)
pg.run()
