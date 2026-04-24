"""
Page 2 — Lịch sử dự đoán
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import streamlit as st
from utils.api_client import get_history, clear_history

st.set_page_config(
    page_title="Lịch sử — PharmaLink",
    page_icon="📋",
    layout="wide",
)

st.markdown('<base href="/">', unsafe_allow_html=True)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*, body { font-family: 'Inter', sans-serif !important; }
[data-testid="stAppViewContainer"] {
    background: #f1f5f9;
}
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #1e293b;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
.stDeployButton { display: none; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #6366f1; border-radius: 4px; }

.sb-brand { padding: 24px 20px 20px; border-bottom: 1px solid rgba(99,102,241,0.25); margin-bottom: 8px; }
.sb-brand-icon {
    width: 42px; height: 42px; border-radius: 12px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 1.4rem; margin-bottom: 10px; box-shadow: 0 4px 15px rgba(99,102,241,0.4);
}
.sb-brand-name {
    font-size: 1rem; font-weight: 700;
    background: linear-gradient(90deg, #818cf8, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.sb-brand-sub { font-size: 0.72rem; color: #475569; }
.sb-nav-label {
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.08em;
    color: #475569; text-transform: uppercase; padding: 16px 20px 6px;
}

.page-header {
    display: flex; align-items: center; justify-content: space-between;
    background: linear-gradient(135deg, #eef2ff, #f8fafc);
    border: 1px solid rgba(99,102,241,0.25); border-radius: 16px;
    padding: 20px 28px; margin-bottom: 24px;
}
.page-header-left h2 {
    font-size: 1.4rem; font-weight: 800; margin: 0 0 4px 0;
    background: linear-gradient(90deg, #1e293b, #4338ca);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.page-header-left p { color: #475569; font-size: 0.8rem; margin: 0; }

.hist-stat-card {
    background: #ffffff;
    border-radius: 14px; padding: 18px 20px;
    border: 1px solid #1e293b; text-align: center;
    box-shadow: 0 2px 12px rgba(99,102,241,0.08);
}
.hist-stat-val {
    font-size: 1.6rem; font-weight: 800;
    background: linear-gradient(135deg, #1e293b, #4338ca);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hist-stat-lbl { font-size: 0.72rem; color: #475569; margin-top: 5px; font-weight: 500; }

.empty-state {
    background: #f8fafc;
    border-radius: 20px; padding: 80px 30px;
    border: 1px dashed rgba(99,102,241,0.3); text-align: center;
    margin-top: 24px;
}
.empty-icon  { font-size: 3.5rem; margin-bottom: 16px; opacity: 0.6; }
.empty-title { color: #1e293b; font-size: 1.1rem; font-weight: 600; margin-bottom: 8px; }
.empty-sub   { color: #475569; font-size: 0.85rem; }

/* Table styling */
[data-testid="stDataFrame"] {
    border-radius: 14px !important; overflow: hidden;
    border: 1px solid rgba(99,102,241,0.15) !important;
}
[data-testid="stDataFrame"] table { background: transparent !important; }
[data-testid="stDataFrame"] thead tr th {
    background: rgba(99,102,241,0.1) !important; color: #4f46e5 !important;
    font-size: 0.78rem !important; font-weight: 600 !important;
    border-bottom: 1px solid rgba(99,102,241,0.2) !important;
    letter-spacing: 0.04em !important; text-transform: uppercase !important;
}
[data-testid="stDataFrame"] tbody tr td {
    color: #475569 !important; font-size: 0.82rem !important;
    border-bottom: 1px solid rgba(255,255,255,0.03) !important;
}
[data-testid="stDataFrame"] tbody tr:hover td {
    background: rgba(99,102,241,0.06) !important; color: #1e293b !important;
}

.stButton > button {
    border-radius: 10px !important; font-weight: 600 !important;
    font-size: 0.85rem !important; transition: all 0.2s !important; border: none !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.35) !important;
}
.stButton > button:not([kind="primary"]) {
    background: rgba(239,68,68,0.1) !important;
    border: 1px solid rgba(239,68,68,0.3) !important; color: #fca5a5 !important;
}
.stDownloadButton > button {
    background: rgba(16,185,129,0.1) !important; border: 1px solid rgba(16,185,129,0.3) !important;
    color: #34d399 !important; border-radius: 8px !important;
}
[data-testid="stPageLink"] a {
    border-radius: 10px !important; padding: 10px 16px !important;
    color: #475569 !important; font-size: 0.88rem !important; font-weight: 500 !important;
}
[data-testid="stPageLink"] a:hover { background: rgba(99,102,241,0.12) !important; color: #c7d2fe !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-icon">🧬</div>
        <div class="sb-brand-name">PharmaLink GCN</div>
        <div class="sb-brand-sub">Drug-Disease AI Platform v2.0</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="sb-nav-label">Điều hướng</div>', unsafe_allow_html=True)
    if st.button("🏠  Tổng quan",          use_container_width=True): st.switch_page("home.py")
    if st.button("🔬  Dự đoán & Phân tích", use_container_width=True): st.switch_page("pages/1_prediction.py")
    if st.button("📋  Lịch sử",            use_container_width=True): st.switch_page("pages/2_history.py")

# ── Data ──────────────────────────────────────────────────────────────
history = get_history(100)
df = pd.DataFrame(history) if history else pd.DataFrame()

# ── Page header ────────────────────────────────────────────────────────
col_h, col_a = st.columns([3, 1])
with col_h:
    st.markdown("""
    <div class="page-header">
        <div class="page-header-left">
            <h2>📋 Lịch sử dự đoán</h2>
            <p>Tổng quan / Lịch sử · Tất cả phiên dự đoán đã thực hiện</p>
        </div>
    </div>""", unsafe_allow_html=True)
with col_a:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑  Xóa tất cả", key="clear_hist", use_container_width=True):
        clear_history()
        st.rerun()

# ── Stats ─────────────────────────────────────────────────────────────
total      = len(df) if not df.empty else 0
models_cnt = df["model"].nunique() if not df.empty and "model" in df.columns else 0
datasets   = df["dataset"].nunique() if not df.empty and "dataset" in df.columns else 0
last_time  = df["timestamp"].iloc[-1] if not df.empty and "timestamp" in df.columns else "—"

sc1, sc2, sc3, sc4 = st.columns(4)
for col, val, lbl in [
    (sc1, total,      "Lần dự đoán"),
    (sc2, models_cnt, "Mô hình dùng"),
    (sc3, datasets,   "Dataset"),
    (sc4, str(last_time)[:10] if last_time != "—" else "—", "Lần cuối"),
]:
    col.markdown(f"""
    <div class="hist-stat-card">
        <div class="hist-stat-val">{val}</div>
        <div class="hist-stat-lbl">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Table or empty ────────────────────────────────────────────────────
if df.empty:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">📋</div>
        <div class="empty-title">Chưa có lịch sử dự đoán</div>
        <div class="empty-sub">
            Thực hiện dự đoán ở trang <b>Dự đoán &amp; Phân tích</b> để bắt đầu lưu lịch sử.
        </div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("🔬  Đến trang Dự đoán", type="primary"):
        st.switch_page("pages/1_prediction.py")
else:
    display_cols = [c for c in ["timestamp","drug","direction","model","top_k","dataset","num_results"]
                    if c in df.columns]
    rename_map = {
        "timestamp":   "⏰ Thời gian",
        "drug":        "💊 Thuốc",
        "direction":   "↔ Hướng",
        "model":       "🤖 Mô hình",
        "top_k":       "Top K",
        "dataset":     "📂 Dataset",
        "num_results": "# Kết quả",
    }

    # Filter controls
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        if "model" in df.columns:
            model_filter = st.multiselect("Lọc theo mô hình", df["model"].unique().tolist(),
                                           default=df["model"].unique().tolist(),
                                           label_visibility="collapsed")
    with fc2:
        if "dataset" in df.columns:
            ds_filter = st.multiselect("Lọc theo dataset", df["dataset"].unique().tolist(),
                                        default=df["dataset"].unique().tolist(),
                                        label_visibility="collapsed")
    with fc3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            "⬇  Tải toàn bộ CSV",
            data=df[display_cols].rename(columns=rename_map).to_csv(index=False),
            file_name="prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Apply filters
    filtered_df = df.copy()
    if "model" in df.columns and model_filter:
        filtered_df = filtered_df[filtered_df["model"].isin(model_filter)]
    if "dataset" in df.columns and ds_filter:
        filtered_df = filtered_df[filtered_df["dataset"].isin(ds_filter)]

    st.markdown(f"""
    <div style="color:#4b5563;font-size:0.78rem;margin-bottom:10px;">
        Hiển thị <b style="color:#818cf8">{len(filtered_df)}</b> / {total} bản ghi
    </div>""", unsafe_allow_html=True)

    st.dataframe(
        filtered_df[display_cols].rename(columns=rename_map),
        use_container_width=True,
        hide_index=True,
        height=min(38 + len(filtered_df) * 36, 600),
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔬  Thực hiện dự đoán mới", type="primary"):
        st.switch_page("pages/1_prediction.py")

