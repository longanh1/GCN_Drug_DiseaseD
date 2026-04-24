"""
PharmaLink GCN — Drug-Disease Prediction Platform
Main entry: Home page
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import plotly.graph_objects as go
from utils.api_client import get_stats, get_global_stats, get_datasets, get_comparison
from utils.chart_utils import donut_chart

# ── Global CSS ─────────────────────────────────────────────────────────
st.markdown('<base href="/">', unsafe_allow_html=True)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & Base ── */
*, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stAppViewContainer"] {
    background: #f1f5f9;
    min-height: 100vh;
}
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0; }

/* ── Hide default decoration ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
.stDeployButton { display: none; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #6366f1; border-radius: 4px; }

/* ── Sidebar Brand ── */
.sb-brand {
    padding: 24px 20px 20px;
    border-bottom: 1px solid rgba(99,102,241,0.25);
    margin-bottom: 8px;
}
.sb-brand-icon {
    width: 42px; height: 42px; border-radius: 12px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem; margin-bottom: 10px;
    box-shadow: 0 4px 15px rgba(99,102,241,0.4);
}
.sb-brand-name {
    font-size: 1rem; font-weight: 700;
    background: linear-gradient(90deg, #818cf8, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.sb-brand-sub { font-size: 0.72rem; color: #475569; margin-top: 2px; }

/* ── Sidebar nav ── */
.sb-nav-label {
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.08em;
    color: #475569; text-transform: uppercase; padding: 16px 20px 6px;
}

/* ── Dataset pill ── */
.sb-dataset {
    margin: 0 12px; padding: 10px 14px;
    background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.2);
    border-radius: 10px; margin-bottom: 12px;
}
.sb-dataset-label { font-size: 0.7rem; color: #6b7280; margin-bottom: 6px; }

/* ── Hero ── */
.hero-wrap {
    position: relative; overflow: hidden;
    background: linear-gradient(135deg, #eef2ff 0%, #f0f4ff 40%, #f8fafc 100%);
    border-radius: 20px; padding: 40px 44px;
    border: 1px solid rgba(99,102,241,0.3);
    margin-bottom: 28px;
    box-shadow: 0 4px 24px rgba(99,102,241,0.1), inset 0 1px 0 rgba(255,255,255,0.8);
}
.hero-glow {
    position: absolute; top: -60px; right: -60px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%);
    border-radius: 50%; pointer-events: none;
}
.hero-glow2 {
    position: absolute; bottom: -80px; left: 30%;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(139,92,246,0.12) 0%, transparent 70%);
    border-radius: 50%; pointer-events: none;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.35);
    color: #4338ca; border-radius: 20px; padding: 5px 14px;
    font-size: 0.75rem; font-weight: 500; margin-bottom: 18px;
}
.hero-badge-dot {
    width: 6px; height: 6px; border-radius: 50%; background: #22c55e;
    animation: pulse-green 2s infinite;
}
@keyframes pulse-green {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(34,197,94,0.4); }
    50% { opacity: 0.7; box-shadow: 0 0 0 6px rgba(34,197,94,0); }
}
.hero-title {
    font-size: 2rem; font-weight: 800; line-height: 1.2;
    background: linear-gradient(90deg, #1e293b 0%, #4338ca 50%, #c084fc 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 10px 0;
}
.hero-sub {
    font-size: 0.92rem; color: #475569; line-height: 1.6; margin: 0;
    max-width: 640px;
}
.hero-sub b { color: #4f46e5; }
.hero-tech-tags {
    display: flex; gap: 8px; flex-wrap: wrap; margin-top: 22px;
}
.tech-tag {
    background: rgba(15,23,42,0.8); border: 1px solid rgba(99,102,241,0.25);
    color: #475569; border-radius: 6px; padding: 4px 12px; font-size: 0.75rem;
}

/* ── Stat cards ── */
.stat-grid { display: flex; gap: 14px; margin-bottom: 28px; }
.stat-card {
    flex: 1;
    background: #ffffff;
    border-radius: 16px; padding: 20px 18px;
    border: 1px solid #1e293b;
    text-align: center; position: relative; overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
    box-shadow: 0 2px 12px rgba(99,102,241,0.08);
}
.stat-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: var(--accent, linear-gradient(90deg, #6366f1, #8b5cf6));
    border-radius: 16px 16px 0 0;
}
.stat-card:hover { transform: translateY(-3px); border-color: rgba(99,102,241,0.3); }
.stat-icon { font-size: 1.4rem; margin-bottom: 8px; }
.stat-val {
    font-size: 1.7rem; font-weight: 800;
    background: linear-gradient(135deg, #1e293b, #4338ca);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1;
}
.stat-lbl { font-size: 0.75rem; color: #475569; margin-top: 6px; font-weight: 500; letter-spacing: 0.02em; }

/* ── Section header ── */
.section-header {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 16px;
}
.section-header-line {
    flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(99,102,241,0.3), transparent);
}
.section-title {
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em;
    color: #6366f1; text-transform: uppercase;
}

/* ── Model card ── */
.model-card {
    background: #ffffff;
    border-radius: 14px; padding: 16px 18px;
    border: 1px solid #1e293b;
    margin-bottom: 12px; position: relative; overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
}
.model-card:hover { border-color: rgba(99,102,241,0.4); transform: translateX(3px); }
.model-card-accent {
    position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    background: linear-gradient(180deg, #6366f1, #8b5cf6);
    border-radius: 14px 0 0 14px;
}
.model-card-header { display: flex; align-items: center; justify-content: space-between; }
.model-dot {
    width: 7px; height: 7px; border-radius: 50%; background: #22c55e;
    display: inline-block; margin-right: 8px;
    box-shadow: 0 0 6px rgba(34,197,94,0.6);
}
.model-name { color: #1e293b; font-weight: 600; font-size: 0.9rem; }
.model-auc {
    font-size: 0.8rem; font-weight: 700;
    background: linear-gradient(90deg, #818cf8, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.model-desc { color: #475569; font-size: 0.76rem; margin-top: 6px; }

/* ── Quick access ── */
.qa-card {
    background: #ffffff;
    border-radius: 12px; padding: 14px 16px;
    border: 1px solid #e8ecf0; margin-bottom: 10px;
    cursor: pointer; transition: all 0.2s; display: flex; align-items: center; gap: 12px;
}
.qa-card:hover { border-color: rgba(99,102,241,0.4); background: rgba(99,102,241,0.08); transform: translateX(4px); }
.qa-icon {
    width: 36px; height: 36px; border-radius: 10px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 1.1rem;
}
.qa-body .qa-title { color: #1e293b; font-weight: 600; font-size: 0.88rem; }
.qa-body .qa-desc  { color: #475569; font-size: 0.75rem; margin-top: 2px; }

/* ── Status pill ── */
.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.3);
    color: #4ade80; border-radius: 20px; padding: 5px 14px; font-size: 0.78rem; font-weight: 500;
}
.status-live {
    width: 6px; height: 6px; border-radius: 50%; background: #22c55e;
    animation: pulse-green 2s infinite;
}

/* ── Streamlit overrides ── */
.stButton > button {
    border-radius: 10px !important; font-weight: 600 !important;
    font-size: 0.85rem !important; transition: all 0.2s !important;
    border: none !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(99,102,241,0.5) !important; transform: translateY(-1px);
}
.stButton > button:not([kind="primary"]) {
    background: rgba(99,102,241,0.1) !important;
    border: 1px solid rgba(99,102,241,0.3) !important; color: #4338ca !important;
}
.stSelectbox > div > div {
    background: #f8fafc !important;
    border: 1px solid rgba(99,102,241,0.25) !important; border-radius: 10px !important;
    color: #1e293b !important;
}
[data-testid="stPageLink"] a {
    border-radius: 10px !important; padding: 10px 16px !important;
    transition: background 0.15s !important; font-weight: 500 !important;
    color: #475569 !important; font-size: 0.88rem !important;
}
[data-testid="stPageLink"] a:hover { background: rgba(99,102,241,0.12) !important; color: #c7d2fe !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────
if "dataset" not in st.session_state:
    st.session_state.dataset = "C-dataset"
if "username" not in st.session_state:
    st.session_state.username = "researcher"

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
    st.page_link("home.py",               label="🏠  Tổng quan",         help="Trang chủ")
    st.page_link("pages/1_prediction.py", label="🔬  Dự đoán & Phân tích")
    st.page_link("pages/2_history.py",    label="📋  Lịch sử dự đoán")

    st.markdown('<div class="sb-nav-label">Cấu hình</div>', unsafe_allow_html=True)
    datasets = get_datasets()
    if datasets:
        st.markdown('<div class="sb-dataset"><div class="sb-dataset-label">Dataset đang dùng</div>', unsafe_allow_html=True)
        chosen = st.selectbox("", datasets, label_visibility="collapsed",
                               index=datasets.index("C-dataset") if "C-dataset" in datasets else 0)
        st.session_state.dataset = chosen
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    uname = st.session_state.username
    st.markdown(f"""
    <div style="padding:10px 16px;display:flex;align-items:center;gap:10px;">
        <div style="width:36px;height:36px;border-radius:50%;
                    background:linear-gradient(135deg,#6366f1,#8b5cf6);
                    display:flex;align-items:center;justify-content:center;
                    font-weight:700;color:#fff;font-size:0.9rem;
                    box-shadow:0 3px 10px rgba(99,102,241,0.4);">
            {uname[0].upper()}
        </div>
        <div>
            <div style="color:#1e293b;font-size:0.85rem;font-weight:600;">{uname}</div>
            <div style="color:#4b5563;font-size:0.72rem;">Researcher</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Đăng xuất", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ── Data ──────────────────────────────────────────────────────────────
dataset  = st.session_state.dataset
stats    = get_stats(dataset)
gstats   = get_global_stats()
cmp      = get_comparison(dataset)

s_drugs    = stats.get("num_drugs", 0)
s_diseases = stats.get("num_diseases", 0)
s_proteins = stats.get("num_proteins", 0)
s_links    = stats.get("num_known_links", 0)
s_models   = stats.get("num_models", 2)
best_auc   = stats.get("best_auc")
models_data = cmp.get("models", {})
auc_gcn   = models_data.get("AMNTDDA", {}).get("AUC_mean")
auc_fuzzy = models_data.get("AMNTDDA_Fuzzy", {}).get("AUC_mean")

# ── Hero ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-glow"></div>
    <div class="hero-glow2"></div>
    <div class="hero-badge">
        <span class="hero-badge-dot"></span>
        Hệ thống hoạt động · Dataset: {dataset}
    </div>
    <h1 class="hero-title">PharmaLink GCN</h1>
    <p class="hero-sub">
        Nền tảng dự đoán liên kết <b>Thuốc–Bệnh</b> sử dụng
        <b>Graph Neural Network AMNTDDA</b> kết hợp
        <b>Hệ suy luận mờ Mamdani</b> và phân tích cấu trúc topo
        để khám phá các liên kết tiềm năng chưa được biết đến.
    </p>
    <div class="hero-tech-tags">
        <span class="tech-tag">GCN · AMNTDDA</span>
        <span class="tech-tag">Fuzzy Mamdani FIS</span>
        <span class="tech-tag">Topological Analysis</span>
        <span class="tech-tag">10-fold CV</span>
        <span class="tech-tag">CUDA GPU</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Action buttons ────────────────────────────────────────────────────
cb1, cb2, cb3, cb4 = st.columns(4)
with cb1:
    if st.button("🔬  Bắt đầu dự đoán", use_container_width=True, type="primary"):
        st.switch_page("pages/1_prediction.py")
with cb2:
    if st.button("⊞  So sánh ma trận", use_container_width=True):
        st.switch_page("pages/1_prediction.py")
with cb3:
    if st.button("🕸️  Khám phá đồ thị", use_container_width=True):
        st.switch_page("pages/1_prediction.py")
with cb4:
    if st.button("✦  Sinh phân tử mới", use_container_width=True):
        st.switch_page("pages/1_prediction.py")

st.markdown("<br>", unsafe_allow_html=True)

# ── Stats ─────────────────────────────────────────────────────────────
stat_items = [
    ("💊", f"{s_drugs:,}",    "Hợp chất thuốc",    "linear-gradient(135deg,#6366f1,#818cf8)"),
    ("🦠", f"{s_diseases:,}", "Bệnh/rối loạn",     "linear-gradient(135deg,#ec4899,#f43f5e)"),
    ("🔬", f"{s_proteins:,}", "Protein mục tiêu",  "linear-gradient(135deg,#0ea5e9,#38bdf8)"),
    ("🔗", f"{s_links:,}",    "Liên kết đã biết",  "linear-gradient(135deg,#10b981,#34d399)"),
    ("🤖", str(s_models),      "Mô hình AI",        "linear-gradient(135deg,#8b5cf6,#a78bfa)"),
    ("📊", "10-fold",          "Cross-Validation",  "linear-gradient(135deg,#f59e0b,#fbbf24)"),
]
cols = st.columns(6)
for col, (icon, val, lbl, acc) in zip(cols, stat_items):
    with col:
        st.markdown(f"""
        <div class="stat-card" style="--accent:{acc}">
            <div class="stat-icon">{icon}</div>
            <div class="stat-val">{val}</div>
            <div class="stat-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Bottom section ────────────────────────────────────────────────────
c_left, c_mid, c_right = st.columns([1.15, 1.5, 1])

with c_left:
    st.markdown("""
    <div class="section-header">
        <span class="section-title">Mô hình AI</span>
        <div class="section-header-line"></div>
    </div>""", unsafe_allow_html=True)

    for name, auc_val, desc in [
        ("GCN · AMNTDDA",     auc_gcn,   "Graph Neural Network cơ sở"),
        ("GCN + Fuzzy",       auc_fuzzy, "AMNTDDA + Mamdani FIS tích hợp"),
    ]:
        auc_html = f'<span class="model-auc">AUC {auc_val:.4f}</span>' if auc_val else \
                   '<span style="color:#374151;font-size:0.78rem;">Chưa huấn luyện</span>'
        status_dot = '<span class="model-dot"></span>' if auc_val else \
                     '<span style="width:7px;height:7px;border-radius:50%;background:#374151;display:inline-block;margin-right:8px;"></span>'
        st.markdown(f"""
        <div class="model-card">
            <div class="model-card-accent"></div>
            <div class="model-card-header">
                <span>{status_dot}<span class="model-name">{name}</span></span>
                {auc_html}
            </div>
            <div class="model-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

    if best_auc:
        st.markdown(f"""
        <div style="background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.25);
                    border-radius:10px;padding:12px 14px;margin-top:4px;">
            <span style="color:#4ade80;font-size:0.82rem;font-weight:600;">
                ✓ Mô hình sẵn sàng · Best AUC: {best_auc:.4f}
            </span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);
                    border-radius:10px;padding:12px 14px;margin-top:4px;">
            <span style="color:#818cf8;font-size:0.8rem;">
                ℹ Chạy train_DDA_fuzzy.py để huấn luyện
            </span>
        </div>""", unsafe_allow_html=True)

with c_mid:
    st.markdown("""
    <div class="section-header">
        <span class="section-title">Phân phối dữ liệu</span>
        <div class="section-header-line"></div>
    </div>""", unsafe_allow_html=True)
    fig = donut_chart(
        labels=["Thuốc", "Bệnh", "Protein", "Liên kết"],
        values=[s_drugs, s_diseases, s_proteins, s_links],
        title="",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with c_right:
    st.markdown("""
    <div class="section-header">
        <span class="section-title">Truy cập nhanh</span>
        <div class="section-header-line"></div>
    </div>""", unsafe_allow_html=True)

    qa_items = [
        ("🔍", "linear-gradient(135deg,#6366f1,#818cf8)", "Dự đoán đơn",     "1 thuốc/bệnh → Top K kết quả"),
        ("⊞",  "linear-gradient(135deg,#0ea5e9,#38bdf8)", "Ma trận so sánh", "N×M heatmap GCN vs Fuzzy"),
        ("🕸️", "linear-gradient(135deg,#10b981,#34d399)", "Đồ thị mạng",     "Khám phá liên kết tương tác"),
        ("✦",  "linear-gradient(135deg,#f59e0b,#fbbf24)", "Sinh phân tử",    "VGAE đề xuất ứng viên mới"),
    ]
    for icon, bg, title, desc in qa_items:
        st.markdown(f"""
        <div class="qa-card">
            <div class="qa-icon" style="background:{bg};">{icon}</div>
            <div class="qa-body">
                <div class="qa-title">{title}</div>
                <div class="qa-desc">{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Đến trang Dự đoán →", use_container_width=True, type="primary"):
        st.switch_page("pages/1_prediction.py")
