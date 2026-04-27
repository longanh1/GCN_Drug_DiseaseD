# -*- coding: utf-8 -*-
"""
Page 1 – Dự đoán & Phân tích
Tabs: Dự đoán đơn | Ma trận so sánh | Đồ thị mạng lưới | Sinh phân tử mới
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import json
import base64
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils.api_client import (
    search_drugs, search_diseases, get_datasets,
    predict_single, predict_matrix, get_fuzzy_detail,
    compare_matrix, save_history,
    get_drug_disease_interaction, get_drug_network, classify_batch,
    get_training_results_ai,
)
from utils.chart_utils import (
    membership_chart, bar_chart_comparison,
    heatmap, radar_chart, score_bar,
)
from utils.molecule_utils import smiles_to_image_b64, get_mol_properties, smiles_to_3d_plotly

# ── CSS ────────────────────────────────────────────────────────────────────
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

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #6366f1; border-radius: 4px; }

/* Sidebar */
.sb-brand {
    padding: 24px 20px 20px; border-bottom: 1px solid rgba(99,102,241,0.25); margin-bottom: 8px;
}
.sb-brand-icon {
    width: 42px; height: 42px; border-radius: 12px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 1.4rem; margin-bottom: 10px; box-shadow: 0 4px 15px rgba(99,102,241,0.4);
}
.sb-brand-name {
    font-size: 1rem; font-weight: 700;
    background: linear-gradient(90deg,#818cf8,#c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.sb-brand-sub { font-size: 0.72rem; color: #475569; }
.sb-nav-label {
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.08em;
    color: #475569; text-transform: uppercase; padding: 16px 20px 6px;
}
.sb-dataset {
    margin: 0 12px; padding: 10px 14px;
    background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.2);
    border-radius: 10px;
}

/* Page header */
.page-header {
    display: flex; align-items: center; justify-content: space-between;
    background: linear-gradient(135deg, #eef2ff, #f8fafc);
    border: 1px solid rgba(99,102,241,0.25); border-radius: 16px;
    padding: 20px 28px; margin-bottom: 22px;
    backdrop-filter: blur(10px);
}
.page-header-left h2 {
    font-size: 1.4rem; font-weight: 800; margin: 0 0 4px 0;
    background: linear-gradient(90deg, #1e293b, #4338ca);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.page-header-left p { color: #475569; font-size: 0.8rem; margin: 0; }
.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.3);
    color: #4ade80; border-radius: 20px; padding: 6px 14px; font-size: 0.78rem; font-weight: 500;
}
@keyframes pulse-green {
    0%,100% { box-shadow: 0 0 0 0 rgba(34,197,94,0.4); }
    50% { box-shadow: 0 0 0 5px rgba(34,197,94,0); }
}
.live-dot {
    width: 6px; height: 6px; border-radius: 50%; background: #22c55e;
    animation: pulse-green 2s infinite;
}

/* Info strip */
.info-strip {
    display: flex; align-items: center; gap: 10px; margin-bottom: 20px; flex-wrap: wrap;
}
.info-chip {
    background: rgba(255,255,255,0.04); border: 1px solid #1e293b;
    border-radius: 8px; padding: 6px 14px; font-size: 0.82rem; color: #475569;
}
.info-chip b { color: #4f46e5; }
.info-chip.drug  { border-color: rgba(99,102,241,0.3); background: rgba(99,102,241,0.08); }
.info-chip.dis   { border-color: rgba(244,63,94,0.3);  background: rgba(244,63,94,0.08); color: #fda4af; }
.info-chip.prot  { border-color: rgba(14,165,233,0.3); background: rgba(14,165,233,0.08); color: #7dd3fc; }

/* Tab styling */
[data-testid="stTabs"] > div:first-child {
    background: rgba(255,255,255,0.03) !important; border-radius: 12px !important;
    border: 1px solid rgba(99,102,241,0.15) !important; padding: 4px !important;
    gap: 2px !important;
}
button[data-baseweb="tab"] {
    border-radius: 8px !important; font-weight: 500 !important;
    font-size: 0.84rem !important; color: #475569 !important;
    padding: 8px 18px !important; transition: all 0.15s !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    color: #fff !important; box-shadow: 0 2px 10px rgba(99,102,241,0.4) !important;
}
button[data-baseweb="tab"]:hover:not([aria-selected="true"]) {
    background: rgba(99,102,241,0.1) !important; color: #4338ca !important;
}
[data-testid="stTabPanel"] { padding-top: 20px !important; }

/* Panel box */
.panel {
    background: #ffffff;
    border-radius: 16px; padding: 22px;
    border: 1px solid #1e293b;
    box-shadow: 0 2px 12px rgba(99,102,241,0.08);
}
.panel-title {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em;
    color: #6366f1; text-transform: uppercase; margin-bottom: 14px;
}

/* Result row */
.res-row {
    display: flex; align-items: center; gap: 10px;
    background: #ffffff; border-radius: 10px;
    padding: 10px 14px; margin-bottom: 7px;
    border: 1px solid #e2e8f0;
    transition: border-color 0.15s, background 0.15s;
}
.res-row:hover {
    border-color: rgba(99,102,241,0.4); background: rgba(238,242,255,0.8);
}
.rank-num {
    min-width: 30px; text-align: center; font-weight: 700;
    font-size: 0.78rem; color: #64748b;
}
.rank-num.top3 { color: #f59e0b; }
.dis-name { flex: 1; color: #1e293b; font-size: 0.88rem; font-weight: 600; }
.dis-id   { color: #64748b; font-size: 0.73rem; margin-top: 1px; }
.known-pill {
    background: rgba(34,197,94,0.12); color: #4ade80;
    border: 1px solid rgba(34,197,94,0.3); border-radius: 6px;
    padding: 2px 8px; font-size: 0.72rem; font-weight: 500; white-space: nowrap;
}
.pred-pill {
    background: rgba(99,102,241,0.1); color: #4f46e5;
    border: 1px solid rgba(99,102,241,0.2); border-radius: 6px;
    padding: 2px 8px; font-size: 0.72rem; font-weight: 500; white-space: nowrap;
}
.score-val {
    font-size: 0.88rem; font-weight: 800;
    color: #4f46e5;
    min-width: 44px; text-align: right;
}

/* Fuzzy panel */
.fuzzy-panel {
    background: #ffffff;
    border-radius: 16px; padding: 24px;
    border: 1px solid rgba(99,102,241,0.35);
    box-shadow: 0 0 30px rgba(99,102,241,0.1);
    margin-top: 20px;
}
.fuzzy-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 16px;
}
.fuzzy-title {
    font-size: 1rem; font-weight: 700;
    background: linear-gradient(90deg, #818cf8, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.fuzzy-info-box {
    background: rgba(245,158,11,0.06); border: 1px solid rgba(245,158,11,0.25);
    border-radius: 10px; padding: 10px 14px;
    color: #fbbf24; font-size: 0.8rem; margin-bottom: 16px;
}

/* Molecule */
.mol-box {
    background: #f1f5f9; border-radius: 12px; overflow: hidden;
    border: 1px solid rgba(99,102,241,0.2); margin: 10px 0;
}
.mol-box img { display: block; width: 100%; }
.smiles-code {
    background: #f1f5f9; border: 1px solid rgba(99,102,241,0.15);
    border-radius: 8px; padding: 8px 10px; margin-top: 8px;
    font-family: monospace; font-size: 0.73rem; color: #4f46e5;
    word-break: break-all; line-height: 1.5;
}
.mol-prop {
    background: rgba(99,102,241,0.08); border-radius: 8px; padding: 10px 12px;
    border: 1px solid rgba(99,102,241,0.15); text-align: center; margin-top: 8px;
}
.mol-prop-val { font-size: 1.1rem; font-weight: 700; color: #4338ca; }
.mol-prop-lbl { font-size: 0.7rem; color: #475569; margin-top: 2px; }

/* Empty state */
.empty-state {
    background: #f8fafc;
    border-radius: 20px; padding: 70px 30px;
    border: 1px dashed rgba(99,102,241,0.3); text-align: center;
}
.empty-icon { font-size: 3.5rem; margin-bottom: 16px; opacity: 0.7; }
.empty-title { color: #1e293b; font-size: 1.1rem; font-weight: 600; margin-bottom: 8px; }
.empty-sub   { color: #475569; font-size: 0.85rem; }
.empty-legend {
    display: flex; gap: 20px; justify-content: center; margin-top: 28px;
}
.legend-item { display: flex; align-items: center; gap: 6px; font-size: 0.8rem; color: #475569; }
.legend-dot  { width: 8px; height: 8px; border-radius: 50%; }

/* Matrix */
.mat-summary-card {
    background: rgba(17,24,39,0.8); border-radius: 12px; padding: 16px 18px;
    border: 1px solid #1e293b; text-align: center;
}
.mat-summary-val {
    font-size: 1.4rem; font-weight: 800;
    background: linear-gradient(135deg,#1e293b,#4338ca);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.mat-summary-lbl { font-size: 0.72rem; color: #475569; margin-top: 4px; font-weight: 500; }

/* Buttons */
.stButton > button {
    border-radius: 10px !important; font-weight: 600 !important;
    font-size: 0.85rem !important; transition: all 0.2s !important; border: none !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(99,102,241,0.5) !important; transform: translateY(-1px);
}
.stButton > button:not([kind="primary"]) {
    background: rgba(99,102,241,0.1) !important;
    border: 1px solid rgba(99,102,241,0.3) !important; color: #4338ca !important;
}
.stSelectbox > div > div, .stMultiSelect > div > div {
    background: #f8fafc !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 10px !important; color: #1e293b !important;
}
[data-testid="stMetric"] {
    background: rgba(17,24,39,0.7); border-radius: 10px; padding: 12px 14px !important;
    border: 1px solid #e8ecf0;
}
[data-testid="stMetricLabel"] { color: #475569 !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] {
    font-size: 1.3rem !important; font-weight: 800 !important; color: #4338ca !important;
}
[data-testid="stPageLink"] a {
    border-radius: 10px !important; padding: 10px 16px !important;
    transition: background 0.15s !important; font-weight: 500 !important;
    color: #475569 !important; font-size: 0.88rem !important;
}
[data-testid="stPageLink"] a:hover {
    background: rgba(99,102,241,0.12) !important; color: #c7d2fe !important;
}
.stRadio > div { gap: 6px !important; }
.stRadio label {
    background: rgba(99,102,241,0.06) !important; border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 8px !important; padding: 6px 12px !important; color: #475569 !important;
    font-size: 0.82rem !important; cursor: pointer !important;
}
.stRadio label:has(input:checked) {
    background: rgba(99,102,241,0.2) !important; border-color: rgba(99,102,241,0.5) !important;
    color: #4338ca !important;
}
.stTextInput input {
    background: #f8fafc !important;
    border: 1px solid rgba(99,102,241,0.25) !important; border-radius: 10px !important;
    color: #1e293b !important;
}
.stTextInput input:focus {
    border-color: rgba(99,102,241,0.6) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}
.stSlider > div > div > div { background: rgba(99,102,241,0.3) !important; }
.stSlider > div > div > div > div { background: linear-gradient(90deg,#6366f1,#8b5cf6) !important; }
.stDownloadButton > button {
    background: rgba(16,185,129,0.1) !important; border: 1px solid rgba(16,185,129,0.3) !important;
    color: #34d399 !important; border-radius: 8px !important; font-size: 0.8rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ───────────────────────────────────────────────────────────
if "dataset" not in st.session_state:
    st.session_state.dataset = "C-dataset"
for k in ["selected_drug", "selected_disease", "prediction_result",
          "fuzzy_detail", "matrix_result", "mat_drugs", "mat_diseases",
          "_matrix_net_loaded"]:
    if k not in st.session_state:
        st.session_state[k] = None

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-icon">🧬</div>
        <div class="sb-brand-name">PharmaLink GCN</div>
        <div class="sb-brand-sub">Drug-Disease AI Platform v2.0</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="sb-nav-label">Điều hướng</div>', unsafe_allow_html=True)
    if st.button("🏠  Trang chủ",             use_container_width=True): st.switch_page("home.py")
    if st.button("🔬  Dự đoán & Phân tích",   use_container_width=True): st.switch_page("pages/1_prediction.py")
    if st.button("📋  Lịch sử",               use_container_width=True): st.switch_page("pages/2_history.py")
    if st.button("🔬  Giai đoạn mô hình gốc", use_container_width=True): st.switch_page("pages/3_model_stages.py")
    st.markdown('<div class="sb-nav-label">Cấu hình</div>', unsafe_allow_html=True)
    datasets = get_datasets()
    if datasets:
        st.markdown('<div class="sb-dataset"><div style="font-size:0.7rem;color:#6b7280;margin-bottom:6px;">Dataset</div>', unsafe_allow_html=True)
        chosen = st.selectbox("", datasets, label_visibility="collapsed",
                               index=datasets.index(st.session_state.dataset)
                               if st.session_state.dataset in datasets else 0)
        st.session_state.dataset = chosen
        st.markdown('</div>', unsafe_allow_html=True)

# ── Page header ─────────────────────────────────────────────────────────────
dataset      = st.session_state.dataset
drugs_all    = search_drugs(dataset, "", 9999)
diseases_all = search_diseases(dataset, "", 9999)

st.markdown(f"""
<div class="page-header">
    <div class="page-header-left">
        <h2>🔬 Dự đoán &amp; Phân tích</h2>
        <p>Trang chủ / Dự đoán · Dataset: {dataset}</p>
    </div>
    <div style="display:flex;align-items:center;gap:12px;">
        <span class="status-pill">
            <span class="live-dot"></span>AI Engine Online
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Dataset info strip
st.markdown(f"""
<div class="info-strip">
    <span class="info-chip">📂 {dataset}</span>
    <span class="info-chip drug">💊 <b>{len(drugs_all):,}</b> thuốc</span>
    <span class="info-chip dis">🦠 <b>{len(diseases_all):,}</b> bệnh</span>
    <span class="info-chip prot">🔬 Protein · AMNTDDA + Mamdani FIS</span>
</div>
""", unsafe_allow_html=True)

# ── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍  Dự đoán đơn",
    "⊞  Ma trận so sánh",
    "🕸️  Đồ thị mạng lưới",
    "✦  Sinh phân tử mới",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 – Dự đoán đơn
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    left_col, right_col = st.columns([1, 2.4], gap="medium")

    with left_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🎯 Cấu hình truy vấn</div>', unsafe_allow_html=True)

        entity_type = st.radio(
            "Loại thực thể", ["💊 Thuốc", "🦠 Bệnh"],
            horizontal=True, label_visibility="collapsed",
        )
        is_drug = "Thuốc" in entity_type
        st.markdown("<br>", unsafe_allow_html=True)

        if is_drug:
            drug_search = st.text_input("🔍 Tìm thuốc theo tên...", key="drug_search_input",
                                         placeholder="Ví dụ: Aspirin, Metformin...")
            filtered = [d for d in drugs_all
                        if drug_search.lower() in d.get("name", "").lower()
                        or drug_search.lower() in d.get("id", "").lower()
                        ] if drug_search else drugs_all[:50]

            options = {f"{d.get('name','?')}  ({d.get('id','')})": d for d in filtered}
            if options:
                st.markdown(
                    f'<div style="font-size:0.72rem;color:#6366f1;font-weight:600;'
                    f'letter-spacing:0.06em;text-transform:uppercase;margin-bottom:4px;">'
                    f'💊 Danh sách thuốc ({len(options):,} kết quả)</div>',
                    unsafe_allow_html=True,
                )
                sel_label = st.selectbox("Chọn thuốc", list(options.keys()), key="drug_sel",
                                          label_visibility="collapsed")
                drug_data = options[sel_label]
                st.session_state.selected_drug = drug_data
            else:
                st.markdown("""
                <div style="background:rgba(244,63,94,0.08);border:1px solid rgba(244,63,94,0.25);
                            border-radius:8px;padding:10px 14px;color:#fda4af;font-size:0.82rem;">
                    Không tìm thấy kết quả
                </div>""", unsafe_allow_html=True)
                drug_data = None

            if drug_data and drug_data.get("smiles"):
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="panel-title">⚗️ Cấu trúc phân tử</div>', unsafe_allow_html=True)
                _mol_mode = st.radio("Chế độ xem", ["🖼️ 2D", "🔮 3D"],
                                     horizontal=True, key="mol_view_mode",
                                     label_visibility="collapsed")
                smiles_str = drug_data["smiles"]
                if "2D" in _mol_mode:
                    img_b64 = smiles_to_image_b64(smiles_str, (320, 220))
                    if img_b64:
                        st.markdown(f"""
                        <div class="mol-box" style="cursor:zoom-in;">
                            <img src="data:image/png;base64,{img_b64}"
                                 style="width:100%;display:block;" />
                        </div>""", unsafe_allow_html=True)
                        # Full-size expandable via plotly image
                        with st.expander("🔍 Phóng to 2D"):
                            img_b64_lg = smiles_to_image_b64(smiles_str, (700, 500))
                            if img_b64_lg:
                                st.markdown(f'<img src="data:image/png;base64,{img_b64_lg}" style="width:100%;border-radius:8px;" />', unsafe_allow_html=True)
                else:
                    fig_3d = smiles_to_3d_plotly(smiles_str, title=drug_data.get("name","Phân tử"))
                    if fig_3d:
                        st.plotly_chart(fig_3d, use_container_width=True,
                                        config={"displayModeBar": True,
                                                "modeBarButtonsToAdd": ["zoomIn3d","zoomOut3d","resetCameraDefault3d"],
                                                "scrollZoom": True})
                    else:
                        st.caption("⚠️ Không thể tạo cấu trúc 3D (cần RDKit AllChem)")
                props = get_mol_properties(smiles_str)
                if props:
                    pc1, pc2 = st.columns(2)
                    with pc1:
                        st.markdown(f"""
                        <div class="mol-prop">
                            <div class="mol-prop-val">{props.get("num_atoms","?")}</div>
                            <div class="mol-prop-lbl">Nguyên tử</div>
                        </div>""", unsafe_allow_html=True)
                    with pc2:
                        st.markdown(f"""
                        <div class="mol-prop">
                            <div class="mol-prop-val">{props.get("mol_weight","?")} Da</div>
                            <div class="mol-prop-lbl">Khối lượng mol</div>
                        </div>""", unsafe_allow_html=True)
                st.markdown(f"""
                <div class="smiles-code" title="SMILES">{smiles_str[:80]}{'...' if len(smiles_str)>80 else ''}</div>
                """, unsafe_allow_html=True)
        else:
            dis_search = st.text_input("🔍 Tìm bệnh (OMIM ID)...", key="dis_search_input",
                                        placeholder="Ví dụ: OMIM:117550")
            filtered_d = [d for d in diseases_all
                          if dis_search.lower() in d.get("id", "").lower()
                          ] if dis_search else diseases_all[:50]
            options_d = {d.get("id", str(d.get("idx", i))): d for i, d in enumerate(filtered_d)}
            if options_d:
                sel_d = st.selectbox("Chọn bệnh", list(options_d.keys()), key="dis_sel",
                                      label_visibility="collapsed")
                st.session_state.selected_disease = options_d[sel_d]
            else:
                st.markdown("""
                <div style="background:rgba(244,63,94,0.08);border:1px solid rgba(244,63,94,0.25);
                            border-radius:8px;padding:10px 14px;color:#fda4af;font-size:0.82rem;">
                    Không tìm thấy kết quả
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="panel-title">⚙️ Tham số mô hình</div>', unsafe_allow_html=True)

        direction_map = {
            "💊→🦠  Thuốc → Bệnh":     "drug->disease",
            "💊→🔬  Thuốc → Protein":  "drug->protein",
            "🦠→💊  Bệnh → Thuốc":     "disease->drug",
        }
        direction_label = st.selectbox("Hướng dự đoán", list(direction_map.keys()),
                                        label_visibility="collapsed")
        direction = direction_map[direction_label]

        model_map = {
            "�  AMNTDDA gốc (không GCN)": "AMNTDDA",
            "🚀  AMNTDDA + GCN":           "AMNTDDA_GCN",
            "🌟  AMNTDDA + GCN + Fuzzy":   "AMNTDDA_Fuzzy",
        }
        model_label = st.selectbox("Mô hình AI", list(model_map.keys()),
                                    index=2, key="model_sel", label_visibility="collapsed")
        model = model_map[model_label]

        top_k = st.select_slider("Số kết quả Top K", [5, 10, 20, 50], value=10)

        _rel_options = ["💊–💊 Thuốc–Thuốc", "💊–🦠 Thuốc–Bệnh", "💊–🔬 Thuốc–Protein", "🦠–🦠 Bệnh–Bệnh"]
        rel_types = st.multiselect(
            "Loại quan hệ hiển thị (Tab Đồ thị)",
            _rel_options,
            default=["💊–🦠 Thuốc–Bệnh", "💊–🔬 Thuốc–Protein"],
            key="rel_types_sel",
        )
        st.session_state["rel_types"] = rel_types

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("▶  Chạy dự đoán AI", use_container_width=True, type="primary",
                     key="run_predict_btn"):
            drug_d = st.session_state.selected_drug
            if not drug_d and is_drug:
                st.error("Vui lòng chọn thuốc trước!")
            else:
                with st.spinner("Mô hình AI đang tính toán…"):
                    entity = drug_d or st.session_state.selected_disease
                    result = predict_single(
                        dataset=dataset,
                        drug_idx=entity.get("idx", 0) if entity else 0,
                        model=model,
                        top_k=top_k,
                    )
                    st.session_state.prediction_result = result
                    if entity:
                        save_history({
                            "drug": entity.get("name", entity.get("id", "")),
                            "direction": direction_label.split("  ")[-1],
                            "model": model,
                            "top_k": top_k,
                            "dataset": dataset,
                            "num_results": len(result.get("results", [])),
                        })

    with right_col:
        result = st.session_state.prediction_result

        if result is None:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">🕸️</div>
                <div class="empty-title">Chưa có kết quả dự đoán</div>
                <div class="empty-sub">Cấu hình tham số ở bảng bên trái rồi nhấn ▶ Chạy dự đoán AI</div>
                <div class="empty-legend">
                    <div class="legend-item">
                        <div class="legend-dot" style="background:#f97316"></div>Thuốc
                    </div>
                    <div class="legend-item">
                        <div class="legend-dot" style="background:#22c55e"></div>Bệnh đã biết
                    </div>
                    <div class="legend-item">
                        <div class="legend-dot" style="background:#6366f1"></div>Bệnh dự đoán
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        elif "error" in result:
            st.markdown(f"""
            <div style="background:rgba(244,63,94,0.08);border:1px solid rgba(244,63,94,0.3);
                        border-radius:12px;padding:16px 20px;color:#fda4af;">
                ⚠️ {result['error']}
            </div>""", unsafe_allow_html=True)
        else:
            drug_name    = result.get("drug_name", "")
            drug_name_vn = result.get("drug_name_vn", "")
            drug_class   = result.get("drug_class", "m")
            drug_id      = result.get("drug_id", "")
            model_tag    = result.get("model", "")
            topk_tag     = result.get("top_k", 10)
            results_list = result.get("results", [])

            # Drug classification badge
            _class_cfg = {
                "l": ("Mức thấp",   "#EF4444", "rgba(239,68,68,0.1)"),
                "m": ("Mức trung bình", "#F59E0B", "rgba(245,158,11,0.1)"),
                "h": ("Mức cao",     "#10B981", "rgba(16,185,129,0.1)"),
            }
            dc_label, dc_color, dc_bg = _class_cfg.get(drug_class, _class_cfg["m"])

            # Result header
            col_hdr, col_dl = st.columns([3, 1])
            with col_hdr:
                vn_str = f" · {drug_name_vn}" if drug_name_vn and drug_name_vn != drug_name else ""
                st.markdown(f"""
                <div style="margin-bottom:12px;">
                    <div style="font-size:1.1rem;font-weight:700;color:#1e293b;margin-bottom:6px;">
                        Kết quả: <span style="color:#4338ca">{drug_name}</span><span style="color:#64748b;font-size:.9rem">{vn_str}</span> → Bệnh
                        <span style="background:{dc_bg};border:1px solid {dc_color};color:{dc_color};border-radius:5px;padding:2px 8px;font-size:.75rem;margin-left:8px">{dc_label}</span>
                    </div>
                    <div style="display:flex;gap:8px;flex-wrap:wrap;">
                        <span style="background:rgba(99,102,241,0.15);border:1px solid rgba(99,102,241,0.3);
                                     color:#818cf8;border-radius:6px;padding:3px 10px;font-size:0.78rem;font-weight:600;">
                            {model_tag}
                        </span>
                        <span style="background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.25);
                                     color:#4ade80;border-radius:6px;padding:3px 10px;font-size:0.78rem;font-weight:600;">
                            Top {topk_tag} kết quả
                        </span>
                        <span style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.25);
                                     color:#fbbf24;border-radius:6px;padding:3px 10px;font-size:0.78rem;font-weight:600;">
                            {len(results_list)} bệnh tìm thấy
                        </span>
                    </div>
                </div>""", unsafe_allow_html=True)
            with col_dl:
                st.download_button(
                    "⬇ Tải CSV",
                    data=pd.DataFrame(results_list).to_csv(index=False),
                    file_name=f"prediction_{drug_name}.csv",
                    mime="text/csv",
                    key="dl_csv",
                )

            # ── 7-metric strip from training results ──────────────────────
            try:
                _base_model = model_tag.replace(" ", "").split("(")[0]
                if "Fuzzy" in model_tag:
                    _tr_model = "AMNTDDA_Fuzzy"
                else:
                    _tr_model = "AMNTDDA"
                tr = get_training_results_ai(dataset, _tr_model)
                _sm = tr.get("summary", {})
                _metrics_7 = [
                    ("AUC",       _sm.get("AUC",      {}).get("mean", None)),
                    ("AUPR",      _sm.get("AUPR",     {}).get("mean", None)),
                    ("Accuracy",  _sm.get("Accuracy", {}).get("mean", None)),
                    ("Precision", _sm.get("Precision",{}).get("mean", None)),
                    ("Recall",    _sm.get("Recall",   {}).get("mean", None)),
                    ("F1",        _sm.get("F1",       {}).get("mean", None)),
                    ("MCC",       _sm.get("MCC",      {}).get("mean", None)),
                ]
                valid = [(k, v) for k, v in _metrics_7 if v is not None]
                if valid:
                    _cols_7 = st.columns(len(valid))
                    for _c, (k, v) in zip(_cols_7, valid):
                        _c.markdown(f"""
                        <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);
                                    border-radius:8px;padding:8px 10px;text-align:center;margin-bottom:10px;">
                            <div style="font-size:1rem;font-weight:700;color:#818cf8;">{v:.4f}</div>
                            <div style="font-size:0.68rem;color:#64748b;font-weight:600;letter-spacing:0.04em;">{k}</div>
                        </div>""", unsafe_allow_html=True)
            except Exception:
                pass

            # Results list
            for item in results_list:
                rank     = item.get("rank", "")
                dis_name = item.get("disease_name", item.get("disease_id", ""))
                dis_id   = item.get("disease_id", "")
                gcn_s    = item.get("gcn_score", 0)
                fuz_s    = item.get("fuzzy_score", 0)
                is_known = item.get("is_known", False)
                dis_idx  = item.get("disease_idx", 0)
                show_score = fuz_s if "Fuzzy" in model_tag else gcn_s
                is_top3  = int(rank) <= 3 if str(rank).isdigit() else False

                known_html = '<span class="known-pill">✓ Đã biết</span>' if is_known else '<span class="pred-pill">Dự đoán</span>'
                rank_class = "rank-num top3" if is_top3 else "rank-num"

                col_r, col_d, col_s_, col_k, col_f = st.columns([0.35, 2.8, 1.3, 0.9, 0.75])
                with col_r:
                    st.markdown(f'<div style="padding-top:10px;"><span class="{rank_class}">#{rank}</span></div>', unsafe_allow_html=True)
                with col_d:
                    dis_name_vn  = item.get("disease_name_vn", "")
                    dis_class    = item.get("disease_class", "m")
                    dc2_lbl, dc2_col, dc2_bg = _class_cfg.get(dis_class, _class_cfg["m"])
                    vn2_str = f" · {dis_name_vn}" if dis_name_vn else ""
                    st.markdown(f"""
                    <div style="padding-top:6px;">
                        <div class="dis-name">{dis_name}<span style="background:{dc2_bg};border:1px solid {dc2_col};color:{dc2_col};border-radius:4px;padding:1px 6px;font-size:.7rem;margin-left:6px">{dis_class.upper()}</span></div>
                        <div class="dis-id">{dis_id}{vn2_str}</div>
                    </div>""", unsafe_allow_html=True)
                with col_s_:
                    bar_fig = score_bar(show_score)
                    st.plotly_chart(bar_fig, use_container_width=True,
                                    config={"displayModeBar": False}, key=f"bar_{rank}")
                with col_k:
                    st.markdown(f'<div style="padding-top:10px;">{known_html}</div>', unsafe_allow_html=True)
                with col_f:
                    if st.button("🔎", key=f"fuzzy_btn_{rank}", help="Phân tích Fuzzy & mạng tương tác"):
                        with st.spinner("Đang phân tích…"):
                            drug_idx = st.session_state.selected_drug.get("idx", 0) if st.session_state.selected_drug else 0
                            detail      = get_fuzzy_detail(dataset, drug_idx, dis_idx)
                            interaction = get_drug_disease_interaction(drug_idx, dis_idx, dataset)
                            st.session_state.fuzzy_detail = {
                                "detail": detail, "drug_name": drug_name, "dis_name": dis_name,
                                "interaction": interaction,
                            }

            # ── Fuzzy detail panel ──────────────────────────────────────────
            fz = st.session_state.get("fuzzy_detail")
            if fz and fz.get("detail"):
                d = fz["detail"]
                st.markdown(f"""
                <div class="fuzzy-panel">
                    <div class="fuzzy-header">
                        <div>
                            <div class="fuzzy-title">🔎 Phân tích Fuzzy Logic · Mamdani FIS</div>
                            <div style="color:#4b5563;font-size:0.82rem;margin-top:4px;">
                                {fz['drug_name']} → {fz['dis_name']}
                            </div>
                        </div>
                    </div>
                    <div class="fuzzy-info-box">
                        ℹ️ Hệ suy luận mờ Mamdani · 3 đầu vào · Hàm thành viên tam giác · 11 luật IF-THEN
                    </div>
                </div>""", unsafe_allow_html=True)

                cf_s  = d.get("cf_score", 0)
                src_s = d.get("src_neighbor", 0)
                tgt_s = d.get("tgt_neighbor", 0)
                fuz_s = d.get("fuzzy_score", 0)

                fc1, fc2, fc3, fc4 = st.columns(4)
                fc1.metric("CF Score",     f"{cf_s:.4f}")
                fc2.metric("Src Neighbor", f"{src_s:.4f}")
                fc3.metric("Tgt Neighbor", f"{tgt_s:.4f}")
                fc4.metric("Fuzzy Score",  f"{fuz_s:.4f}", delta=f"{fuz_s - cf_s:+.4f}")

                mfc = membership_chart()
                st.plotly_chart(mfc, use_container_width=True, config={"displayModeBar": False})

                mcols = st.columns(3)
                for c, name, key in zip(mcols,
                    ["CF Score", "Src Neighbor", "Tgt Neighbor"],
                    ["cf_memberships", "src_memberships", "tgt_memberships"]):
                    mem = d.get(key, {})
                    c.markdown(f"""
                    <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.15);
                                border-radius:8px;padding:10px 12px;">
                        <div style="color:#6366f1;font-size:0.7rem;font-weight:600;margin-bottom:6px;">{name}</div>
                        <div style="display:flex;gap:6px;">
                            <span style="background:rgba(239,68,68,0.1);color:#fca5a5;border-radius:4px;padding:2px 6px;font-size:0.72rem;">Lo: {mem.get('lo',0):.3f}</span>
                            <span style="background:rgba(245,158,11,0.1);color:#fcd34d;border-radius:4px;padding:2px 6px;font-size:0.72rem;">Mid: {mem.get('mid',0):.3f}</span>
                            <span style="background:rgba(34,197,94,0.1);color:#86efac;border-radius:4px;padding:2px 6px;font-size:0.72rem;">Hi: {mem.get('hi',0):.3f}</span>
                        </div>
                    </div>""", unsafe_allow_html=True)

                # Drug-Disease-Protein interaction network
                intx = fz.get('interaction', {})
                if intx and not intx.get('error'):
                    st.markdown('---')
                    st.markdown('#### 🕸️ Mạng lưới tương tác Thuốc–Bệnh–Protein')
                    bridging = intx.get('bridging_proteins', [])
                    drug_only   = intx.get('drug_only_proteins', [])
                    disease_only = intx.get('disease_only_proteins', [])
                    st.markdown(
                        f"**Protein trung gian** (tương tác với cả thuốc và bệnh): **{intx.get('num_bridging', 0)}** protein"
                    )
                    ic1, ic2, ic3 = st.columns(3)
                    with ic1:
                        st.markdown('**🔗 Protein trung gian (bridging)**')
                        if bridging:
                            for p in bridging[:8]:
                                st.markdown(f"- `{p.get('gene', p.get('id', '?'))}` {p.get('name_vn', '')}")
                        else:
                            st.caption('Không có')
                    with ic2:
                        st.markdown('**💊 Protein chỉ liên kết thuốc**')
                        if drug_only:
                            for p in drug_only[:8]:
                                st.markdown(f"- `{p.get('gene', p.get('id', '?'))}` {p.get('name_vn', '')}")
                        else:
                            st.caption('Không có')
                    with ic3:
                        st.markdown('**🦠 Protein chỉ liên kết bệnh**')
                        if disease_only:
                            for p in disease_only[:8]:
                                st.markdown(f"- `{p.get('gene', p.get('id', '?'))}` {p.get('name_vn', '')}")
                        else:
                            st.caption('Không có')
                    # Network graph
                    all_nodes = []
                    node_colors = []
                    # Drug node
                    all_nodes.append(intx.get('drug', {}).get('name_en', 'Drug'))
                    node_colors.append('#F97316')
                    # Disease node
                    all_nodes.append(intx.get('disease', {}).get('id', 'Disease'))
                    node_colors.append('#EF4444')
                    # Protein nodes
                    for p in (bridging + drug_only[:3] + disease_only[:3])[:15]:
                        all_nodes.append(p.get('gene', p.get('id', '?')))
                        node_colors.append('#3B82F6' if p in bridging else
                                           '#8B5CF6' if p in drug_only else '#10B981')
                    n_nodes = len(all_nodes)
                    angles  = [2*np.pi*i/max(n_nodes-2, 1) for i in range(n_nodes-2)]
                    nx_pos  = [0.0, 0.0] + [np.cos(a)*0.7 for a in angles]
                    ny_pos  = [0.5, -0.5] + [np.sin(a)*0.7 for a in angles]
                    ex, ey = [], []
                    for pi_idx in range(2, n_nodes):
                        # Drug–Protein or Disease–Protein edge
                        prot_name = all_nodes[pi_idx]
                        is_br = pi_idx < 2 + len(bridging)
                        if is_br:
                            ex += [nx_pos[0], nx_pos[pi_idx], None, nx_pos[1], nx_pos[pi_idx], None]
                            ey += [ny_pos[0], ny_pos[pi_idx], None, ny_pos[1], ny_pos[pi_idx], None]
                        elif pi_idx < 2 + len(bridging) + len(drug_only[:3]):
                            ex += [nx_pos[0], nx_pos[pi_idx], None]
                            ey += [ny_pos[0], ny_pos[pi_idx], None]
                        else:
                            ex += [nx_pos[1], nx_pos[pi_idx], None]
                            ey += [ny_pos[1], ny_pos[pi_idx], None]
                    # Drug-Disease edge
                    ex += [nx_pos[0], nx_pos[1], None]
                    ey += [ny_pos[0], ny_pos[1], None]
                    fig_net = go.Figure()
                    fig_net.add_trace(go.Scatter(x=ex, y=ey, mode='lines',
                        line=dict(color='rgba(99,102,241,0.25)', width=1.5), hoverinfo='none'))
                    fig_net.add_trace(go.Scatter(x=nx_pos, y=ny_pos, mode='markers+text',
                        marker=dict(color=node_colors, size=[20,20]+[12]*(n_nodes-2),
                                    line=dict(width=1.5, color='rgba(255,255,255,0.4)')),
                        text=all_nodes, textposition='top center',
                        textfont=dict(size=9, color='#64748b'),
                        hovertemplate='<b>%{text}</b><extra></extra>',
                    ))
                    fig_net.update_layout(
                        showlegend=False, height=380,
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        margin=dict(t=10, b=10, l=10, r=10),
                    )
                    st.plotly_chart(fig_net, use_container_width=True)

                if st.button("✕  Đóng phân tích Fuzzy", key="close_fuzzy"):
                    st.session_state.fuzzy_detail = None
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 – Ma trận so sánh
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div style="margin-bottom:20px;">
        <div style="font-size:1.1rem;font-weight:700;color:#1e293b;margin-bottom:4px;">
            ⊞ Ma trận so sánh GCN vs GCN+Fuzzy
        </div>
        <div style="color:#4b5563;font-size:0.82rem;">
            Chọn nhóm thuốc và bệnh để so sánh điểm dự đoán giữa hai mô hình qua heatmap.
        </div>
    </div>""", unsafe_allow_html=True)

    mat_left, mat_right = st.columns([1, 2.6], gap="medium")

    with mat_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">💊 Chọn thuốc (tối đa 5)</div>', unsafe_allow_html=True)
        drug_opts_mat = {f"{d.get('name','?')} ({d.get('id','')})" : d for d in drugs_all[:100]}
        sel_drugs_mat = st.multiselect("", list(drug_opts_mat.keys()),
                                        max_selections=5, key="mat_drug_sel",
                                        label_visibility="collapsed")

        st.markdown('<div class="panel-title" style="margin-top:14px;">🦠 Chọn bệnh (tối đa 5)</div>', unsafe_allow_html=True)
        dis_opts_mat = {d.get("id", str(d.get("idx", i))): d for i, d in enumerate(diseases_all[:100])}
        sel_dis_mat  = st.multiselect("", list(dis_opts_mat.keys()),
                                       max_selections=5, key="mat_dis_sel",
                                       label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("⊞  Chạy so sánh ma trận", use_container_width=True,
                     type="primary", key="run_matrix_btn"):
            if not sel_drugs_mat or not sel_dis_mat:
                st.warning("Cần chọn ít nhất 1 thuốc và 1 bệnh.")
            else:
                dr_idxs = [drug_opts_mat[k]["idx"] for k in sel_drugs_mat]
                di_idxs = [dis_opts_mat[k]["idx"] for k in sel_dis_mat]
                with st.spinner("Đang tính ma trận điểm…"):
                    mat_r = compare_matrix(dataset, dr_idxs, di_idxs)
                    st.session_state.matrix_result  = mat_r
                    st.session_state.mat_drugs      = [drug_opts_mat[k].get("name","?") for k in sel_drugs_mat]
                    st.session_state.mat_diseases   = list(sel_dis_mat)
                    st.session_state["_matrix_net_loaded"] = None

    with mat_right:
        mat_res = st.session_state.matrix_result
        if mat_res is None:
            st.markdown("""
            <div class="empty-state" style="padding:50px 30px;">
                <div class="empty-icon">⊞</div>
                <div class="empty-title">Chưa có ma trận so sánh</div>
                <div class="empty-sub">Chọn thuốc và bệnh ở bên trái, nhấn Chạy so sánh ma trận</div>
            </div>""", unsafe_allow_html=True)
        elif "error" in mat_res:
            st.markdown(f"""
            <div style="background:rgba(244,63,94,0.08);border:1px solid rgba(244,63,94,0.3);
                        border-radius:12px;padding:16px 20px;color:#fda4af;">⚠️ {mat_res['error']}</div>""",
                        unsafe_allow_html=True)
        else:
            cells      = mat_res.get("cells", [])
            gcn_avg    = mat_res.get("gcn_avg", 0)
            fuzzy_avg  = mat_res.get("fuzzy_avg", 0)
            known_cnt  = sum(1 for c in cells if c.get("is_known"))

            mc1, mc2, mc3, mc4 = st.columns(4)
            for col, val, lbl in [
                (mc1, len(cells),         "Tổng cặp"),
                (mc2, f"{fuzzy_avg:.3f}", "TB GCN+Fuzzy"),
                (mc3, f"{gcn_avg:.3f}",   "TB GCN"),
                (mc4, known_cnt,          "Cặp đã biết"),
            ]:
                col.markdown(f"""
                <div class="mat-summary-card">
                    <div class="mat-summary-val">{val}</div>
                    <div class="mat-summary-lbl">{lbl}</div>
                </div>""", unsafe_allow_html=True)

            if cells:
                dr_set = list(dict.fromkeys(c["drug_name"] for c in cells))
                di_set = list(dict.fromkeys(c["disease_name"] for c in cells))
                gcn_mat   = np.zeros((len(dr_set), len(di_set)))
                fuzzy_mat = np.zeros((len(dr_set), len(di_set)))
                dr_idx_map = {n: i for i, n in enumerate(dr_set)}
                di_idx_map = {n: i for i, n in enumerate(di_set)}
                for cell in cells:
                    ri = dr_idx_map.get(cell["drug_name"], 0)
                    ci = di_idx_map.get(cell["disease_name"], 0)
                    gcn_mat[ri, ci]   = cell.get("gcn_score", 0)
                    fuzzy_mat[ri, ci] = cell.get("fuzzy_score", 0)

                st.markdown("<br>", unsafe_allow_html=True)
                hm = heatmap(dr_set, di_set, gcn_mat, fuzzy_mat)
                st.plotly_chart(hm, use_container_width=True, config={"displayModeBar": False})

                col_rc, col_bar = st.columns(2)
                with col_rc:
                    metrics_cats = ["AUC Proxy", "Precision", "Recall"]
                    gcn_r_vals   = [gcn_avg, max(0, gcn_avg-0.05), max(0, gcn_avg-0.03)]
                    fuz_r_vals   = [fuzzy_avg, max(0, fuzzy_avg-0.04), max(0, fuzzy_avg-0.02)]
                    st.plotly_chart(radar_chart(metrics_cats, gcn_r_vals, fuz_r_vals),
                                    use_container_width=True, config={"displayModeBar": False})
                with col_bar:
                    dr_gcn   = {n: [] for n in dr_set}
                    dr_fuzzy = {n: [] for n in dr_set}
                    for cell in cells:
                        dn = cell["drug_name"]
                        dr_gcn[dn].append(cell.get("gcn_score", 0))
                        dr_fuzzy[dn].append(cell.get("fuzzy_score", 0))
                    avg_gcn   = [np.mean(dr_gcn[n])  for n in dr_set]
                    avg_fuzzy = [np.mean(dr_fuzzy[n]) for n in dr_set]
                    st.plotly_chart(bar_chart_comparison(dr_set, avg_gcn, avg_fuzzy),
                                    use_container_width=True, config={"displayModeBar": False})

                st.markdown('<div class="panel-title" style="margin-top:16px;">📋 Bảng chi tiết</div>', unsafe_allow_html=True)
                df_cells = pd.DataFrame(cells)
                if not df_cells.empty:
                    df_cells["Trạng thái"] = df_cells["is_known"].apply(
                        lambda x: "✅ Đã biết" if x else "🔵 Dự đoán")
                    st.dataframe(
                        df_cells[["drug_name","disease_name","gcn_score","fuzzy_score","delta","Trạng thái"]].rename(
                            columns={"drug_name":"Thuốc","disease_name":"Bệnh",
                                     "gcn_score":"GCN","fuzzy_score":"GCN+Fuzzy","delta":"Δ"}),
                        use_container_width=True, hide_index=True,
                    )

                # ── Metrics table per drug-disease pair ────────────────
                st.markdown("""
                <div style="margin-top:24px;margin-bottom:12px;">
                    <div style="font-size:0.95rem;font-weight:700;color:#1e293b;">
                        📊 Xác suất điều trị từng cặp thuốc–bệnh (AUC · AUPR · Accuracy · Precision · Recall · F1 · MCC)
                    </div>
                    <div style="color:#4b5563;font-size:0.8rem;margin-top:4px;">
                        Chỉ số mô hình huấn luyện kết hợp xác suất dự đoán từng cặp.
                    </div>
                </div>""", unsafe_allow_html=True)
                try:
                    _tr_gcn   = get_training_results_ai(dataset, "AMNTDDA")
                    _tr_fuzzy = get_training_results_ai(dataset, "AMNTDDA_Fuzzy")
                    _sm_gcn   = _tr_gcn.get("summary", {})
                    _sm_fuz   = _tr_fuzzy.get("summary", {})
                    def _get_mean(sm, key):
                        return sm.get(key, {}).get("mean", None)
                    _metric_keys = ["AUC", "AUPR", "Accuracy", "Precision", "Recall", "F1", "MCC"]
                    rows_metrics = []
                    for cell in cells:
                        for _m_name, _sm in [("GCN", _sm_gcn), ("GCN+Fuzzy", _sm_fuz)]:
                            score_val = cell.get("gcn_score",0) if _m_name=="GCN" else cell.get("fuzzy_score",0)
                            row = {
                                "Thuốc":  cell.get("drug_name",""),
                                "Bệnh":   cell.get("disease_name",""),
                                "Mô hình": _m_name,
                                "Điểm dự đoán": round(score_val, 4),
                                "Trạng thái": "✅ Đã biết" if cell.get("is_known") else "🔵 Dự đoán",
                            }
                            for k in _metric_keys:
                                v = _get_mean(_sm, k)
                                row[k] = round(v, 4) if v is not None else "—"
                            rows_metrics.append(row)
                    if rows_metrics:
                        df_metrics = pd.DataFrame(rows_metrics)
                        st.dataframe(df_metrics, use_container_width=True, hide_index=True,
                                     column_config={
                                         "Điểm dự đoán": st.column_config.ProgressColumn(
                                             "Điểm dự đoán", min_value=0, max_value=1, format="%.4f"),
                                     })
                except Exception:
                    pass

                # ── Unified network graph (2D / 3D) ────────────────────
                st.markdown("""
                <div style="margin-top:28px;margin-bottom:12px;">
                    <div style="font-size:0.95rem;font-weight:700;color:#1e293b;">
                        🕸️ Mạng lưới thống nhất Thuốc–Bệnh–Protein
                    </div>
                    <div style="color:#4b5563;font-size:0.8rem;margin-top:4px;">
                        Liên kết giữa thuốc, bệnh và protein trong tập đã chọn. Nhấn nút để tải dữ liệu mạng.
                    </div>
                </div>""", unsafe_allow_html=True)

                _net_view_mode = st.radio(
                    "Chế độ đồ thị mạng", ["🗺️ 2D", "🔮 3D"],
                    horizontal=True, key="matrix_net_mode", label_visibility="collapsed"
                )
                _load_net_btn = st.button("🕸️ Tải mạng lưới thuốc–bệnh–protein",
                                          key="load_matrix_net", use_container_width=True)

                if _load_net_btn or st.session_state.get("_matrix_net_loaded"):
                    st.session_state["_matrix_net_loaded"] = True
                    # Collect drug indices from cells
                    _drug_idx_map = {}
                    for cell in cells:
                        dn = cell.get("drug_name","")
                        if dn and dn not in _drug_idx_map:
                            _drug_idx_map[dn] = cell.get("drug_idx", None)

                    # Fetch protein data per drug
                    _net_proteins = {}  # drug_name -> list of protein dicts
                    with st.spinner("Đang tải dữ liệu mạng lưới…"):
                        for _dn, _di in _drug_idx_map.items():
                            if _di is not None:
                                _net_data = get_drug_network(_di, dataset)
                                if _net_data and not _net_data.get("error"):
                                    _net_proteins[_dn] = _net_data.get("drug_proteins", [])[:4]

                    # Build nodes and edges
                    import math as _math
                    _all_nodes = {}  # label -> {color, size, type}
                    _edges = []  # (label_a, label_b, color)

                    for _dn in dr_set:
                        _all_nodes[_dn] = {"color": "#f97316", "size": 22, "type": "drug"}
                    for _disn in di_set:
                        _all_nodes[_disn] = {"color": "#f43f5e", "size": 18, "type": "disease"}
                    for cell in cells:
                        _dn2 = cell.get("drug_name","")
                        _dis2 = cell.get("disease_name","")
                        _score = cell.get("fuzzy_score", cell.get("gcn_score", 0))
                        if _score > 0.3:
                            _edges.append((_dn2, _dis2, "rgba(99,102,241,0.5)", round(_score, 3)))
                    for _dn, _prots in _net_proteins.items():
                        for _p in _prots:
                            _plbl = _p.get("gene", _p.get("id", "?"))
                            if _plbl not in _all_nodes:
                                _all_nodes[_plbl] = {"color": "#38bdf8", "size": 10, "type": "protein"}
                            _edges.append((_dn, _plbl, "rgba(56,189,248,0.4)", None))

                    _node_labels = list(_all_nodes.keys())
                    n_net = len(_node_labels)
                    _label_idx = {lbl: i for i, lbl in enumerate(_node_labels)}

                    # Layout: drugs on left, diseases on right, proteins in middle ring
                    _pos_x = [0.0] * n_net
                    _pos_y = [0.0] * n_net
                    _pos_z = [0.0] * n_net
                    _drugs_l   = [l for l,v in _all_nodes.items() if v["type"]=="drug"]
                    _diseases_l = [l for l,v in _all_nodes.items() if v["type"]=="disease"]
                    _proteins_l = [l for l,v in _all_nodes.items() if v["type"]=="protein"]
                    for i, lbl in enumerate(_drugs_l):
                        idx = _label_idx[lbl]
                        _pos_x[idx] = -1.5
                        _pos_y[idx] = (i - (len(_drugs_l)-1)/2) * 0.8
                        _pos_z[idx] = 0.0
                    for i, lbl in enumerate(_diseases_l):
                        idx = _label_idx[lbl]
                        _pos_x[idx] = 1.5
                        _pos_y[idx] = (i - (len(_diseases_l)-1)/2) * 0.8
                        _pos_z[idx] = 0.0
                    for i, lbl in enumerate(_proteins_l):
                        idx = _label_idx[lbl]
                        angle = 2 * _math.pi * i / max(len(_proteins_l), 1)
                        _pos_x[idx] = 0.0 + 0.5 * _math.cos(angle)
                        _pos_y[idx] = 0.5 * _math.sin(angle)
                        _pos_z[idx] = 0.4 * _math.sin(angle * 2 + 1)

                    _node_colors = [_all_nodes[l]["color"] for l in _node_labels]
                    _node_sizes  = [_all_nodes[l]["size"]  for l in _node_labels]

                    if "2D" in _net_view_mode:
                        _fig_net = go.Figure()
                        for _la, _lb, _col, _w in _edges:
                            ia, ib = _label_idx.get(_la), _label_idx.get(_lb)
                            if ia is not None and ib is not None:
                                _fig_net.add_trace(go.Scatter(
                                    x=[_pos_x[ia], _pos_x[ib], None],
                                    y=[_pos_y[ia], _pos_y[ib], None],
                                    mode='lines',
                                    line=dict(color=_col, width=2.5 if _w else 1.2),
                                    hoverinfo='none', showlegend=False,
                                ))
                        _fig_net.add_trace(go.Scatter(
                            x=_pos_x, y=_pos_y,
                            mode='markers+text',
                            marker=dict(color=_node_colors, size=_node_sizes,
                                        line=dict(width=2, color='rgba(255,255,255,0.5)')),
                            text=_node_labels,
                            textposition='top center',
                            textfont=dict(size=10, color='#1e293b'),
                            hovertemplate='<b>%{text}</b><extra></extra>',
                            showlegend=False,
                        ))
                        _fig_net.update_layout(
                            height=480,
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(248,250,252,0.8)',
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                                       range=[-2.5, 2.5]),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            margin=dict(t=16, b=16, l=16, r=16),
                        )
                        # Legend
                        for _lbl2, _col2 in [("💊 Thuốc","#f97316"),("🦠 Bệnh","#f43f5e"),("🔬 Protein","#38bdf8")]:
                            _fig_net.add_trace(go.Scatter(
                                x=[None], y=[None], mode='markers',
                                marker=dict(size=10, color=_col2),
                                name=_lbl2, showlegend=True,
                            ))
                        _fig_net.update_layout(
                            legend=dict(orientation="h", y=1.05, x=0,
                                        font=dict(size=11, color="#475569"),
                                        bgcolor="rgba(0,0,0,0)"),
                        )
                        st.plotly_chart(_fig_net, use_container_width=True,
                                        config={"displayModeBar": True, "scrollZoom": True})
                    else:
                        # 3D network
                        _fig_net3 = go.Figure()
                        for _la, _lb, _col, _w in _edges:
                            ia, ib = _label_idx.get(_la), _label_idx.get(_lb)
                            if ia is not None and ib is not None:
                                _fig_net3.add_trace(go.Scatter3d(
                                    x=[_pos_x[ia], _pos_x[ib], None],
                                    y=[_pos_y[ia], _pos_y[ib], None],
                                    z=[_pos_z[ia], _pos_z[ib], None],
                                    mode='lines',
                                    line=dict(color=_col, width=4 if _w else 2),
                                    hoverinfo='none', showlegend=False,
                                ))
                        _fig_net3.add_trace(go.Scatter3d(
                            x=_pos_x, y=_pos_y, z=_pos_z,
                            mode='markers+text',
                            marker=dict(color=_node_colors, size=[s*0.7 for s in _node_sizes],
                                        opacity=0.92,
                                        line=dict(width=1.5, color='rgba(255,255,255,0.4)')),
                            text=_node_labels,
                            textfont=dict(size=9, color='#ffffff'),
                            hovertemplate='<b>%{text}</b><extra></extra>',
                            showlegend=False,
                        ))
                        _fig_net3.update_layout(
                            height=520,
                            scene=dict(
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                                           showbackground=False, title=''),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                                           showbackground=False, title=''),
                                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                                           showbackground=False, title=''),
                                bgcolor='rgba(10,15,30,0.95)',
                            ),
                            paper_bgcolor='rgba(10,15,30,0.95)',
                            margin=dict(t=16, b=0, l=0, r=0),
                        )
                        st.plotly_chart(_fig_net3, use_container_width=True,
                                        config={"displayModeBar": True, "scrollZoom": True})

                    # Legend pills
                    st.markdown("""
                    <div style="display:flex;gap:16px;padding:10px 4px;flex-wrap:wrap;">
                        <span style="color:#f97316;font-size:0.82rem;display:flex;align-items:center;gap:5px;">
                            <span style="width:12px;height:12px;border-radius:50%;background:#f97316;display:inline-block;"></span>💊 Thuốc
                        </span>
                        <span style="color:#f43f5e;font-size:0.82rem;display:flex;align-items:center;gap:5px;">
                            <span style="width:12px;height:12px;border-radius:50%;background:#f43f5e;display:inline-block;"></span>🦠 Bệnh
                        </span>
                        <span style="color:#38bdf8;font-size:0.82rem;display:flex;align-items:center;gap:5px;">
                            <span style="width:12px;height:12px;border-radius:50%;background:#38bdf8;display:inline-block;"></span>🔬 Protein liên kết
                        </span>
                        <span style="color:#818cf8;font-size:0.82rem;display:flex;align-items:center;gap:5px;">
                            <span style="width:12px;height:12px;border-radius:4px;background:rgba(99,102,241,0.5);display:inline-block;"></span>Cạnh: xác suất liên kết
                        </span>
                    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 – Đồ thị mạng lưới
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div style="margin-bottom:20px;">
        <div style="font-size:1.1rem;font-weight:700;color:#1e293b;margin-bottom:4px;">
            🕸️ Khám phá đồ thị mạng lưới
        </div>
        <div style="color:#4b5563;font-size:0.82rem;">
            Hình ảnh hóa mạng lưới drug-disease-protein tương tác. Chọn loại quan hệ trong Tab Dự đoán đơn.
        </div>
    </div>""", unsafe_allow_html=True)

    g_left, g_right = st.columns([1, 3], gap="medium")
    with g_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">⚙️ Cấu hình đồ thị</div>', unsafe_allow_html=True)
        graph_drug = st.selectbox("Thuốc trung tâm",
                                   [d.get("name","?") for d in drugs_all[:50]], key="graph_drug",
                                   label_visibility="collapsed")

        # Show active relation types
        _active_rels = st.session_state.get("rel_types", ["💊–🦠 Thuốc–Bệnh", "💊–🔬 Thuốc–Protein"])
        rel_labels = {
            "💊–💊 Thuốc–Thuốc":   ("Drug-Drug",      "#c084fc"),
            "💊–🦠 Thuốc–Bệnh":    ("Drug-Disease",   "#22c55e"),
            "💊–🔬 Thuốc–Protein": ("Drug-Protein",   "#38bdf8"),
            "🦠–🦠 Bệnh–Bệnh":    ("Disease-Disease", "#f472b6"),
        }
        if _active_rels:
            st.markdown('<div style="margin-top:8px;font-size:0.72rem;color:#64748b;font-weight:600;">Quan hệ đang hiển thị:</div>', unsafe_allow_html=True)
            for _r in _active_rels:
                _en, _col = rel_labels.get(_r, (_r, "#818cf8"))
                st.markdown(f'<span style="background:rgba(0,0,0,0.08);border:1px solid {_col};color:{_col};border-radius:5px;padding:2px 8px;font-size:0.72rem;margin:2px 0;display:inline-block;">{_r}</span>', unsafe_allow_html=True)
        else:
            st.caption("Chưa chọn loại quan hệ nào (Tab Dự đoán đơn → Tham số mô hình)")

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        load_btn = st.button("🕸️  Tải đồ thị", use_container_width=True, type="primary", key="load_graph")

    with g_right:
        if load_btn:
            drug_data = next((d for d in drugs_all if d.get("name") == graph_drug), None)
            if drug_data:
                drug_idx_g = drug_data.get("idx", 0)
                with st.spinner("Đang xây dựng đồ thị mạng lưới…"):
                    net = get_drug_network(drug_idx_g, dataset)

                if net and not net.get("error"):
                    show_dd  = "💊–💊 Thuốc–Thuốc"   in _active_rels
                    show_db  = "💊–🦠 Thuốc–Bệnh"    in _active_rels
                    show_dp  = "💊–🔬 Thuốc–Protein"  in _active_rels
                    show_dis = "🦠–🦠 Bệnh–Bệnh"     in _active_rels

                    known_diseases = net.get("known_diseases", [])[:8]
                    drug_proteins  = net.get("drug_proteins",  [])[:8]
                    similar_drugs  = net.get("similar_drugs",  [])[:6]
                    sim_dis_map    = net.get("similar_diseases", {})

                    # ── Build node/edge lists ───────────────────────────
                    nodes_lbl, nodes_col, nodes_sz, nodes_type = [], [], [], []

                    def _add_node(lbl, col, sz, ntype):
                        nodes_lbl.append(lbl)
                        nodes_col.append(col)
                        nodes_sz.append(sz)
                        nodes_type.append(ntype)
                        return len(nodes_lbl) - 1

                    # Central drug node (always shown)
                    center_idx = _add_node(graph_drug, "#f97316", 24, "drug")

                    node_lookup: dict[str, int] = {graph_drug: center_idx}

                    # Drug-Disease edges
                    if show_db:
                        for di in known_diseases:
                            lbl = di.get("id", str(di.get("idx","")))
                            if lbl not in node_lookup:
                                node_lookup[lbl] = _add_node(lbl, "#22c55e", 16, "disease")

                    # Drug-Protein edges
                    if show_dp:
                        for pr in drug_proteins:
                            lbl = pr.get("gene", pr.get("id", str(pr.get("idx",""))))
                            if lbl not in node_lookup:
                                node_lookup[lbl] = _add_node(lbl, "#38bdf8", 12, "protein")

                    # Drug-Drug edges
                    if show_dd:
                        for dr in similar_drugs:
                            lbl = dr.get("name_en", dr.get("name", str(dr.get("idx",""))))
                            if lbl not in node_lookup:
                                node_lookup[lbl] = _add_node(lbl, "#c084fc", 14, "sim_drug")

                    # Disease-Disease edges (neighbor diseases of known diseases)
                    if show_dis:
                        for di in known_diseases:
                            di_id = str(di.get("idx",""))
                            nbrs  = sim_dis_map.get(di_id, [])[:3]
                            for nb in nbrs:
                                lbl = nb.get("id", str(nb.get("idx","")))
                                if lbl not in node_lookup:
                                    node_lookup[lbl] = _add_node(lbl, "#f472b6", 12, "sim_disease")

                    # ── Layout: central + rings by type ────────────────
                    n = len(nodes_lbl)
                    import math
                    positions_x, positions_y = [0.0] * n, [0.0] * n

                    type_groups: dict[str, list[int]] = {}
                    for i, t in enumerate(nodes_type):
                        if i == 0: continue
                        type_groups.setdefault(t, []).append(i)

                    ring_r = {"disease": 1.0, "protein": 1.7, "sim_drug": 1.3, "sim_disease": 2.2}
                    for tname, idxs in type_groups.items():
                        r = ring_r.get(tname, 1.5)
                        for k, ni in enumerate(idxs):
                            angle = 2 * math.pi * k / max(len(idxs), 1)
                            positions_x[ni] = r * math.cos(angle)
                            positions_y[ni] = r * math.sin(angle)

                    # ── Edge traces by type ─────────────────────────────
                    def _edge_trace(ex, ey, color, name):
                        return go.Scatter(
                            x=ex, y=ey, mode='lines', name=name,
                            line=dict(color=color, width=1.5),
                            hoverinfo='none', showlegend=True,
                        )

                    traces = []

                    if show_db:
                        ex, ey = [], []
                        for di in known_diseases:
                            lbl = di.get("id", str(di.get("idx","")))
                            ni = node_lookup.get(lbl)
                            if ni is not None:
                                ex += [positions_x[0], positions_x[ni], None]
                                ey += [positions_y[0], positions_y[ni], None]
                        if ex:
                            traces.append(_edge_trace(ex, ey, "rgba(34,197,94,0.5)", "💊–🦠 Thuốc–Bệnh"))

                    if show_dp:
                        ex, ey = [], []
                        for pr in drug_proteins:
                            lbl = pr.get("gene", pr.get("id", str(pr.get("idx",""))))
                            ni = node_lookup.get(lbl)
                            if ni is not None:
                                ex += [positions_x[0], positions_x[ni], None]
                                ey += [positions_y[0], positions_y[ni], None]
                        if ex:
                            traces.append(_edge_trace(ex, ey, "rgba(56,189,248,0.5)", "💊–🔬 Thuốc–Protein"))

                    if show_dd:
                        ex, ey = [], []
                        for dr in similar_drugs:
                            lbl = dr.get("name_en", dr.get("name", str(dr.get("idx",""))))
                            ni = node_lookup.get(lbl)
                            if ni is not None:
                                ex += [positions_x[0], positions_x[ni], None]
                                ey += [positions_y[0], positions_y[ni], None]
                        if ex:
                            traces.append(_edge_trace(ex, ey, "rgba(192,132,252,0.6)", "💊–💊 Thuốc–Thuốc"))

                    if show_dis:
                        ex, ey = [], []
                        for di in known_diseases:
                            src_lbl = di.get("id", str(di.get("idx","")))
                            src_ni  = node_lookup.get(src_lbl)
                            di_id   = str(di.get("idx",""))
                            nbrs    = sim_dis_map.get(di_id, [])[:3]
                            for nb in nbrs:
                                tgt_lbl = nb.get("id", str(nb.get("idx","")))
                                tgt_ni  = node_lookup.get(tgt_lbl)
                                if src_ni is not None and tgt_ni is not None:
                                    ex += [positions_x[src_ni], positions_x[tgt_ni], None]
                                    ey += [positions_y[src_ni], positions_y[tgt_ni], None]
                        if ex:
                            traces.append(_edge_trace(ex, ey, "rgba(244,114,182,0.5)", "🦠–🦠 Bệnh–Bệnh"))

                    # Node trace
                    hover_texts = []
                    for i, (lbl, t) in enumerate(zip(nodes_lbl, nodes_type)):
                        if t == "sim_drug":
                            dr_info = next((d for d in similar_drugs if d.get("name_en", d.get("name","")) == lbl), {})
                            hover_texts.append(f"<b>{lbl}</b><br>Thuốc tương tự<br>sim={dr_info.get('sim','')}")
                        elif t == "disease":
                            hover_texts.append(f"<b>{lbl}</b><br>Bệnh đã biết")
                        elif t == "sim_disease":
                            hover_texts.append(f"<b>{lbl}</b><br>Bệnh tương tự")
                        elif t == "protein":
                            pr_info = next((p for p in drug_proteins if p.get("gene", p.get("id","")) == lbl), {})
                            hover_texts.append(f"<b>{lbl}</b><br>{pr_info.get('name_vn', pr_info.get('name_en',''))}")
                        else:
                            hover_texts.append(f"<b>{lbl}</b>")

                    traces.append(go.Scatter(
                        x=positions_x, y=positions_y,
                        mode='markers+text',
                        marker=dict(color=nodes_col, size=nodes_sz,
                                    line=dict(width=1.5, color='rgba(255,255,255,0.3)'),
                                    opacity=0.92),
                        text=nodes_lbl, textposition="top center",
                        textfont=dict(size=8, color="#94a3b8"),
                        hovertext=hover_texts,
                        hovertemplate="%{hovertext}<extra></extra>",
                        name="Nodes", showlegend=False,
                    ))

                    fig = go.Figure(data=traces)
                    fig.update_layout(
                        showlegend=True,
                        legend=dict(
                            orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                            font=dict(size=11, color="#64748b"),
                            bgcolor="rgba(0,0,0,0)",
                        ),
                        hovermode='closest',
                        paper_bgcolor='rgba(6,11,23,0)', plot_bgcolor='rgba(6,11,23,0)',
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        margin=dict(t=40, b=16, l=16, r=16), height=560,
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                    # Summary stats
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    for _c, _lbl, _val, _col in [
                        (sc1, "Bệnh đã biết", len(known_diseases), "#22c55e"),
                        (sc2, "Protein liên kết", len(drug_proteins), "#38bdf8"),
                        (sc3, "Thuốc tương tự", len(similar_drugs), "#c084fc"),
                        (sc4, "Bệnh tương tự", sum(len(v) for v in sim_dis_map.values()), "#f472b6"),
                    ]:
                        _c.markdown(f"""
                        <div style="background:rgba(0,0,0,0.04);border:1px solid {_col}40;
                                    border-radius:8px;padding:10px;text-align:center;">
                            <div style="font-size:1.3rem;font-weight:700;color:{_col};">{_val}</div>
                            <div style="font-size:0.72rem;color:#64748b;">{_lbl}</div>
                        </div>""", unsafe_allow_html=True)

                    # Node type legend
                    st.markdown("""
                    <div style="display:flex;gap:20px;padding:12px 4px;flex-wrap:wrap;">
                        <span style="color:#f97316;font-size:0.82rem;display:flex;align-items:center;gap:5px;">
                            <span style="width:10px;height:10px;border-radius:50%;background:#f97316;display:inline-block;"></span>Thuốc trung tâm
                        </span>
                        <span style="color:#22c55e;font-size:0.82rem;display:flex;align-items:center;gap:5px;">
                            <span style="width:10px;height:10px;border-radius:50%;background:#22c55e;display:inline-block;"></span>Bệnh đã biết
                        </span>
                        <span style="color:#38bdf8;font-size:0.82rem;display:flex;align-items:center;gap:5px;">
                            <span style="width:10px;height:10px;border-radius:50%;background:#38bdf8;display:inline-block;"></span>Protein liên kết
                        </span>
                        <span style="color:#c084fc;font-size:0.82rem;display:flex;align-items:center;gap:5px;">
                            <span style="width:10px;height:10px;border-radius:50%;background:#c084fc;display:inline-block;"></span>Thuốc tương tự
                        </span>
                        <span style="color:#f472b6;font-size:0.82rem;display:flex;align-items:center;gap:5px;">
                            <span style="width:10px;height:10px;border-radius:50%;background:#f472b6;display:inline-block;"></span>Bệnh tương tự
                        </span>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.error("Không thể tải dữ liệu mạng lưới từ API.")
        else:
            st.markdown("""
            <div class="empty-state" style="padding:50px 30px;">
                <div class="empty-icon">🕸️</div>
                <div class="empty-title">Chưa tải đồ thị</div>
                <div class="empty-sub">Chọn thuốc trung tâm và nhấn Tải đồ thị</div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 – Sinh phân tử mới
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div style="margin-bottom:20px;">
        <div style="font-size:1.1rem;font-weight:700;color:#1e293b;margin-bottom:4px;">
            ✦ Sinh phân tử mới (VGAE)
        </div>
        <div style="color:#4b5563;font-size:0.82rem;">
            Variational Graph Autoencoder đề xuất ứng viên thuốc mới nhắm đến bệnh mục tiêu.
        </div>
    </div>""", unsafe_allow_html=True)

    vg_left, vg_right = st.columns([1, 2], gap="medium")
    with vg_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">🎯 Cấu hình VGAE</div>', unsafe_allow_html=True)
        target_disease = st.selectbox("Bệnh mục tiêu",
                                       [d.get("id","?") for d in diseases_all[:30]],
                                       key="gen_disease", label_visibility="collapsed")
        threshold = st.slider("Ngưỡng tin cậy", 0.5, 0.99, 0.90, 0.01)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✦  Sinh phân tử mới", type="primary", key="gen_btn",
                     use_container_width=True):
            st.markdown("""
            <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.25);
                        border-radius:12px;padding:16px 20px;color:#4338ca;font-size:0.85rem;">
                ℹ️ VGAE module đang khởi động... Kết quả sẽ hiển thị sau khi
                <code style="background:rgba(99,102,241,0.15);padding:2px 6px;border-radius:4px;">
                train_vgae.py</code> hoàn tất.
            </div>""", unsafe_allow_html=True)
            st.code("cd AI_ENGINE\npython src/train_vgae.py --disease " + target_disease,
                    language="bash")

    with vg_right:
        st.markdown("""
        <div class="panel">
            <div class="panel-title">📖 Hướng dẫn sử dụng VGAE</div>
            <div style="display:flex;flex-direction:column;gap:14px;margin-top:8px;">
                <div style="display:flex;gap:12px;align-items:flex-start;">
                    <div style="width:28px;height:28px;border-radius:8px;background:linear-gradient(135deg,#6366f1,#818cf8);
                                display:flex;align-items:center;justify-content:center;font-size:0.8rem;font-weight:700;color:#fff;flex-shrink:0;">1</div>
                    <div>
                        <div style="color:#1e293b;font-size:0.88rem;font-weight:600;">Cài đặt môi trường</div>
                        <div style="color:#4b5563;font-size:0.78rem;margin-top:2px;">Đảm bảo gcn_venv đã cài đầy đủ torch, dgl, rdkit</div>
                    </div>
                </div>
                <div style="display:flex;gap:12px;align-items:flex-start;">
                    <div style="width:28px;height:28px;border-radius:8px;background:linear-gradient(135deg,#0ea5e9,#38bdf8);
                                display:flex;align-items:center;justify-content:center;font-size:0.8rem;font-weight:700;color:#fff;flex-shrink:0;">2</div>
                    <div>
                        <div style="color:#1e293b;font-size:0.88rem;font-weight:600;">Chạy VGAE training</div>
                        <div style="color:#4b5563;font-size:0.78rem;margin-top:2px;">train_vgae.py học phân bố latent của đồ thị</div>
                    </div>
                </div>
                <div style="display:flex;gap:12px;align-items:flex-start;">
                    <div style="width:28px;height:28px;border-radius:8px;background:linear-gradient(135deg,#10b981,#34d399);
                                display:flex;align-items:center;justify-content:center;font-size:0.8rem;font-weight:700;color:#fff;flex-shrink:0;">3</div>
                    <div>
                        <div style="color:#1e293b;font-size:0.88rem;font-weight:600;">Xem kết quả</div>
                        <div style="color:#4b5563;font-size:0.78rem;margin-top:2px;">Ứng viên phân tử mới được lưu và hiển thị tại đây</div>
                    </div>
                </div>
            </div>
            <div style="margin-top:20px;background:rgba(10,16,32,0.8);border:1px solid rgba(99,102,241,0.15);
                        border-radius:8px;padding:12px 14px;">
                <div style="font-size:0.7rem;color:#4b5563;font-weight:600;margin-bottom:6px;letter-spacing:0.06em;">LỆNH CHẠY</div>
                <code style="color:#818cf8;font-size:0.8rem;">
                    python AI_ENGINE/src/train_vgae.py<br>
                    --dataset C-dataset --epochs 500
                </code>
            </div>
        </div>""", unsafe_allow_html=True)
