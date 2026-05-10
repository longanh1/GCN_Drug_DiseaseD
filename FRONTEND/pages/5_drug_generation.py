"""
5_drug_generation.py — Sinh thuốc mới bằng VGAE (Variational Graph Autoencoder)

Hiển thị quá trình:
  1. Load dataset Drug-Protein từ CSV
  2. Huấn luyện VGAE (GraphVAE)
  3. Sinh liên kết thuốc-protein mới (chưa có trong dataset)
  4. Trực quan hoá mạng lưới trước/sau

Cấu trúc VGAE:
  - Input: Drug features (Fingerprint + GIP) + Protein features (ESM)
  - Encoder: 2x GCNConv → mu/logstd
  - Decoder: Dot-product sigmoid reconstruction
"""

import os
import sys
import time
import json
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.api_client import _AI, _post, _get

AI = _AI

# ── Helpers ───────────────────────────────────────────────────────────
def run_vgae(dataset: str) -> dict:
    return _post(f"{AI}/vgae/run?dataset={dataset}", timeout=360) or {}

def get_vgae_results(dataset: str) -> dict:
    return _get(f"{AI}/vgae/results", {"dataset": dataset}, timeout=15) or {}

# ── Layout ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Sinh Thuốc Mới – VGAE", layout="wide")

st.markdown("""
<h1 style='text-align:center;margin-bottom:4px'>
    🧪 Sinh Thuốc Mới bằng VGAE
</h1>
<p style='text-align:center;color:#94A3B8;margin-bottom:24px'>
    Variational Graph Autoencoder — Tự động phát hiện liên kết thuốc–protein tiềm năng
</p>
""", unsafe_allow_html=True)

# ── Architecture diagram ──────────────────────────────────────────────
with st.expander("📐 Kiến trúc VGAE", expanded=False):
    st.markdown("""
    ```
    Drug Features           Protein Features
    (Fingerprint + GIP)     (ESM embeddings)
          │                       │
          └──────── concat ────────┘
                       │
               [GCNConv Layer 1]
               hidden_dim = 256
                       │
               [GCNConv Layer 2]
               hidden_dim = 256
                   ↙         ↘
              [mu_layer]  [logstd_layer]
              out_dim=128   out_dim=128
                   ↘         ↙
              z ~ N(mu, exp(logstd))
                       │
             [Dot-product Decoder]
             p(A|z) = σ(z·zᵀ)
                       │
            Ngưỡng tin cậy ≥ 0.95
                       │
         Liên kết thuốc–protein MỚI
    ```
    """)

# ── Controls ──────────────────────────────────────────────────────────
col_cfg, col_run = st.columns([2, 3])
with col_cfg:
    st.markdown("### ⚙️ Cấu hình")
    dataset = st.selectbox(
        "Dataset",
        ["B-dataset", "C-dataset", "F-dataset"],
        label_visibility="collapsed",
        key="vgae_dataset",
    )
    threshold_info = st.info("Ngưỡng tin cậy: **0.95** (cố định trong model)")

with col_run:
    st.markdown("### 🚀 Huấn luyện & Sinh")
    st.markdown("""
    - **500 epochs**, lr=0.01
    - Drug features: **538-dim** (Fingerprint 269 + GIP 269)
    - Protein features: **320-dim** (ESM, padded lên 538)
    - Chỉ lấy liên kết **mới** (chưa tồn tại trong dataset)
    """)
    run_btn = st.button("▶ Chạy VGAE", type="primary", use_container_width=True)

st.divider()

# ── Run VGAE ──────────────────────────────────────────────────────────
if run_btn:
    with st.status("⏳ Đang chạy VGAE...", expanded=True) as status:
        st.write("📂 Đọc dữ liệu từ CSV...")
        st.write(f"   → AMDGT_main/data/{dataset}/DrugFingerprint.csv")
        st.write(f"   → AMDGT_main/data/{dataset}/Protein_ESM.csv")
        st.write(f"   → AMDGT_main/data/{dataset}/DrugProteinAssociationNumber.csv")

        st.write("🧠 Huấn luyện VGAE (500 epochs)...")
        t0 = time.time()
        resp = run_vgae(dataset)
        elapsed = time.time() - t0

        if resp.get("success"):
            status.update(label=f"✅ Hoàn thành trong {elapsed:.1f}s", state="complete")
            st.success(f"Sinh xong! Mất {elapsed:.1f} giây")
            # Show training log
            log = resp.get("log", "")
            if log:
                with st.expander("📋 Training log", expanded=False):
                    st.code(log[-3000:] if len(log) > 3000 else log)
        else:
            status.update(label="❌ Lỗi", state="error")
            st.error("Quá trình chạy thất bại!")
            log = resp.get("log", "Không có thông tin lỗi.")
            with st.expander("📋 Error log"):
                st.code(log[-3000:] if len(log) > 3000 else log)

# ── Load results ──────────────────────────────────────────────────────
results = get_vgae_results(dataset)

if not results.get("generated"):
    st.info("⚠️ Chưa có kết quả cho dataset này. Nhấn **▶ Chạy VGAE** để bắt đầu.")
    st.stop()

edges = results.get("edges", [])
count = results.get("count", 0)

# ── Summary metrics ───────────────────────────────────────────────────
st.markdown("## 📊 Kết quả")
m1, m2, m3 = st.columns(3)
m1.metric("Liên kết mới phát hiện", count)
m2.metric("Dataset", dataset)
m3.metric("Ngưỡng tin cậy", "≥ 0.95")

st.divider()

if not edges:
    st.warning("Không tìm thấy liên kết mới nào với ngưỡng tin cậy hiện tại.")
    st.stop()

# ── Table ─────────────────────────────────────────────────────────────
df = pd.DataFrame(edges)

tab_table, tab_graph, tab_drug, tab_prot = st.tabs([
    "📋 Bảng liên kết", "🕸️ Mạng lưới", "💊 Theo Thuốc", "🔬 Theo Protein"
])

with tab_table:
    st.markdown("### Danh sách liên kết thuốc–protein mới")
    display_df = df[["drug_name", "drug_id", "prot_name", "prot_id"]].copy()
    display_df.columns = ["Tên Thuốc", "Drug ID", "Tên Protein", "Protein ID"]
    st.dataframe(display_df, use_container_width=True, height=400)

# ── Network graph ─────────────────────────────────────────────────────
with tab_graph:
    st.markdown("### 🕸️ Mạng lưới liên kết mới")

    drug_nodes  = df["drug_name"].unique().tolist()
    prot_nodes  = df["prot_name"].unique().tolist()
    all_nodes   = drug_nodes + prot_nodes
    n_total     = len(all_nodes)
    node_idx    = {name: i for i, name in enumerate(all_nodes)}

    # Layout: drugs on left, proteins on right
    n_d = len(drug_nodes)
    n_p = len(prot_nodes)
    x_pos = [0.0] * n_d + [1.0] * n_p
    y_d   = np.linspace(0, 1, max(n_d, 1))
    y_p   = np.linspace(0, 1, max(n_p, 1))
    y_pos = list(y_d) + list(y_p)

    # Colors
    colors = ["#3B82F6"] * n_d + ["#10B981"] * n_p

    edge_x, edge_y = [], []
    for _, row in df.iterrows():
        xi = x_pos[node_idx[row["drug_name"]]]
        xj = x_pos[node_idx[row["prot_name"]]]
        yi = y_pos[node_idx[row["drug_name"]]]
        yj = y_pos[node_idx[row["prot_name"]]]
        edge_x += [xi, xj, None]
        edge_y += [yi, yj, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(color="#6B7280", width=0.8),
        hoverinfo="none",
        name="Liên kết mới",
    ))
    fig.add_trace(go.Scatter(
        x=x_pos, y=y_pos,
        mode="markers+text",
        marker=dict(size=10, color=colors, line=dict(width=1, color="white")),
        text=all_nodes,
        textposition="middle right",
        textfont=dict(size=9),
        hovertext=all_nodes,
        hoverinfo="text",
        name="Nodes",
    ))
    fig.update_layout(
        height=500,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   tickvals=[0, 1], ticktext=["💊 Thuốc", "🔬 Protein"]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#0F172A",
        paper_bgcolor="#0F172A",
        font=dict(color="white"),
        margin=dict(l=20, r=20, t=40, b=20),
        title=dict(text="Liên kết Thuốc–Protein Mới (màu xanh=thuốc, màu lục=protein)",
                   font=dict(size=13, color="#94A3B8")),
    )
    # Annotation labels
    fig.add_annotation(x=0, y=1.05, text="💊 Thuốc", showarrow=False,
                       font=dict(color="#3B82F6", size=13), xref="x", yref="y")
    fig.add_annotation(x=1, y=1.05, text="🔬 Protein", showarrow=False,
                       font=dict(color="#10B981", size=13), xref="x", yref="y")
    st.plotly_chart(fig, use_container_width=True)

# ── Drug-centric view ─────────────────────────────────────────────────
with tab_drug:
    st.markdown("### 💊 Số liên kết mới theo Thuốc")
    drug_counts = df["drug_name"].value_counts().reset_index()
    drug_counts.columns = ["Thuốc", "Số liên kết mới"]
    fig2 = px.bar(drug_counts.head(20), x="Số liên kết mới", y="Thuốc",
                  orientation="h", color="Số liên kết mới",
                  color_continuous_scale="Blues",
                  title="Top 20 thuốc có nhiều liên kết mới nhất")
    fig2.update_layout(height=450, yaxis_title="", xaxis_title="Số liên kết mới phát hiện",
                       plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
                       font=dict(color="white"))
    st.plotly_chart(fig2, use_container_width=True)

    sel_drug = st.selectbox("🔍 Xem chi tiết thuốc",
                            ["(Tất cả)"] + sorted(df["drug_name"].unique().tolist()),
                            label_visibility="visible")
    if sel_drug != "(Tất cả)":
        sub = df[df["drug_name"] == sel_drug][["prot_name", "prot_id"]]
        sub.columns = ["Protein kết nối", "Protein ID"]
        st.dataframe(sub, use_container_width=True)

# ── Protein-centric view ──────────────────────────────────────────────
with tab_prot:
    st.markdown("### 🔬 Số liên kết mới theo Protein")
    prot_counts = df["prot_name"].value_counts().reset_index()
    prot_counts.columns = ["Protein", "Số liên kết mới"]
    fig3 = px.bar(prot_counts.head(20), x="Số liên kết mới", y="Protein",
                  orientation="h", color="Số liên kết mới",
                  color_continuous_scale="Greens",
                  title="Top 20 protein có nhiều liên kết mới nhất")
    fig3.update_layout(height=450, yaxis_title="", xaxis_title="Số liên kết mới phát hiện",
                       plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
                       font=dict(color="white"))
    st.plotly_chart(fig3, use_container_width=True)

    sel_prot = st.selectbox("🔍 Xem chi tiết protein",
                            ["(Tất cả)"] + sorted(df["prot_name"].unique().tolist()),
                            label_visibility="visible")
    if sel_prot != "(Tất cả)":
        sub = df[df["prot_name"] == sel_prot][["drug_name", "drug_id"]]
        sub.columns = ["Thuốc kết nối", "Drug ID"]
        st.dataframe(sub, use_container_width=True)
