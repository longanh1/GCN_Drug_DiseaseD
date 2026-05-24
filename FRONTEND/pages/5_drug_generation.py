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

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:20px 16px 14px;border-bottom:1px solid rgba(99,102,241,0.2);margin-bottom:8px">
        <div style="font-size:1rem;font-weight:700;color:#818cf8">🧬 PharmaLink GCN</div>
        <div style="font-size:0.72rem;color:#475569">Drug-Disease AI Platform v2.0</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.68rem;font-weight:600;color:#475569;text-transform:uppercase;letter-spacing:.08em;padding:12px 4px 6px">Điều hướng</div>', unsafe_allow_html=True)
    if st.button("🏠  Trang chủ",              use_container_width=True): st.switch_page("home.py")
    if st.button("🔬  Dự đoán & Phân tích",    use_container_width=True): st.switch_page("pages/1_prediction.py")
    if st.button("📋  Lịch sử",               use_container_width=True): st.switch_page("pages/2_history.py")
    if st.button("🧬  Giai đoạn mô hình gốc",  use_container_width=True): st.switch_page("pages/3_model_stages.py")
    if st.button("📊  Ablation Study",          use_container_width=True): st.switch_page("pages/4_ablation.py")
    if st.button("🧪  Sinh thuốc mới (VGAE)",   use_container_width=True): st.switch_page("pages/5_drug_generation.py")

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
    st.markdown("### 🕸️ Mạng lưới liên kết thuốc–protein")

    # ── Controls ──────────────────────────────────────────────────────
    _g_ctrl1, _g_ctrl2, _g_ctrl3 = st.columns([3, 1, 1])
    _max_per = max(len(df["drug_name"].unique()), len(df["prot_name"].unique()))
    _top_n   = _g_ctrl1.slider(
        "Số nút mỗi nhóm (top theo số liên kết)",
        min_value=5, max_value=min(50, _max_per),
        value=min(15, _max_per), step=5, key="g_topn",
    )
    _view3d   = _g_ctrl2.toggle("🌐 3D", value=False, key="g_3d")
    _show_lbl = _g_ctrl3.checkbox("Nhãn", value=False, key="g_lbl")

    # ── Filter to top-N most connected per group ───────────────────────
    _top_drugs = df["drug_name"].value_counts().head(_top_n).index.tolist()
    _top_prots = df["prot_name"].value_counts().head(_top_n).index.tolist()
    _df_g = df[df["drug_name"].isin(_top_drugs) & df["prot_name"].isin(_top_prots)]

    if _df_g.empty:
        st.info("Không có liên kết nào sau khi lọc.")
    else:
        _n_d = len(_top_drugs)
        _n_p = len(_top_prots)
        _drug_deg = _df_g["drug_name"].value_counts().to_dict()
        _prot_deg = _df_g["prot_name"].value_counts().to_dict()
        _max_deg  = max(max(_drug_deg.values(), default=1),
                        max(_prot_deg.values(), default=1))

        def _norm_deg(v):
            return v / _max_deg if _max_deg else 0

        if not _view3d:
            # ═══════════════════════════════════════════════════════════
            # 2D — Chord diagram: nodes on a circle arc, bezier edges
            # ═══════════════════════════════════════════════════════════
            _R2  = 2.5
            # Drugs: left arc 100°→260°, Proteins: right arc -80°→80°
            _d_a = np.linspace(np.radians(100), np.radians(260), _n_d)
            _p_a = np.linspace(np.radians(-80),  np.radians(80),  _n_p)
            _dpos = {
                n: (float(_R2 * np.cos(_d_a[i])), float(_R2 * np.sin(_d_a[i])))
                for i, n in enumerate(_top_drugs)
            }
            _ppos = {
                n: (float(_R2 * np.cos(_p_a[i])), float(_R2 * np.sin(_p_a[i])))
                for i, n in enumerate(_top_prots)
            }

            # Cubic bezier — control points pulled to center for chord effect
            def _bz(x0, y0, x1, y1, ns=12):
                cp = 0.15
                cx0, cy0 = x0 * cp, y0 * cp
                cx1, cy1 = x1 * cp, y1 * cp
                t = np.linspace(0, 1, ns)
                bx = (1-t)**3*x0 + 3*(1-t)**2*t*cx0 + 3*(1-t)*t**2*cx1 + t**3*x1
                by = (1-t)**3*y0 + 3*(1-t)**2*t*cy0 + 3*(1-t)*t**2*cy1 + t**3*y1
                return list(bx) + [None], list(by) + [None]

            _ex, _ey = [], []
            for _, _r in _df_g.iterrows():
                _bx, _by = _bz(*_dpos[_r["drug_name"]], *_ppos[_r["prot_name"]])
                _ex += _bx
                _ey += _by

            _fig2d = go.Figure()

            # Edges
            _fig2d.add_trace(go.Scatter(
                x=_ex, y=_ey, mode="lines",
                line=dict(color="rgba(99,102,241,0.10)", width=1.1),
                hoverinfo="none", showlegend=False,
            ))

            # Drug nodes — glow + solid
            _dx2  = [_dpos[n][0] for n in _top_drugs]
            _dy2  = [_dpos[n][1] for n in _top_drugs]
            _dsz  = [12 + int(22 * _norm_deg(_drug_deg.get(n, 1))) for n in _top_drugs]
            _dc   = [_drug_deg.get(n, 1) for n in _top_drugs]
            _dhov = [f"<b>💊 {n}</b><br>Liên kết: {_drug_deg.get(n, 0)}" for n in _top_drugs]

            _fig2d.add_trace(go.Scatter(   # glow halo
                x=_dx2, y=_dy2, mode="markers",
                marker=dict(size=[s * 2.8 for s in _dsz],
                            color="rgba(99,102,241,0.07)"),
                hoverinfo="none", showlegend=False,
            ))
            _fig2d.add_trace(go.Scatter(   # solid node
                x=_dx2, y=_dy2,
                mode="markers+text" if _show_lbl else "markers",
                name="💊 Thuốc",
                marker=dict(
                    size=_dsz, color=_dc,
                    colorscale=[[0, "#312e81"], [0.5, "#6366f1"], [1, "#c7d2fe"]],
                    cmin=0, cmax=_max_deg,
                    line=dict(width=1.5, color="rgba(255,255,255,0.35)"),
                    symbol="circle",
                ),
                text=_top_drugs if _show_lbl else [""] * _n_d,
                textposition="middle left",
                textfont=dict(size=8, color="#c7d2fe"),
                hovertext=_dhov, hoverinfo="text",
            ))

            # Protein nodes — glow + solid
            _px2  = [_ppos[n][0] for n in _top_prots]
            _py2  = [_ppos[n][1] for n in _top_prots]
            _psz  = [12 + int(22 * _norm_deg(_prot_deg.get(n, 1))) for n in _top_prots]
            _pc   = [_prot_deg.get(n, 1) for n in _top_prots]
            _phov = [f"<b>🔬 {n}</b><br>Liên kết: {_prot_deg.get(n, 0)}" for n in _top_prots]

            _fig2d.add_trace(go.Scatter(   # glow halo
                x=_px2, y=_py2, mode="markers",
                marker=dict(size=[s * 2.8 for s in _psz],
                            color="rgba(16,185,129,0.07)"),
                hoverinfo="none", showlegend=False,
            ))
            _fig2d.add_trace(go.Scatter(   # solid node
                x=_px2, y=_py2,
                mode="markers+text" if _show_lbl else "markers",
                name="🔬 Protein",
                marker=dict(
                    size=_psz, color=_pc,
                    colorscale=[[0, "#064e3b"], [0.5, "#10b981"], [1, "#a7f3d0"]],
                    cmin=0, cmax=_max_deg,
                    line=dict(width=1.5, color="rgba(255,255,255,0.35)"),
                    symbol="diamond",
                ),
                text=_top_prots if _show_lbl else [""] * _n_p,
                textposition="middle right",
                textfont=dict(size=8, color="#a7f3d0"),
                hovertext=_phov, hoverinfo="text",
            ))

            _fig2d.update_layout(
                height=640,
                plot_bgcolor="#070c1a",
                paper_bgcolor="#070c1a",
                font=dict(color="#e2e8f0"),
                margin=dict(l=20, r=20, t=55, b=20),
                title=dict(
                    text=(f"Bố cục cung tròn — {len(_df_g)} liên kết  ·  "
                          f"{_n_d} thuốc ●  ·  {_n_p} protein ◆  "
                          f"— kích thước nút ~ số kết nối"),
                    font=dict(size=12, color="#475569"), x=0.5,
                ),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                           range=[-3.4, 3.4]),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                           range=[-3.1, 3.1], scaleanchor="x", scaleratio=1),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.01,
                    xanchor="center", x=0.5,
                    font=dict(size=11),
                    bgcolor="rgba(7,12,26,0.85)",
                    bordercolor="rgba(99,102,241,0.2)", borderwidth=1,
                ),
                hovermode="closest",
            )
            _fig2d.add_annotation(
                x=0.03, y=0.97, xref="paper", yref="paper",
                text="💊 Thuốc", showarrow=False,
                font=dict(color="#a5b4fc", size=12),
                bgcolor="rgba(49,46,129,0.5)",
                bordercolor="#4f46e5", borderwidth=1, borderpad=5,
            )
            _fig2d.add_annotation(
                x=0.97, y=0.97, xref="paper", yref="paper",
                text="🔬 Protein", showarrow=False,
                font=dict(color="#6ee7b7", size=12),
                bgcolor="rgba(6,78,59,0.5)",
                bordercolor="#059669", borderwidth=1, borderpad=5,
            )
            st.plotly_chart(_fig2d, use_container_width=True)

        else:
            # ═══════════════════════════════════════════════════════════
            # 3D — Two-ring layout: drugs circle z=0, proteins circle z=4
            # ═══════════════════════════════════════════════════════════
            _r3   = 4.0
            _d3a  = np.linspace(0, 2 * np.pi, _n_d, endpoint=False)
            _p3a  = np.linspace(np.pi / max(1, _n_p),
                                 2 * np.pi + np.pi / max(1, _n_p),
                                 _n_p, endpoint=False)
            _dpos3 = {
                n: (float(_r3 * np.cos(_d3a[i])),
                    float(_r3 * np.sin(_d3a[i])), 0.0)
                for i, n in enumerate(_top_drugs)
            }
            _ppos3 = {
                n: (float(_r3 * np.cos(_p3a[i])),
                    float(_r3 * np.sin(_p3a[i])), 4.5)
                for i, n in enumerate(_top_prots)
            }

            _ex3, _ey3, _ez3 = [], [], []
            for _, _r in _df_g.iterrows():
                dx, dy, dz = _dpos3[_r["drug_name"]]
                px, py, pz = _ppos3[_r["prot_name"]]
                _ex3 += [dx, px, None]
                _ey3 += [dy, py, None]
                _ez3 += [dz, pz, None]

            _fig3d = go.Figure()

            # Edges 3D
            _fig3d.add_trace(go.Scatter3d(
                x=_ex3, y=_ey3, z=_ez3, mode="lines",
                line=dict(color="rgba(148,163,184,0.13)", width=1),
                hoverinfo="none", showlegend=False,
            ))

            # Drug nodes 3D
            _d3x  = [_dpos3[n][0] for n in _top_drugs]
            _d3y  = [_dpos3[n][1] for n in _top_drugs]
            _d3z  = [_dpos3[n][2] for n in _top_drugs]
            _d3sz = [6 + int(10 * _norm_deg(_drug_deg.get(n, 1))) for n in _top_drugs]
            _d3c  = [_drug_deg.get(n, 1) for n in _top_drugs]

            _fig3d.add_trace(go.Scatter3d(
                x=_d3x, y=_d3y, z=_d3z,
                mode="markers+text" if _show_lbl else "markers",
                name="💊 Thuốc",
                marker=dict(
                    size=_d3sz, color=_d3c,
                    colorscale=[[0, "#312e81"], [0.5, "#6366f1"], [1, "#c7d2fe"]],
                    cmin=0, cmax=_max_deg,
                    line=dict(width=0.5, color="rgba(255,255,255,0.3)"),
                    opacity=0.92,
                ),
                text=_top_drugs if _show_lbl else [""] * _n_d,
                textfont=dict(size=7, color="#c7d2fe"),
                hovertext=[f"<b>💊 {n}</b><br>Liên kết: {_drug_deg.get(n, 0)}"
                           for n in _top_drugs],
                hoverinfo="text",
            ))

            # Protein nodes 3D
            _p3x  = [_ppos3[n][0] for n in _top_prots]
            _p3y  = [_ppos3[n][1] for n in _top_prots]
            _p3z  = [_ppos3[n][2] for n in _top_prots]
            _p3sz = [6 + int(10 * _norm_deg(_prot_deg.get(n, 1))) for n in _top_prots]
            _p3c  = [_prot_deg.get(n, 1) for n in _top_prots]

            _fig3d.add_trace(go.Scatter3d(
                x=_p3x, y=_p3y, z=_p3z,
                mode="markers+text" if _show_lbl else "markers",
                name="🔬 Protein",
                marker=dict(
                    size=_p3sz, color=_p3c,
                    colorscale=[[0, "#064e3b"], [0.5, "#10b981"], [1, "#a7f3d0"]],
                    cmin=0, cmax=_max_deg,
                    line=dict(width=0.5, color="rgba(255,255,255,0.3)"),
                    opacity=0.92, symbol="square",
                ),
                text=_top_prots if _show_lbl else [""] * _n_p,
                textfont=dict(size=7, color="#a7f3d0"),
                hovertext=[f"<b>🔬 {n}</b><br>Liên kết: {_prot_deg.get(n, 0)}"
                           for n in _top_prots],
                hoverinfo="text",
            ))

            _fig3d.update_layout(
                height=660,
                paper_bgcolor="#070c1a",
                margin=dict(l=0, r=0, t=50, b=0),
                title=dict(
                    text=(f"🌐 Không gian 3D — {len(_df_g)} liên kết  ·  "
                          f"{_n_d} thuốc (vòng dưới, z=0)  ·  "
                          f"{_n_p} protein (vòng trên, z=4.5)"),
                    font=dict(size=12, color="#475569"), x=0.5,
                ),
                scene=dict(
                    bgcolor="#070c1a",
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)",
                               showticklabels=False, title="", zeroline=False,
                               backgroundcolor="#070c1a"),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)",
                               showticklabels=False, title="", zeroline=False,
                               backgroundcolor="#070c1a"),
                    zaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                               showticklabels=False, title="", zeroline=False,
                               backgroundcolor="#070c1a",
                               tickvals=[0, 4.5],
                               ticktext=["💊 Thuốc", "🔬 Protein"]),
                    camera=dict(eye=dict(x=1.6, y=1.6, z=1.1)),
                    aspectmode="manual",
                    aspectratio=dict(x=1.4, y=1.4, z=0.65),
                ),
                legend=dict(
                    font=dict(size=11),
                    bgcolor="rgba(7,12,26,0.85)",
                    bordercolor="rgba(99,102,241,0.2)", borderwidth=1,
                ),
            )
            st.plotly_chart(_fig3d, use_container_width=True)

        st.caption(
            "💡 Kéo slider để thay đổi số nút hiển thị.  "
            "Hover vào nút để xem tên đầy đủ.  "
            "Kích thước nút tỷ lệ với số liên kết.  "
            "Bật 🌐 3D để chuyển sang không gian 3 chiều có thể xoay/zoom."
        )

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
