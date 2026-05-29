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
from utils.molecule_utils import (
    smiles_to_image_b64, smiles_to_svg, smiles_to_2d_plotly,
    smiles_to_3d_plotly, smiles_to_3d_html_viewer,
    get_mol_properties, get_lipinski_analysis,
)

AI = _AI

# ── Bezier S-curve helper (for 2D network edges) ──────────────────────────
def _bezier_curve(x0: float, y0: float, x1: float, y1: float, n: int = 40):
    """Cubic Bezier S-curve; control points horizontally centred."""
    cx = (x0 + x1) * 0.5
    t  = np.linspace(0, 1, n)
    bx = (1-t)**3*x0 + 3*(1-t)**2*t*cx + 3*(1-t)*t**2*cx + t**3*x1
    by = (1-t)**3*y0 + 3*(1-t)**2*t*y0 + 3*(1-t)*t**2*y1 + t**3*y1
    return bx.tolist() + [None], by.tolist() + [None]

# ── Load SMILES lookup from metadata.json ─────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_smiles_map(ds: str) -> dict:
    """Returns {drug_name: smiles} from AMDGT_main/data/<ds>/metadata.json"""
    base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "AMDGT_main", "data", ds, "metadata.json",
    )
    try:
        with open(base, encoding="utf-8") as f:
            meta = json.load(f)
        return {
            d.get("name_en", d.get("id", "")): d.get("smiles", "")
            for d in meta.get("drugs", [])
            if d.get("smiles")
        }
    except Exception:
        return {}

# ── Load disease mapping from metadata + association CSVs ─────────────────
@st.cache_data(show_spinner=False)
def _load_disease_map(ds: str):
    """Returns (drug_name→idx, drug_idx→diseases, prot_id→idx, prot_idx→diseases,
                disease_en_names, disease_vn_names)."""
    _base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "AMDGT_main", "data", ds,
    )
    try:
        with open(os.path.join(_base, "metadata.json"), encoding="utf-8") as _f:
            _meta = json.load(_f)
        _dis_en   = {d["idx"]: d.get("name_en", d["id"])              for d in _meta.get("diseases", [])}
        _dis_vn   = {d["idx"]: d.get("name_vn", d.get("name_en", "")) for d in _meta.get("diseases", [])}
        _dn_to_idx = {d["name_en"]: d["idx"]                          for d in _meta.get("drugs",    [])}
        _pi_to_idx = {d["id"]:      d["idx"]                          for d in _meta.get("proteins", [])}

        _dda = pd.read_csv(os.path.join(_base, "DrugDiseaseAssociationNumber.csv"))
        _drug_to_dis: dict = {}
        for _, _r in _dda.iterrows():
            _drug_to_dis.setdefault(int(_r["drug"]), set()).add(int(_r["disease"]))

        _pda = pd.read_csv(os.path.join(_base, "ProteinDiseaseAssociationNumber.csv"))
        _prot_to_dis: dict = {}
        for _, _r in _pda.iterrows():
            _prot_to_dis.setdefault(int(_r["protein"]), set()).add(int(_r["disease"]))

        return _dn_to_idx, _drug_to_dis, _pi_to_idx, _prot_to_dis, _dis_en, _dis_vn
    except Exception:
        return {}, {}, {}, {}, {}, {}


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

tab_table, tab_graph, tab_drug, tab_prot, tab_disease = st.tabs([
    "📋 Bảng liên kết", "🕸️ Mạng lưới", "💊 Theo Thuốc", "🔬 Theo Protein", "🦠 Theo Bệnh"
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
    _g_ctrl1, _g_ctrl2, _g_ctrl3, _g_ctrl4 = st.columns([3, 1, 1, 1])
    _max_per = max(len(df["drug_name"].unique()), len(df["prot_name"].unique()))
    _top_n   = _g_ctrl1.slider(
        "Số nút mỗi nhóm (top theo số liên kết)",
        min_value=5, max_value=min(50, _max_per),
        value=min(15, _max_per), step=5, key="g_topn",
    )
    _view3d   = _g_ctrl2.toggle("🌐 3D", value=False, key="g_3d")
    _show_lbl = _g_ctrl3.checkbox("Nhãn", value=True,  key="g_lbl")
    _edge_col = _g_ctrl4.checkbox("Màu cạnh", value=True, key="g_ecol")

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
            # 2D — Bipartite layout: drugs left column, proteins right
            # ═══════════════════════════════════════════════════════════
            _R2  = 2.2

            # Drugs: evenly spaced on left side (x=-R2), Proteins: right side (x=+R2)
            _d_ys = np.linspace(1, -1, _n_d) * (_n_d / 2.0)
            _p_ys = np.linspace(1, -1, _n_p) * (_n_p / 2.0)
            _dpos = {n: (-_R2, float(_d_ys[i])) for i, n in enumerate(_top_drugs)}
            _ppos = {n: ( _R2, float(_p_ys[i])) for i, n in enumerate(_top_prots)}

            # Build per-edge traces for colored edges
            _fig2d = go.Figure()

            _drug_color_map = {n: f"rgba(99,102,241,{0.25 + 0.55*_norm_deg(_drug_deg.get(n,1))})"
                               for n in _top_drugs}

            if _edge_col:
                # Bezier S-curves grouped per drug color (batched — one trace per drug)
                _drug_bx: dict = {}
                _drug_by: dict = {}
                for _, _r in _df_g.iterrows():
                    _edx, _edy = _dpos[_r["drug_name"]]
                    _epx, _epy = _ppos[_r["prot_name"]]
                    _bx, _by = _bezier_curve(_edx, _edy, _epx, _epy)
                    _drug_bx.setdefault(_r["drug_name"], []).extend(_bx)
                    _drug_by.setdefault(_r["drug_name"], []).extend(_by)
                for _dn in _drug_bx:
                    _fig2d.add_trace(go.Scatter(
                        x=_drug_bx[_dn], y=_drug_by[_dn], mode="lines",
                        line=dict(color=_drug_color_map[_dn], width=1.6),
                        hoverinfo="none", showlegend=False,
                    ))
            else:
                _bx_all, _by_all = [], []
                for _, _r in _df_g.iterrows():
                    _edx, _edy = _dpos[_r["drug_name"]]
                    _epx, _epy = _ppos[_r["prot_name"]]
                    _bx, _by = _bezier_curve(_edx, _edy, _epx, _epy)
                    _bx_all.extend(_bx)
                    _by_all.extend(_by)
                _fig2d.add_trace(go.Scatter(
                    x=_bx_all, y=_by_all, mode="lines",
                    line=dict(color="rgba(148,163,184,0.20)", width=1.3),
                    hoverinfo="none", showlegend=False,
                ))

            # Drug nodes
            _dx2  = [_dpos[n][0] for n in _top_drugs]
            _dy2  = [_dpos[n][1] for n in _top_drugs]
            _dsz  = [14 + int(24 * _norm_deg(_drug_deg.get(n, 1))) for n in _top_drugs]
            _dc   = [_drug_deg.get(n, 1) for n in _top_drugs]
            _dhov = [f"<b>💊 {n}</b><br>Liên kết mới: <b>{_drug_deg.get(n, 0)}</b>" for n in _top_drugs]

            # Glow halo
            _fig2d.add_trace(go.Scatter(
                x=_dx2, y=_dy2, mode="markers",
                marker=dict(size=[s * 2.2 for s in _dsz], color="rgba(99,102,241,0.09)"),
                hoverinfo="none", showlegend=False,
            ))
            _fig2d.add_trace(go.Scatter(
                x=_dx2, y=_dy2,
                mode="markers+text" if _show_lbl else "markers",
                name="💊 Thuốc",
                marker=dict(
                    size=_dsz, color=_dc,
                    colorscale=[[0, "#312e81"], [0.4, "#6366f1"], [1, "#c7d2fe"]],
                    cmin=0, cmax=_max_deg,
                    colorbar=dict(
                        title=dict(text="Liên kết", font=dict(color="#94a3b8", size=11)),
                        thickness=10, x=-0.04, tickfont=dict(color="#94a3b8", size=9),
                    ),
                    line=dict(width=2, color="rgba(255,255,255,0.5)"),
                    symbol="circle",
                ),
                text=["  " + n for n in _top_drugs] if _show_lbl else [""] * _n_d,
                textposition="middle right",
                textfont=dict(size=9, color="#c7d2fe"),
                hovertext=_dhov, hoverinfo="text",
            ))

            # Protein nodes
            _px2  = [_ppos[n][0] for n in _top_prots]
            _py2  = [_ppos[n][1] for n in _top_prots]
            _psz  = [14 + int(24 * _norm_deg(_prot_deg.get(n, 1))) for n in _top_prots]
            _pc   = [_prot_deg.get(n, 1) for n in _top_prots]
            _phov = [f"<b>🔬 {n}</b><br>Liên kết mới: <b>{_prot_deg.get(n, 0)}</b>" for n in _top_prots]

            _fig2d.add_trace(go.Scatter(
                x=_px2, y=_py2, mode="markers",
                marker=dict(size=[s * 2.2 for s in _psz], color="rgba(16,185,129,0.09)"),
                hoverinfo="none", showlegend=False,
            ))
            _fig2d.add_trace(go.Scatter(
                x=_px2, y=_py2,
                mode="markers+text" if _show_lbl else "markers",
                name="🔬 Protein",
                marker=dict(
                    size=_psz, color=_pc,
                    colorscale=[[0, "#064e3b"], [0.4, "#10b981"], [1, "#a7f3d0"]],
                    cmin=0, cmax=_max_deg,
                    line=dict(width=2, color="rgba(255,255,255,0.5)"),
                    symbol="diamond",
                ),
                text=[n + "  " for n in _top_prots] if _show_lbl else [""] * _n_p,
                textposition="middle left",
                textfont=dict(size=9, color="#a7f3d0"),
                hovertext=_phov, hoverinfo="text",
            ))

            # Axis guides (invisible) + column labels
            _y_range = max(_n_d, _n_p) / 2.0 + 1.5
            _fig2d.add_annotation(
                x=-_R2, y=_y_range * 1.06, text="💊 THUỐC", showarrow=False,
                font=dict(color="#a5b4fc", size=13, family="Inter"),
                bgcolor="rgba(49,46,129,0.6)", bordercolor="#4f46e5",
                borderwidth=1, borderpad=6,
            )
            _fig2d.add_annotation(
                x=_R2, y=_y_range * 1.06, text="🔬 PROTEIN", showarrow=False,
                font=dict(color="#6ee7b7", size=13, family="Inter"),
                bgcolor="rgba(6,78,59,0.6)", bordercolor="#059669",
                borderwidth=1, borderpad=6,
            )

            _fig2d.update_layout(
                height=max(560, 28 * max(_n_d, _n_p)),
                plot_bgcolor="#07101f",
                paper_bgcolor="#07101f",
                font=dict(color="#e2e8f0"),
                margin=dict(l=60, r=20, t=60, b=20),
                title=dict(
                    text=(f"Bố cục song song — <b>{len(_df_g)}</b> liên kết  ·  "
                          f"<b>{_n_d}</b> thuốc ●  ·  <b>{_n_p}</b> protein ◆  "
                          "— kích thước & màu sắc ~ số kết nối"),
                    font=dict(size=12, color="#64748b"), x=0.5,
                ),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                           range=[-_R2 - 1.6, _R2 + 1.6]),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                           range=[-_y_range * 1.12, _y_range * 1.15]),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5,
                    font=dict(size=12),
                    bgcolor="rgba(7,16,31,0.9)",
                    bordercolor="rgba(99,102,241,0.25)", borderwidth=1,
                ),
                hovermode="closest",
            )
            st.plotly_chart(_fig2d, use_container_width=True)

        else:
            # ═══════════════════════════════════════════════════════════
            # 3D — Spring-like layout: drugs on z=0 hemisphere,
            #       proteins on z=5 hemisphere, edges color-coded
            # ═══════════════════════════════════════════════════════════
            _r3   = 3.5
            # Arrange drugs on bottom ring with slight radial jitter for depth
            _d3a  = np.linspace(0, 2 * np.pi, _n_d, endpoint=False)
            _p3a  = np.linspace(np.pi / max(1, _n_p),
                                 2 * np.pi + np.pi / max(1, _n_p),
                                 _n_p, endpoint=False)
            # Fibonacci golden-angle for evenly spread protein nodes
            _phi  = (1 + 5 ** 0.5) / 2
            _p3a  = np.array([2 * np.pi * i / _phi for i in range(_n_p)])
            _d3a  = np.array([2 * np.pi * i / _phi for i in range(_n_d)])
            # Proteins: alternate outer/inner ring for visible depth
            _p3r  = np.where(np.arange(_n_p) % 2 == 0, _r3, _r3 * 0.62)
            _dpos3 = {
                n: (float(_r3 * np.cos(_d3a[i])),
                    float(_r3 * np.sin(_d3a[i])), 0.0)
                for i, n in enumerate(_top_drugs)
            }
            _ppos3 = {
                n: (float(_p3r[i] * np.cos(_p3a[i])),
                    float(_p3r[i] * np.sin(_p3a[i])), 5.5)
                for i, n in enumerate(_top_prots)
            }

            _fig3d = go.Figure()

            # Edge traces — per drug color
            _drug_rgba3 = {
                n: f"rgba({int(99 + 80*_norm_deg(_drug_deg.get(n,1)))},{int(102 - 40*_norm_deg(_drug_deg.get(n,1)))},{int(241 - 80*_norm_deg(_drug_deg.get(n,1)))},0.35)"
                for n in _top_drugs
            }
            if _edge_col:
                for _, _r in _df_g.iterrows():
                    _e3dx, _e3dy, _e3dz = _dpos3[_r["drug_name"]]
                    _e3px, _e3py, _e3pz = _ppos3[_r["prot_name"]]
                    _fig3d.add_trace(go.Scatter3d(
                        x=[_e3dx, _e3px, None], y=[_e3dy, _e3py, None], z=[_e3dz, _e3pz, None],
                        mode="lines",
                        line=dict(color=_drug_rgba3[_r["drug_name"]], width=2),
                        hoverinfo="none", showlegend=False,
                    ))
            else:
                _ex3, _ey3, _ez3 = [], [], []
                for _, _r in _df_g.iterrows():
                    _e3dx, _e3dy, _e3dz = _dpos3[_r["drug_name"]]
                    _e3px, _e3py, _e3pz = _ppos3[_r["prot_name"]]
                    _ex3 += [_e3dx, _e3px, None]
                    _ey3 += [_e3dy, _e3py, None]
                    _ez3 += [_e3dz, _e3pz, None]
                _fig3d.add_trace(go.Scatter3d(
                    x=_ex3, y=_ey3, z=_ez3, mode="lines",
                    line=dict(color="rgba(148,163,184,0.20)", width=1.5),
                    hoverinfo="none", showlegend=False,
                ))

            # Drug nodes 3D
            _d3x  = [_dpos3[n][0] for n in _top_drugs]
            _d3y  = [_dpos3[n][1] for n in _top_drugs]
            _d3z  = [_dpos3[n][2] for n in _top_drugs]
            _d3sz = [8 + int(14 * _norm_deg(_drug_deg.get(n, 1))) for n in _top_drugs]
            _d3c  = [_drug_deg.get(n, 1) for n in _top_drugs]

            _fig3d.add_trace(go.Scatter3d(
                x=_d3x, y=_d3y, z=_d3z,
                mode="markers+text" if _show_lbl else "markers",
                name="💊 Thuốc",
                marker=dict(
                    size=_d3sz, color=_d3c,
                    colorscale=[[0, "#1e1b4b"], [0.4, "#4f46e5"], [1, "#c7d2fe"]],
                    cmin=0, cmax=_max_deg,
                    colorbar=dict(
                        title=dict(text="Kết nối", font=dict(color="#94a3b8", size=10)),
                        x=1.02, thickness=10, tickfont=dict(color="#94a3b8", size=9),
                        len=0.4, y=0.25,
                    ),
                    line=dict(width=1, color="rgba(255,255,255,0.5)"),
                    opacity=0.95,
                    symbol="circle",
                ),
                text=_top_drugs if _show_lbl else [""] * _n_d,
                textfont=dict(size=8, color="#c7d2fe"),
                hovertext=[f"<b>💊 {n}</b><br>Liên kết mới: <b>{_drug_deg.get(n, 0)}</b>"
                           for n in _top_drugs],
                hoverinfo="text",
            ))

            # Protein nodes 3D
            _p3x  = [_ppos3[n][0] for n in _top_prots]
            _p3y  = [_ppos3[n][1] for n in _top_prots]
            _p3z  = [_ppos3[n][2] for n in _top_prots]
            _p3sz = [8 + int(14 * _norm_deg(_prot_deg.get(n, 1))) for n in _top_prots]
            _p3c  = [_prot_deg.get(n, 1) for n in _top_prots]

            _fig3d.add_trace(go.Scatter3d(
                x=_p3x, y=_p3y, z=_p3z,
                mode="markers+text" if _show_lbl else "markers",
                name="🔬 Protein",
                marker=dict(
                    size=_p3sz, color=_p3c,
                    colorscale=[[0, "#022c22"], [0.4, "#059669"], [1, "#a7f3d0"]],
                    cmin=0, cmax=_max_deg,
                    line=dict(width=1, color="rgba(255,255,255,0.5)"),
                    opacity=0.95,
                    symbol="diamond",
                ),
                text=_top_prots if _show_lbl else [""] * _n_p,
                textfont=dict(size=8, color="#a7f3d0"),
                hovertext=[f"<b>🔬 {n}</b><br>Liên kết mới: <b>{_prot_deg.get(n, 0)}</b>"
                           for n in _top_prots],
                hoverinfo="text",
            ))

            # Add invisible floor disk for depth reference
            _theta_disk = np.linspace(0, 2 * np.pi, 60)
            _fig3d.add_trace(go.Scatter3d(
                x=(_r3 * 1.1 * np.cos(_theta_disk)).tolist(),
                y=(_r3 * 1.1 * np.sin(_theta_disk)).tolist(),
                z=[0.0] * 60,
                mode="lines",
                line=dict(color="rgba(99,102,241,0.15)", width=1),
                hoverinfo="none", showlegend=False,
            ))
            _fig3d.add_trace(go.Scatter3d(
                x=(_r3 * 1.1 * np.cos(_theta_disk)).tolist(),
                y=(_r3 * 1.1 * np.sin(_theta_disk)).tolist(),
                z=[5.5] * 60,
                mode="lines",
                line=dict(color="rgba(16,185,129,0.15)", width=1),
                hoverinfo="none", showlegend=False,
            ))

            _fig3d.update_layout(
                height=700,
                paper_bgcolor="#07101f",
                margin=dict(l=0, r=0, t=55, b=0),
                title=dict(
                    text=(f"🌐 Không gian 3D — <b>{len(_df_g)}</b> liên kết  ·  "
                          f"<b>{_n_d}</b> thuốc (vòng xanh, z=0)  ·  "
                          f"<b>{_n_p}</b> protein (vòng lá, z=5.5)  "
                          "— Kéo để xoay, cuộn để zoom"),
                    font=dict(size=12, color="#64748b"), x=0.5,
                ),
                scene=dict(
                    bgcolor="#07101f",
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                               showticklabels=False, title="",
                               backgroundcolor="#07101f", zeroline=False,
                               showspikes=False),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                               showticklabels=False, title="",
                               backgroundcolor="#07101f", zeroline=False,
                               showspikes=False),
                    zaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)",
                               showticklabels=True, title="",
                               backgroundcolor="#07101f", zeroline=False,
                               tickvals=[0, 5.5],
                               ticktext=["💊 Thuốc (z=0)", "🔬 Protein (z=5.5)"],
                               tickfont=dict(color="#94a3b8", size=10)),
                    camera=dict(
                        eye=dict(x=1.4, y=1.4, z=0.9),
                        up=dict(x=0, y=0, z=1),
                    ),
                    aspectmode="manual",
                    aspectratio=dict(x=1.3, y=1.3, z=0.7),
                ),
                legend=dict(
                    font=dict(size=12),
                    bgcolor="rgba(7,16,31,0.9)",
                    bordercolor="rgba(99,102,241,0.25)", borderwidth=1,
                    x=0.01, y=0.99,
                ),
            )
            st.plotly_chart(_fig3d, use_container_width=True)

        st.caption(
            "💡 Kéo slider để thay đổi số nút hiển thị. "
            "Hover vào nút để xem tên & số liên kết. "
            "Kích thước và màu nút tỷ lệ với số liên kết mới. "
            "✅ Nhãn bật mặc định — tắt nếu quá chật. "
            "🎨 Màu cạnh — mỗi thuốc một màu để dễ phân biệt. "
            "🌐 3D — kéo để xoay, cuộn chuột để zoom."
        )

    # ── Molecular Structure Viewer ────────────────────────────────────────
    st.divider()
    st.markdown("### 🔬 Xem cấu trúc phân tử thuốc")
    st.caption("Chọn một thuốc từ kết quả VGAE để hiển thị cấu trúc 2D/3D")

    _smiles_map = _load_smiles_map(dataset)
    _drug_list  = sorted(df["drug_name"].unique().tolist())
    _drug_with_smiles = [d for d in _drug_list if _smiles_map.get(d)]
    _drug_no_smiles   = [d for d in _drug_list if not _smiles_map.get(d)]

    _mol_sel_col, _mol_mode_col = st.columns([3, 2])
    with _mol_sel_col:
        _sel_mol = st.selectbox(
            "Chọn thuốc",
            ["— Chọn thuốc —"] + _drug_with_smiles + (
                ["── Không có SMILES ──"] + _drug_no_smiles if _drug_no_smiles else []
            ),
            key="vgae_mol_select",
        )
    with _mol_mode_col:
        _mol_mode = st.radio(
            "Chế độ hiển thị",
            ["🖼️ SVG 2D", "📐 Interactive 2D", "🌐 3D"],
            horizontal=True,
            key="vgae_mol_mode",
            label_visibility="collapsed",
        )

    if _sel_mol and _sel_mol not in ("— Chọn thuốc —", "── Không có SMILES ──") and _smiles_map.get(_sel_mol):
        _smiles = _smiles_map[_sel_mol]
        _mol_left, _mol_right = st.columns([3, 2])

        with _mol_left:
            st.markdown(f"**💊 {_sel_mol}**  ", unsafe_allow_html=False)
            st.code(_smiles, language=None)

            if "SVG" in _mol_mode:
                _svg = smiles_to_svg(_smiles, size=(480, 320), dark_bg=True)
                if _svg:
                    st.markdown(
                        f'<div style="background:#0a0e24;border-radius:12px;padding:10px;'
                        f'border:1px solid rgba(99,102,241,0.25);">{_svg}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    _img = smiles_to_image_b64(_smiles, (480, 320))
                    if _img:
                        st.image(f"data:image/png;base64,{_img}", use_container_width=True)
                    else:
                        st.warning("Cần cài RDKit để hiển thị cấu trúc 2D")

            elif "Interactive 2D" in _mol_mode:
                _fig2d_mol = smiles_to_2d_plotly(_smiles, title=_sel_mol, dark=True)
                if _fig2d_mol:
                    st.plotly_chart(
                        _fig2d_mol, use_container_width=True,
                        config={
                            "displayModeBar": True,
                            "modeBarButtonsToAdd": ["zoomIn2d", "zoomOut2d", "resetScale2d"],
                            "scrollZoom": True,
                            "toImageButtonOptions": {"format": "png", "width": 800, "height": 600},
                        },
                    )
                else:
                    st.warning("Cần cài RDKit để hiển thị 2D interactive")

            else:  # 3D
                _fig3d_mol = smiles_to_3d_plotly(_smiles, title=_sel_mol, dark=True)
                if _fig3d_mol:
                    st.plotly_chart(
                        _fig3d_mol, use_container_width=True,
                        config={
                            "displayModeBar": True,
                            "modeBarButtonsToAdd": ["zoomIn3d", "zoomOut3d", "resetCameraDefault3d"],
                            "scrollZoom": True,
                            "toImageButtonOptions": {"format": "png", "width": 900, "height": 700},
                        },
                    )
                else:
                    _html_3d = smiles_to_3d_html_viewer(_smiles, width=580, height=420, dark=True)
                    if _html_3d:
                        import streamlit.components.v1 as _v1_comp
                        _v1_comp.html(_html_3d, width=600, height=440, scrolling=False)
                    else:
                        st.warning("Không thể tạo cấu trúc 3D (SMILES không hợp lệ)")

        with _mol_right:
            # ── Properties ────────────────────────────────────────────
            _props = get_mol_properties(_smiles)
            if _props:
                st.markdown("**📐 Thuộc tính phân tử**")
                _prop_items = [
                    ("⚛️", _props.get("num_atoms", "?"),            "Nguyên tử"),
                    ("🔗", _props.get("num_bonds", "?"),            "Liên kết"),
                    ("⚖️", f'{_props.get("mol_weight", "?")} Da',   "Khối lượng"),
                    ("💧", _props.get("logp", "?"),                 "LogP"),
                    ("🔄", _props.get("num_rings", "?"),            "Vòng"),
                    ("📊", _props.get("qed", "?"),                  "QED"),
                ]
                _p2 = st.columns(2)
                for _pi, (_ic, _vl, _lb) in enumerate(_prop_items):
                    with _p2[_pi % 2]:
                        st.markdown(
                            f'<div style="background:rgba(99,102,241,0.09);border:1px solid rgba(99,102,241,0.2);'
                            f'border-radius:10px;padding:10px 8px;text-align:center;margin-bottom:8px;">'
                            f'<div style="font-size:1.1rem">{_ic}</div>'
                            f'<div style="font-size:1.15rem;font-weight:700;color:#c7d2fe">{_vl}</div>'
                            f'<div style="font-size:0.72rem;color:#64748b">{_lb}</div></div>',
                            unsafe_allow_html=True,
                        )

            # ── Lipinski Rule-of-5 ────────────────────────────────────
            _lipo = get_lipinski_analysis(_smiles)
            if _lipo:
                _passed    = _lipo.get("passed", 0)
                _total     = _lipo.get("total", 6)
                _druglike  = _lipo.get("drug_like", False)
                _dl_color  = "#4ade80" if _druglike else "#fb923c"
                _dl_badge  = "✅ Drug-like" if _druglike else "⚠️ Non drug-like"
                st.markdown(
                    f'<div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);'
                    f'border-radius:12px;padding:12px 14px;margin-top:8px;">'
                    f'<div style="font-size:0.8rem;color:#818cf8;font-weight:600;margin-bottom:6px;">'
                    f'💊 Lipinski Rule-of-5</div>'
                    f'<div style="font-size:1.4rem;font-weight:800;color:{_dl_color};margin-bottom:4px;">'
                    f'{_dl_badge}</div>'
                    f'<div style="font-size:0.78rem;color:#94a3b8;margin-bottom:8px;">'
                    f'Đạt {_passed}/{_total} tiêu chí</div>',
                    unsafe_allow_html=True,
                )
                for _rule in _lipo.get("rules", []):
                    _ok   = _rule.get("passed", False)
                    _icon = "✅" if _ok else "❌"
                    _rlabel = _rule.get("name", "")
                    _rval   = _rule.get("value", "")
                    _rlimit = _rule.get("limit", "")
                    st.markdown(
                        f'<div style="font-size:0.77rem;color:{"#6ee7b7" if _ok else "#fca5a5"};'
                        f'margin-bottom:3px;">{_icon} {_rlabel}: {_rval} (≤{_rlimit})</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown('</div>', unsafe_allow_html=True)

            # ── Protein connections for this drug ─────────────────────
            _prots_for_drug = df[df["drug_name"] == _sel_mol][["prot_name", "prot_id"]]
            st.markdown(f"**🔬 Protein kết nối mới ({len(_prots_for_drug)})**")
            st.dataframe(
                _prots_for_drug.rename(columns={"prot_name": "Protein", "prot_id": "ID"}),
                use_container_width=True,
                height=min(200, 40 + 35 * len(_prots_for_drug)),
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

# ── Disease-centric view ──────────────────────────────────────────────
with tab_disease:
    st.markdown("### 🦠 Phân tích liên kết theo Bệnh")
    st.caption(
        "Chiếu các liên kết thuốc–protein mới vào không gian bệnh: "
        "nếu thuốc A đã nhắm đến bệnh X → liên kết A→Protein mới là tín hiệu cho bệnh X. "
        "Nếu Protein B liên quan đến bệnh Y → liên kết Thuốc mới→B cũng là tín hiệu cho bệnh Y."
    )

    _dn_idx, _d_to_dis, _pi_to_idx, _p_to_dis, _dis_en, _dis_vn = _load_disease_map(dataset)

    if not _dis_en:
        st.warning("⚠️ Không tải được dữ liệu bệnh từ metadata.json của dataset này.")
        st.stop()

    # ── Aggregate signals ──────────────────────────────────────────────
    _agg_drug: dict = {}   # disease_idx -> set of drug_names
    _agg_prot: dict = {}   # disease_idx -> set of prot_ids
    for _, _er in df.iterrows():
        _dname = _er["drug_name"]
        _didx  = _dn_idx.get(_dname)
        if _didx is not None:
            for _disc in _d_to_dis.get(_didx, set()):
                _agg_drug.setdefault(_disc, set()).add(_dname)
        _pid  = _er.get("prot_id", "")
        _pidx = _pi_to_idx.get(_pid)
        if _pidx is not None:
            for _disc in _p_to_dis.get(_pidx, set()):
                _agg_prot.setdefault(_disc, set()).add(_pid)

    # ── Build summary dataframe ────────────────────────────────────────
    _dis_rows = []
    for _disc in sorted(set(list(_agg_drug.keys()) + list(_agg_prot.keys()))):
        _nd  = len(_agg_drug.get(_disc, set()))
        _np_ = len(_agg_prot.get(_disc, set()))
        _dis_rows.append({
            "idx":                  _disc,
            "Bệnh (VN)":           _dis_vn.get(_disc, _dis_en.get(_disc, f"#{_disc}")),
            "Bệnh (EN)":           _dis_en.get(_disc, f"#{_disc}"),
            "Thuốc mới liên kết":  _nd,
            "Protein mới liên kết": _np_,
            "Tổng tín hiệu":        _nd + _np_,
        })
    _df_dis = (pd.DataFrame(_dis_rows)
               .sort_values("Tổng tín hiệu", ascending=False)
               .reset_index(drop=True))

    if _df_dis.empty:
        st.info("ℹ️ Không có dữ liệu bệnh phù hợp với dataset này.")
    else:
        # ── Metrics ───────────────────────────────────────────────────
        _dc1, _dc2, _dc3, _dc4 = st.columns(4)
        _dc1.metric("Bệnh được ảnh hưởng",    len(_df_dis))
        _dc2.metric("Qua kênh thuốc",          _df_dis["Thuốc mới liên kết"].gt(0).sum())
        _dc3.metric("Qua kênh protein",        _df_dis["Protein mới liên kết"].gt(0).sum())
        _dc4.metric("Tổng liên kết VGAE mới",  len(df))

        st.divider()
        _vis_col, _scatter_col = st.columns([3, 2])

        # ── Bar chart: grouped drug + protein signals ──────────────────
        with _vis_col:
            _top20 = _df_dis.head(20)
            _fig_bar = go.Figure()
            _fig_bar.add_trace(go.Bar(
                y=_top20["Bệnh (VN)"],
                x=_top20["Thuốc mới liên kết"],
                orientation="h", name="💊 Qua thuốc",
                marker=dict(
                    color=_top20["Thuốc mới liên kết"],
                    colorscale=[[0, "#312e81"], [0.5, "#6366f1"], [1, "#c7d2fe"]],
                    line=dict(width=0),
                ),
                hovertemplate="<b>%{y}</b><br>💊 Thuốc mới: <b>%{x}</b><extra></extra>",
            ))
            _fig_bar.add_trace(go.Bar(
                y=_top20["Bệnh (VN)"],
                x=_top20["Protein mới liên kết"],
                orientation="h", name="🔬 Qua protein",
                marker=dict(
                    color=_top20["Protein mới liên kết"],
                    colorscale=[[0, "#064e3b"], [0.5, "#10b981"], [1, "#a7f3d0"]],
                    line=dict(width=0),
                ),
                hovertemplate="<b>%{y}</b><br>🔬 Protein mới: <b>%{x}</b><extra></extra>",
            ))
            _fig_bar.update_layout(
                barmode="group",
                title=dict(
                    text="Top 20 bệnh — tín hiệu từ thuốc & protein mới",
                    font=dict(size=13, color="#94a3b8"), x=0.5,
                ),
                height=560,
                plot_bgcolor="#07101f", paper_bgcolor="#07101f",
                font=dict(color="#e2e8f0"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Số lượng"),
                yaxis=dict(autorange="reversed", tickfont=dict(size=9)),
                legend=dict(
                    orientation="h", y=1.04, x=0.5, xanchor="center",
                    bgcolor="rgba(7,16,31,0.8)",
                    bordercolor="rgba(99,102,241,0.2)", borderwidth=1,
                ),
                margin=dict(l=200, r=20, t=60, b=30),
            )
            st.plotly_chart(_fig_bar, use_container_width=True)

        # ── Scatter: drug signal vs protein signal per disease ──────────
        with _scatter_col:
            _fig_sc = px.scatter(
                _df_dis.head(60),
                x="Thuốc mới liên kết", y="Protein mới liên kết",
                text="Bệnh (VN)", size="Tổng tín hiệu",
                color="Tổng tín hiệu", color_continuous_scale="Viridis",
                hover_data={"Bệnh (EN)": True, "Tổng tín hiệu": True},
                title="Thuốc ↔ Protein (mỗi điểm = 1 bệnh)",
                size_max=28,
            )
            _fig_sc.update_traces(
                textposition="top center",
                textfont=dict(size=8, color="#94a3b8"),
                marker=dict(opacity=0.85,
                            line=dict(width=0.5, color="rgba(255,255,255,0.3)")),
            )
            _fig_sc.update_layout(
                height=560, plot_bgcolor="#07101f", paper_bgcolor="#07101f",
                font=dict(color="#e2e8f0"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                coloraxis_colorbar=dict(
                    thickness=10,
                    tickfont=dict(size=9, color="#94a3b8"),
                    title=dict(text="Tổng", font=dict(size=10, color="#94a3b8")),
                ),
                margin=dict(l=40, r=20, t=50, b=40),
            )
            st.plotly_chart(_fig_sc, use_container_width=True)

        # ── Drill-down ─────────────────────────────────────────────────
        st.divider()
        st.markdown("#### 🔍 Chi tiết theo bệnh")
        _sel_dis = st.selectbox(
            "Chọn bệnh để xem danh sách liên kết VGAE liên quan",
            ["(Tất cả)"] + _df_dis["Bệnh (VN)"].tolist(),
            key="vgae_disease_sel",
        )

        if _sel_dis == "(Tất cả)":
            st.dataframe(
                _df_dis[["Bệnh (VN)", "Bệnh (EN)",
                          "Thuốc mới liên kết", "Protein mới liên kết", "Tổng tín hiệu"]],
                use_container_width=True, height=400,
            )
        else:
            _sel_row = _df_dis[_df_dis["Bệnh (VN)"] == _sel_dis].iloc[0]
            _sel_idx = int(_sel_row["idx"])
            st.markdown(
                f'<div style="background:rgba(99,102,241,0.1);border-left:3px solid #6366f1;'
                f'padding:10px 16px;border-radius:8px;margin-bottom:14px;">'
                f'<b style="color:#c7d2fe;font-size:1rem">{_sel_dis}</b>'
                f'<span style="color:#64748b;font-size:0.8rem">  ({_sel_row["Bệnh (EN)"]})</span><br>'
                f'<span style="color:#94a3b8;font-size:0.8rem">'
                f'💊 Thuốc mới: <b style="color:#c7d2fe">{_sel_row["Thuốc mới liên kết"]}</b>'
                f'&nbsp;&nbsp;|&nbsp;&nbsp;'
                f'🔬 Protein mới: <b style="color:#6ee7b7">{_sel_row["Protein mới liên kết"]}</b>'
                f'</span></div>',
                unsafe_allow_html=True,
            )
            _sub_drugs = _agg_drug.get(_sel_idx, set())
            _sub_prots = _agg_prot.get(_sel_idx, set())
            _sub_edges = df[
                df["drug_name"].isin(_sub_drugs) | df["prot_id"].isin(_sub_prots)
            ][["drug_name", "drug_id", "prot_name", "prot_id"]].copy()
            _sub_edges.columns = ["Thuốc", "Drug ID", "Protein mới kết nối", "Protein ID"]
            st.dataframe(_sub_edges.drop_duplicates(), use_container_width=True, height=340)

