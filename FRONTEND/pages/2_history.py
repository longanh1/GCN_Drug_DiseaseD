"""
Page 2 — Lịch sử dự đoán
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from utils.api_client import get_history, clear_history

st.markdown('<base href="/">', unsafe_allow_html=True)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*, body { font-family: 'Inter', sans-serif !important; }
[data-testid="stAppViewContainer"] { background: #f1f5f9; }
[data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid #1e293b; }
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
    background: #ffffff; border-radius: 14px; padding: 18px 20px;
    border: 1px solid #e2e8f0; text-align: center;
    box-shadow: 0 2px 12px rgba(99,102,241,0.08);
    transition: box-shadow 0.2s;
}
.hist-stat-card:hover { box-shadow: 0 4px 20px rgba(99,102,241,0.16); }
.hist-stat-val {
    font-size: 1.8rem; font-weight: 800;
    background: linear-gradient(135deg, #4338ca, #7c3aed);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hist-stat-lbl { font-size: 0.72rem; color: #475569; margin-top: 5px; font-weight: 500; }
.hist-stat-icon { font-size: 1.4rem; margin-bottom: 6px; }

.dash-section {
    background: #ffffff; border-radius: 16px; padding: 20px 22px;
    border: 1px solid #e2e8f0; margin-bottom: 18px;
    box-shadow: 0 2px 10px rgba(99,102,241,0.06);
}
.dash-section-title {
    font-size: 0.82rem; font-weight: 700; color: #4338ca;
    text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 14px;
}

.top-drug-row {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 12px; border-radius: 8px; margin-bottom: 6px;
    background: rgba(99,102,241,0.04); border: 1px solid rgba(99,102,241,0.1);
    transition: background 0.15s;
}
.top-drug-row:hover { background: rgba(99,102,241,0.1); }
.top-drug-rank { font-size: 0.8rem; font-weight: 700; color: #64748b; min-width: 24px; }
.top-drug-rank.gold { color: #f59e0b; }
.top-drug-rank.silver { color: #94a3b8; }
.top-drug-rank.bronze { color: #b45309; }
.top-drug-name { flex: 1; font-size: 0.88rem; font-weight: 600; color: #1e293b; }
.top-drug-count {
    font-size: 0.88rem; font-weight: 700; color: #6366f1;
    background: rgba(99,102,241,0.1); padding: 2px 8px;
    border-radius: 6px;
}

.empty-state {
    background: #f8fafc; border-radius: 20px; padding: 80px 30px;
    border: 1px dashed rgba(99,102,241,0.3); text-align: center; margin-top: 24px;
}
.empty-icon  { font-size: 3.5rem; margin-bottom: 16px; opacity: 0.6; }
.empty-title { color: #1e293b; font-size: 1.1rem; font-weight: 600; margin-bottom: 8px; }
.empty-sub   { color: #475569; font-size: 0.85rem; }

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
history = get_history(500)
df = pd.DataFrame(history) if history else pd.DataFrame()

# ── Page header ────────────────────────────────────────────────────────
col_h, col_a = st.columns([3, 1])
with col_h:
    st.markdown("""
    <div class="page-header">
        <div class="page-header-left">
            <h2>📋 Lịch sử &amp; Dashboard</h2>
            <p>Tổng quan · Phân tích lịch sử tra cứu · Thuốc phổ biến · Xu hướng theo thời gian</p>
        </div>
    </div>""", unsafe_allow_html=True)
with col_a:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑  Xóa tất cả", key="clear_hist", use_container_width=True):
        clear_history()
        st.rerun()

# ── KPI Stats ─────────────────────────────────────────────────────────
total      = len(df) if not df.empty else 0
models_cnt = df["model"].nunique() if not df.empty and "model" in df.columns else 0
datasets   = df["dataset"].nunique() if not df.empty and "dataset" in df.columns else 0
unique_drugs = df["drug"].nunique() if not df.empty and "drug" in df.columns else 0
last_time  = df["timestamp"].iloc[-1] if not df.empty and "timestamp" in df.columns else "—"

sc1, sc2, sc3, sc4, sc5 = st.columns(5)
for col, icon, val, lbl in [
    (sc1, "🔍", total,       "Lần dự đoán"),
    (sc2, "💊", unique_drugs, "Thuốc tra cứu"),
    (sc3, "🤖", models_cnt,  "Mô hình dùng"),
    (sc4, "📂", datasets,    "Dataset"),
    (sc5, "⏰", str(last_time)[:10] if last_time != "—" else "—", "Lần cuối"),
]:
    col.markdown(f"""
    <div class="hist-stat-card">
        <div class="hist-stat-icon">{icon}</div>
        <div class="hist-stat-val">{val}</div>
        <div class="hist-stat-lbl">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# DASHBOARD SECTION (only when data exists)
# ═══════════════════════════════════════════════════════════════════════
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
    # ── Preprocess timestamps ─────────────────────────────────────────
    if "timestamp" in df.columns:
        df["_ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["_date"] = df["_ts"].dt.date

    # ══════════════════════════════════════════════════════════════════
    # ROW 1: Top drugs + Queries by day
    # ══════════════════════════════════════════════════════════════════
    dash_col1, dash_col2 = st.columns([1, 1.8], gap="medium")

    with dash_col1:
        st.markdown('<div class="dash-section">', unsafe_allow_html=True)
        st.markdown('<div class="dash-section-title">🏆 Thuốc được tra cứu nhiều nhất</div>', unsafe_allow_html=True)
        if "drug" in df.columns:
            top_drugs = df["drug"].value_counts().head(10)
            rank_classes = {1: "gold", 2: "silver", 3: "bronze"}
            for i, (drug_name, cnt) in enumerate(top_drugs.items(), 1):
                rc = rank_classes.get(i, "")
                medal = "🥇" if i==1 else "🥈" if i==2 else "🥉" if i==3 else f"#{i}"
                st.markdown(f"""
                <div class="top-drug-row">
                    <span class="top-drug-rank {rc}">{medal}</span>
                    <span class="top-drug-name">{drug_name}</span>
                    <span class="top-drug-count">{cnt} lần</span>
                </div>""", unsafe_allow_html=True)

            # Horizontal bar chart
            fig_topdrug = go.Figure(go.Bar(
                x=top_drugs.values[::-1],
                y=top_drugs.index[::-1],
                orientation='h',
                marker=dict(
                    color=top_drugs.values[::-1],
                    colorscale=[[0, '#c7d2fe'], [1, '#4338ca']],
                    showscale=False,
                    line=dict(width=0),
                ),
                text=top_drugs.values[::-1],
                textposition='outside',
                textfont=dict(size=10, color='#4338ca'),
                hovertemplate='<b>%{y}</b>: %{x} lần<extra></extra>',
            ))
            fig_topdrug.update_layout(
                height=280,
                margin=dict(t=8, b=8, l=4, r=40),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='rgba(99,102,241,0.1)',
                           zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False,
                           tickfont=dict(size=10, color='#475569')),
            )
            st.plotly_chart(fig_topdrug, use_container_width=True,
                            config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with dash_col2:
        st.markdown('<div class="dash-section">', unsafe_allow_html=True)
        st.markdown('<div class="dash-section-title">📅 Số lượng tra cứu theo ngày</div>', unsafe_allow_html=True)
        if "_date" in df.columns:
            daily = df.groupby("_date").size().reset_index(name="count")
            daily["_date"] = pd.to_datetime(daily["_date"])
            daily = daily.sort_values("_date")

            fig_daily = go.Figure()
            fig_daily.add_trace(go.Scatter(
                x=daily["_date"], y=daily["count"],
                mode='lines+markers',
                line=dict(color='#6366f1', width=2.5, shape='spline'),
                marker=dict(size=7, color='#6366f1',
                            line=dict(width=2, color='#ffffff')),
                fill='tozeroy',
                fillcolor='rgba(99,102,241,0.1)',
                hovertemplate='<b>%{x|%d/%m/%Y}</b><br>%{y} tra cứu<extra></extra>',
                name='Tra cứu/ngày',
            ))
            # Rolling average if enough data
            if len(daily) >= 5:
                rolling = daily["count"].rolling(3, min_periods=1).mean()
                fig_daily.add_trace(go.Scatter(
                    x=daily["_date"], y=rolling,
                    mode='lines',
                    line=dict(color='#c084fc', width=1.5, dash='dot'),
                    hoverinfo='none',
                    name='TB 3 ngày',
                ))
            fig_daily.update_layout(
                height=260,
                margin=dict(t=8, b=8, l=4, r=4),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, zeroline=False,
                           tickfont=dict(size=10, color='#475569'),
                           tickformat='%d/%m'),
                yaxis=dict(showgrid=True, gridcolor='rgba(99,102,241,0.1)',
                           zeroline=False, tickfont=dict(size=10, color='#475569'),
                           title=None),
                legend=dict(orientation="h", y=1.05, x=1, xanchor="right",
                            font=dict(size=10, color="#475569"),
                            bgcolor="rgba(0,0,0,0)"),
                hovermode='x unified',
            )
            st.plotly_chart(fig_daily, use_container_width=True,
                            config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # ROW 2: Model distribution + Dataset distribution + Direction
    # ══════════════════════════════════════════════════════════════════
    dash_col3, dash_col4, dash_col5 = st.columns(3, gap="medium")

    with dash_col3:
        st.markdown('<div class="dash-section">', unsafe_allow_html=True)
        st.markdown('<div class="dash-section-title">🤖 Phân bố mô hình AI</div>', unsafe_allow_html=True)
        if "model" in df.columns:
            model_counts = df["model"].value_counts()
            MODEL_COLORS = {
                "AMNTDDA_Fuzzy": "#6366f1",
                "AMNTDDA_GCN":   "#0ea5e9",
                "AMNTDDA":       "#10b981",
            }
            colors_pie = [MODEL_COLORS.get(m, "#94a3b8") for m in model_counts.index]
            fig_model = go.Figure(go.Pie(
                labels=model_counts.index,
                values=model_counts.values,
                hole=0.55,
                marker=dict(colors=colors_pie, line=dict(color='#ffffff', width=2)),
                textinfo='percent',
                textfont=dict(size=11),
                hovertemplate='<b>%{label}</b>: %{value} lần (%{percent})<extra></extra>',
            ))
            fig_model.update_layout(
                height=220,
                margin=dict(t=8, b=8, l=8, r=8),
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=True,
                legend=dict(orientation="v", x=1.02, y=0.5,
                            font=dict(size=10, color="#475569"),
                            bgcolor="rgba(0,0,0,0)"),
                annotations=[dict(
                    text=f"<b>{total}</b>", x=0.5, y=0.5,
                    font=dict(size=16, color="#4338ca"), showarrow=False,
                )],
            )
            st.plotly_chart(fig_model, use_container_width=True,
                            config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with dash_col4:
        st.markdown('<div class="dash-section">', unsafe_allow_html=True)
        st.markdown('<div class="dash-section-title">📂 Phân bố Dataset</div>', unsafe_allow_html=True)
        if "dataset" in df.columns:
            ds_counts = df["dataset"].value_counts()
            DS_COLORS = {"B-dataset": "#f97316", "C-dataset": "#8b5cf6", "F-dataset": "#10b981"}
            colors_ds = [DS_COLORS.get(d, "#94a3b8") for d in ds_counts.index]
            fig_ds = go.Figure(go.Bar(
                x=ds_counts.index,
                y=ds_counts.values,
                marker=dict(color=colors_ds, line=dict(width=0)),
                text=ds_counts.values,
                textposition='outside',
                textfont=dict(size=12, color='#1e293b'),
                hovertemplate='<b>%{x}</b>: %{y} lần<extra></extra>',
            ))
            fig_ds.update_layout(
                height=220,
                margin=dict(t=8, b=8, l=4, r=4),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, zeroline=False,
                           tickfont=dict(size=11, color='#475569')),
                yaxis=dict(showgrid=True, gridcolor='rgba(99,102,241,0.1)',
                           zeroline=False, showticklabels=False),
                showlegend=False,
            )
            st.plotly_chart(fig_ds, use_container_width=True,
                            config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with dash_col5:
        st.markdown('<div class="dash-section">', unsafe_allow_html=True)
        st.markdown('<div class="dash-section-title">↔ Hướng dự đoán</div>', unsafe_allow_html=True)
        if "direction" in df.columns:
            dir_counts = df["direction"].value_counts()
            DIR_ICONS = {
                "Thuốc → Bệnh": "💊→🦠",
                "Thuốc → Protein": "💊→🔬",
                "Bệnh → Thuốc": "🦠→💊",
            }
            for d_name, d_cnt in dir_counts.items():
                icon = DIR_ICONS.get(d_name, "↔")
                pct = int(d_cnt / max(total, 1) * 100)
                st.markdown(f"""
                <div style="margin-bottom:10px;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                        <span style="font-size:0.82rem;color:#1e293b;font-weight:600;">{icon} {d_name}</span>
                        <span style="font-size:0.82rem;color:#6366f1;font-weight:700;">{d_cnt}</span>
                    </div>
                    <div style="background:#e2e8f0;border-radius:4px;height:6px;overflow:hidden;">
                        <div style="width:{pct}%;background:linear-gradient(90deg,#6366f1,#8b5cf6);height:100%;border-radius:4px;"></div>
                    </div>
                </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════
    # HISTORY TABLE with filters
    # ══════════════════════════════════════════════════════════════════
    st.markdown("""
    <div style="font-size:0.95rem;font-weight:700;color:#1e293b;margin-bottom:12px;">
        📋 Chi tiết lịch sử tra cứu
    </div>""", unsafe_allow_html=True)

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

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        if "model" in df.columns:
            model_filter = st.multiselect("Lọc mô hình", df["model"].unique().tolist(),
                                           default=df["model"].unique().tolist(),
                                           label_visibility="collapsed",
                                           placeholder="Lọc theo mô hình…")
        else:
            model_filter = []
    with fc2:
        if "dataset" in df.columns:
            ds_filter = st.multiselect("Lọc dataset", df["dataset"].unique().tolist(),
                                        default=df["dataset"].unique().tolist(),
                                        label_visibility="collapsed",
                                        placeholder="Lọc theo dataset…")
        else:
            ds_filter = []
    with fc3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            "⬇  Tải toàn bộ CSV",
            data=df[display_cols].rename(columns=rename_map).to_csv(index=False),
            file_name="prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

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

