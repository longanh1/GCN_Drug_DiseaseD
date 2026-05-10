"""Chart utilities using Plotly."""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Dict


def donut_chart(labels: List[str], values: List[int], title: str = "") -> go.Figure:
    colors = ["#3b82f6", "#f97316", "#ef4444", "#22c55e", "#a855f7"]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55,
        marker_colors=colors[:len(labels)],
        textinfo="percent",
        hovertemplate="%{label}: %{value:,}<extra></extra>",
    ))
    fig.update_layout(
        title=title, showlegend=True,
        margin=dict(t=40, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        legend=dict(font=dict(color="#e2e8f0")),
    )
    return fig


def bar_chart_comparison(drugs: List[str], gcn_scores: List[float],
                          fuzzy_scores: List[float]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="GCN", x=drugs, y=gcn_scores,
                         marker_color="#3b82f6"))
    fig.add_trace(go.Bar(name="GCN+Fuzzy", x=drugs, y=fuzzy_scores,
                         marker_color="#8b5cf6"))
    fig.update_layout(
        barmode="group",
        xaxis_title="Thuốc", yaxis_title="Điểm trung bình",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        legend=dict(font=dict(color="#e2e8f0")),
        margin=dict(t=20, b=40, l=40, r=20),
    )
    return fig


def heatmap(drugs: List[str], diseases: List[str],
            gcn_matrix: np.ndarray, fuzzy_matrix: np.ndarray) -> go.Figure:
    """Side-by-side heatmaps for GCN and Fuzzy scores."""
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["GCN (AMNTDDA)", "GCN + Fuzzy"])

    fig.add_trace(go.Heatmap(
        z=gcn_matrix, x=diseases, y=drugs,
        colorscale="Blues", showscale=False,
        hovertemplate="Thuốc: %{y}<br>Bệnh: %{x}<br>Score: %{z:.4f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        z=fuzzy_matrix, x=diseases, y=drugs,
        colorscale="Greens", showscale=True,
        hovertemplate="Thuốc: %{y}<br>Bệnh: %{x}<br>Score: %{z:.4f}<extra></extra>",
    ), row=1, col=2)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        margin=dict(t=40, b=40, l=40, r=40),
    )
    return fig


def radar_chart(categories: List[str],
                gcn_values: List[float],
                fuzzy_values: List[float]) -> go.Figure:
    cats = categories + [categories[0]]
    gcn  = gcn_values + [gcn_values[0]]
    fuz  = fuzzy_values + [fuzzy_values[0]]
    fig  = go.Figure()
    fig.add_trace(go.Scatterpolar(r=gcn, theta=cats, fill='toself',
                                   name='GCN', line_color='#3b82f6', opacity=0.7))
    fig.add_trace(go.Scatterpolar(r=fuz, theta=cats, fill='toself',
                                   name='GCN+Fuzzy', line_color='#8b5cf6', opacity=0.7))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        margin=dict(t=20, b=20, l=20, r=20),
    )
    return fig


def membership_chart(universe: np.ndarray = None) -> go.Figure:
    """Triangular membership function chart (Low / Mid / High)."""
    if universe is None:
        universe = np.linspace(0, 1, 101)
    low_mf  = np.maximum(0, 1 - 2 * universe)
    mid_mf  = np.maximum(0, 1 - 2 * np.abs(universe - 0.5))
    high_mf = np.maximum(0, 2 * universe - 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=universe, y=low_mf,  fill='tozeroy',
                              name='Low',  line_color='#ef4444', fillcolor='rgba(239,68,68,0.2)'))
    fig.add_trace(go.Scatter(x=universe, y=mid_mf,  fill='tozeroy',
                              name='Mid',  line_color='#eab308', fillcolor='rgba(234,179,8,0.2)'))
    fig.add_trace(go.Scatter(x=universe, y=high_mf, fill='tozeroy',
                              name='High', line_color='#22c55e', fillcolor='rgba(34,197,94,0.2)'))
    fig.update_layout(
        xaxis_title="Giá trị đầu vào", yaxis_title="Độ thành viên μ",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        legend=dict(font=dict(color="#e2e8f0")),
        margin=dict(t=10, b=40, l=40, r=10),
        yaxis=dict(range=[0, 1.05]),
    )
    return fig


def score_bar(score: float, label: str = "") -> go.Figure:
    fig = go.Figure(go.Bar(
        x=[score], y=[label or ""],
        orientation='h',
        marker=dict(color="#22c55e" if score > 0.5 else "#f97316",
                    line=dict(width=0)),
        text=[f"{score:.4f}"],
        textposition="outside",
        textfont=dict(color="#1e293b", size=13, family="Inter, sans-serif"),
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 1.25]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1e293b"),
        margin=dict(t=5, b=5, l=5, r=45),
        height=60,
    )
    return fig
