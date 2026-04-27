"""
3_model_stages.py — Các giai đoạn chạy của mô hình gốc AMNTDDA
Hiển thị 4 giai đoạn của mô hình gốc (KHÔNG có GCN, KHÔNG có Fuzzy Logic)
với biểu đồ 2D và 3D tương tác.

Kết quả được lưu trong AMDGT_main/Run_Base/ sau khi chạy run_base_stages.py
"""

import os
import subprocess
import sys
import json
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import requests

# ── API helper ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.api_client import list_stages, get_stage_result, get_training_results_ai, _AI

AI_ENGINE_URL = _AI
AMDGT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'AMDGT_main')
)

STAGE_META = {
    'stage1_input_layer': {
        'title':    'Giai đoạn 1: Xây dựng mạng lưới (Input Layer)',
        'subtitle': 'Similarity Network + Mạng lưới sinh hóa dị thể (Biochemical Heterogeneous Network)',
        'icon':     '🕸️',
        'color':    '#3B82F6',
        'desc': (
            'Giai đoạn này xây dựng 2 loại mạng lưới: '
            '(1) Ma trận tương đồng thuốc–thuốc và bệnh–bệnh dựa trên GIP và fingerprint, '
            '(2) Mạng lưới dị thể bao gồm thuốc, bệnh và protein với liên kết đã biết.'
        ),
    },
    'stage2_feature_extraction': {
        'title':    'Giai đoạn 2: Trích xuất đặc trưng (Feature Extraction)',
        'subtitle': 'Graph Transformer + HGT (Heterogeneous Graph Transformer)',
        'icon':     '🧠',
        'color':    '#8B5CF6',
        'desc': (
            'Graph Transformer trích xuất biểu diễn từ đồ thị tương đồng. '
            'HGT (Heterogeneous Graph Transformer) học biểu diễn từ mạng dị thể '
            'gồm thuốc, bệnh và protein.'
        ),
    },
    'stage3_modality_interaction': {
        'title':    'Giai đoạn 3: Tương tác đa phương thức (Modality Interaction)',
        'subtitle': 'Cross-Modal Transformer Interaction Module',
        'icon':     '🔄',
        'color':    '#10B981',
        'desc': (
            'TransformerEncoder kết hợp thông tin từ đồ thị tương đồng và đồ thị dị thể. '
            'Cross-attention Transformer cho phép thuốc và bệnh "trao đổi" thông tin đa phương thức.'
        ),
    },
    'stage4_prediction': {
        'title':    'Giai đoạn 4: Dự đoán & Kiểm chứng (Prediction Module)',
        'subtitle': 'Prediction Module + Case Study & Molecular Docking',
        'icon':     '🎯',
        'color':    '#F59E0B',
        'desc': (
            'MLP phân loại cặp thuốc–bệnh. '
            'Đánh giá với AUC, AUPR, Accuracy, Precision, Recall, F1, MCC '
            'qua k-fold cross-validation. Kết quả giai đoạn này là baseline '
            'để so sánh với các mô hình nâng cấp.'
        ),
    },
}

METRICS_LIST = ['AUC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC']


# ══════════════════════════════════════════════════════════════════════
# Page setup
# ══════════════════════════════════════════════════════════════════════
st.title('🔬 Các giai đoạn chạy của mô hình gốc')
st.caption(
    'Mô hình gốc AMNTDDA — **Không có GCN, Không có Fuzzy Logic.** '
    'Kết quả các giai đoạn được lưu để so sánh với mô hình nâng cấp.'
)

# ── Sidebar controls ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown('### ⚙️ Cài đặt')
    dataset = st.selectbox('Dataset', ['B-dataset', 'C-dataset', 'F-dataset'], key='stages_dataset')
    st.markdown('---')
    st.markdown('### 🚀 Chạy các giai đoạn')
    stages_to_run = st.multiselect(
        'Chọn giai đoạn cần chạy',
        options=['1', '2', '3', '4'],
        default=['1', '2', '3', '4'],
        format_func=lambda x: f'Giai đoạn {x}',
    )
    epochs_s4 = st.slider('Số epoch (Giai đoạn 4)', 3, 50, 10)
    run_btn   = st.button('▶  Chạy ngay', type='primary', use_container_width=True)
    st.markdown('---')
    st.markdown(
        '**Lưu ý:** Kết quả được lưu vào  \n'
        '`AMDGT_main/Run_Base/`  \n'
        'Không sửa đổi để dùng làm baseline.'
    )

# ── Run stages if requested ──────────────────────────────────────────
if run_btn:
    if not stages_to_run:
        st.warning('Vui lòng chọn ít nhất một giai đoạn.')
    else:
        run_script = os.path.join(AMDGT_DIR, 'run_base_stages.py')
        python_exe = sys.executable
        cmd = [
            python_exe, run_script,
            '--dataset', dataset,
            '--epochs', str(epochs_s4),
            '--stages', *stages_to_run,
        ]
        with st.status(f'Đang chạy giai đoạn {stages_to_run} trên {dataset}...', expanded=True) as status:
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True,
                    cwd=AMDGT_DIR, timeout=600
                )
                if result.returncode == 0:
                    status.update(label='✅ Hoàn thành!', state='complete')
                    st.success('Đã chạy xong. Làm mới trang để xem kết quả.')
                    with st.expander('Xem log đầu ra'):
                        st.code(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
                else:
                    status.update(label='❌ Lỗi khi chạy', state='error')
                    st.error('Có lỗi xảy ra:')
                    st.code(result.stderr[-2000:])
            except subprocess.TimeoutExpired:
                status.update(label='⏱️ Timeout', state='error')
                st.error('Quá thời gian chờ (10 phút).')
            except Exception as ex:
                status.update(label='❌ Exception', state='error')
                st.error(str(ex))

st.markdown('---')

# ── Load stage status ────────────────────────────────────────────────
stage_list_data = list_stages(dataset)
stage_status = {s['folder']: s['has_result'] for s in stage_list_data.get('stages', [])}

# ── Stage pipeline overview ──────────────────────────────────────────
st.subheader('📊 Tổng quan pipeline mô hình gốc')
cols = st.columns(4)
stage_folders = list(STAGE_META.keys())

for col, folder in zip(cols, stage_folders):
    meta    = STAGE_META[folder]
    has_res = stage_status.get(folder, False)
    with col:
        st.markdown(
            f"""
            <div style="
                background: {'#f0fdf4' if has_res else '#fef9c3'};
                border: 2px solid {meta['color']};
                border-radius: 12px;
                padding: 14px;
                text-align: center;
            ">
                <div style="font-size:2rem">{meta['icon']}</div>
                <div style="font-weight:600; color:{meta['color']}; font-size:.85rem; margin-top:6px">
                    {meta['title'].split(':')[0]}
                </div>
                <div style="font-size:.75rem; color:#6B7280; margin-top:4px">
                    {'✅ Có kết quả' if has_res else '⏳ Chưa chạy'}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown('<br>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# 4 stage tabs
# ══════════════════════════════════════════════════════════════════════
tab_labels = [f"{STAGE_META[f]['icon']} {STAGE_META[f]['title'].split(':')[0]}"
              for f in stage_folders]
tabs = st.tabs(tab_labels)


def _color_badge(text: str, color: str) -> str:
    return (f'<span style="background:{color};color:#fff;padding:2px 8px;'
            f'border-radius:4px;font-size:.8rem">{text}</span>')


# ── Shared: metric bar chart ─────────────────────────────────────────
def _metric_bar(metrics: dict, title: str):
    keys = [k for k in METRICS_LIST if k in metrics]
    vals = [round(float(metrics[k]), 4) for k in keys]
    fig = go.Figure(go.Bar(
        x=keys, y=vals,
        text=[f'{v:.4f}' for v in vals],
        textposition='auto',
        marker=dict(
            color=['#3B82F6','#8B5CF6','#10B981','#F59E0B','#EF4444','#06B6D4','#84CC16'],
            line=dict(width=0),
        ),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        yaxis_range=[0, 1.05], height=370,
        template='plotly_white',
        plot_bgcolor='rgba(248,250,252,1)',
        bargap=0.35,
    )
    st.plotly_chart(fig, use_container_width=True)


PALETTES = {
    'drug':    'Turbo',
    'disease': 'Plasma',
    'default': 'Viridis',
}

_METHOD_LABEL = {'tsne': 't-SNE', 'pca': 'PCA'}


def _method_str(raw: str) -> str:
    return _METHOD_LABEL.get(raw.lower(), raw.upper())


def _scatter2d(pts: np.ndarray, title: str, method: str = 't-SNE',
               palette: str = 'Turbo', point_size: int = 7):
    """Styled 2D embedding scatter (t-SNE or PCA)."""
    if pts is None or len(pts) == 0:
        return None
    # Normalise method label
    _method_labels = {'tsne': 't-SNE', 'pca': 'PCA'}
    method = _method_labels.get(method.lower(), method)
    n = len(pts)
    idx = list(range(n))
    fig = go.Figure(go.Scatter(
        x=pts[:, 0], y=pts[:, 1],
        mode='markers',
        marker=dict(
            size=point_size,
            color=idx,
            colorscale=palette,
            showscale=True,
            colorbar=dict(thickness=10, title='', outlinewidth=0),
            opacity=0.82,
            line=dict(width=0.4, color='rgba(255,255,255,0.5)'),
        ),
        text=[f'Index {i}' for i in idx],
        hovertemplate='<b>%{text}</b><br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text=f'{title}  [{method}]',
                   font=dict(size=13, color='#1e293b')),
        height=360,
        template='plotly_white',
        plot_bgcolor='rgba(248,250,252,1)',
        paper_bgcolor='rgba(255,255,255,1)',
        xaxis=dict(title=f'{method}-1', gridcolor='rgba(0,0,0,0.06)', zeroline=False,
                   showline=True, linecolor='#e2e8f0'),
        yaxis=dict(title=f'{method}-2', gridcolor='rgba(0,0,0,0.06)', zeroline=False,
                   showline=True, linecolor='#e2e8f0'),
        margin=dict(l=20, r=40, t=50, b=30),
    )
    return fig


def _scatter3d(pts: np.ndarray, title: str, palette: str = 'Turbo'):
    """Styled 3D embedding scatter."""
    if pts is None or len(pts) == 0:
        return None
    n = len(pts)
    fig = go.Figure(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode='markers',
        marker=dict(
            size=4.5,
            color=list(range(n)),
            colorscale=palette,
            showscale=True,
            opacity=0.88,
            line=dict(width=0),
        ),
        text=[f'Index {i}' for i in range(n)],
        hovertemplate='<b>%{text}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color='#1e293b')),
        height=460,
        scene=dict(
            xaxis=dict(title='Dim-1', backgroundcolor='rgba(238,242,255,0.7)',
                       gridcolor='rgba(99,102,241,0.1)'),
            yaxis=dict(title='Dim-2', backgroundcolor='rgba(238,242,255,0.7)',
                       gridcolor='rgba(99,102,241,0.1)'),
            zaxis=dict(title='Dim-3', backgroundcolor='rgba(238,242,255,0.7)',
                       gridcolor='rgba(99,102,241,0.1)'),
            bgcolor='rgba(248,250,252,1)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor='rgba(255,255,255,1)',
    )
    return fig


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — Input Layer
# ══════════════════════════════════════════════════════════════════════
with tabs[0]:
    meta   = STAGE_META['stage1_input_layer']
    st.markdown(f"### {meta['icon']} {meta['title']}")
    st.info(meta['desc'])

    data = get_stage_result('stage1_input_layer', dataset)
    if 'error' in data:
        st.warning(f'⏳ Chưa có kết quả cho {dataset}. Vui lòng chạy giai đoạn 1 từ sidebar.')
    else:
        # Network stats
        ns = data.get('network_stats', {})
        fd = data.get('feature_dims', {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Số thuốc',   ns.get('num_drugs', '-'))
        c2.metric('Số bệnh',    ns.get('num_diseases', '-'))
        c3.metric('Số protein', ns.get('num_proteins', '-'))
        c4.metric('Liên kết thuốc–bệnh đã biết', ns.get('drug_disease_links', '-'))

        col_l, col_r = st.columns(2)

        with col_l:
            # Similarity matrix heatmap 2D (DrugGIP 5×5 sample)
            sim_mats = data.get('similarity_matrices', [])
            for sm in sim_mats:
                s5 = np.array(sm.get('sample_5x5', []))
                if s5.size == 0:
                    continue
                fig = go.Figure(go.Heatmap(
                    z=s5.tolist(), colorscale='Blues',
                    text=[[f'{v:.3f}' for v in row] for row in s5.tolist()],
                    texttemplate='%{text}', hoverongaps=False,
                    colorbar=dict(thickness=12, outlinewidth=0),
                ))
                fig.update_layout(
                    title=dict(text=f'Ma trận tương đồng — {sm["name"]} (5×5 mẫu)',
                               font=dict(size=13)),
                    height=310,
                    template='plotly_white',
                    plot_bgcolor='rgba(248,250,252,1)',
                    margin=dict(l=20, r=20, t=45, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    f'**{sm["name"]}** | Kích thước: {sm["shape"]} | '
                    f'Mean: {sm["mean"]:.4f} | Sparsity: {sm["sparsity"]:.2%}'
                )

        with col_r:
            # 3D surface của DrugGIP sample
            drg_mat = next((m for m in sim_mats if 'GIP' in m.get('name', '') and 'Drug' in m.get('name', '')), None)
            if drg_mat:
                z_data = np.array(drg_mat.get('sample_5x5', []))
                if z_data.size > 0:
                    fig3 = go.Figure(go.Surface(
                        z=z_data.tolist(), colorscale='Portland',
                        contours=dict(
                            x=dict(show=True, highlightcolor='#e2e8f0', project_x=True),
                            y=dict(show=True, highlightcolor='#e2e8f0', project_y=True),
                        ),
                        opacity=0.92,
                        colorbar=dict(thickness=12, outlinewidth=0),
                    ))
                    fig3.update_layout(
                        title=dict(text='3D Surface — DrugGIP (5×5 mẫu)',
                                   font=dict(size=13)),
                        height=370,
                        scene=dict(
                            xaxis_title='Drug j',
                            yaxis_title='Drug i',
                            zaxis_title='Similarity',
                            xaxis=dict(backgroundcolor='rgba(238,242,255,0.7)'),
                            yaxis=dict(backgroundcolor='rgba(238,242,255,0.7)'),
                            zaxis=dict(backgroundcolor='rgba(238,242,255,0.7)'),
                            bgcolor='rgba(248,250,252,1)',
                            camera=dict(eye=dict(x=1.5, y=-1.5, z=0.9)),
                        ),
                        paper_bgcolor='rgba(255,255,255,1)',
                        margin=dict(l=0, r=0, t=50, b=0),
                    )
                    st.plotly_chart(fig3, use_container_width=True)

            # Heterogeneous network edge stats
            st.markdown('#### Mạng lưới dị thể')
            edge_data = {
                'Loại liên kết': ['Thuốc–Bệnh', 'Thuốc–Protein', 'Bệnh–Protein'],
                'Số cạnh': [
                    ns.get('drug_disease_links', 0),
                    ns.get('drug_protein_links', 0),
                    ns.get('disease_protein_links', 0),
                ],
            }
            fig_edge = go.Figure(go.Bar(
                x=edge_data['Loại liên kết'],
                y=edge_data['Số cạnh'],
                marker_color=['#3B82F6', '#10B981', '#F59E0B'],
                text=edge_data['Số cạnh'], textposition='auto',
            ))
            fig_edge.update_layout(
                title='Số cạnh mạng dị thể', height=280,
                template='plotly_white', margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_edge, use_container_width=True)
            st.caption(f'Sparsity mạng dị thể: {ns.get("hetero_sparsity", 0):.2%}')

        # t-SNE 2D scatter — Drug & Disease input features
        pca2_data = data.get('pca2d', {})
        col_s1, col_s2 = st.columns(2)
        for col_s, key, label, pal in [
            (col_s1, 'drug',    'Đặc trưng thuốc đầu vào (mol2vec)',    'Turbo'),
            (col_s2, 'disease', 'Đặc trưng bệnh đầu vào (DiseaseFeature)', 'Plasma'),
        ]:
            pts = pca2_data.get(key, {}).get('points')
            if pts:
                method = _method_str(pca2_data.get(key, {}).get('method', 'tsne'))
                fig_s = _scatter2d(np.array(pts), label, method=method, palette=pal)
                if fig_s:
                    with col_s:
                        st.plotly_chart(fig_s, use_container_width=True)

        # PCA 3D scatter
        pca3_data = data.get('pca3d', {})
        for key, label, pal in [
            ('drug',    'Đặc trưng thuốc — PCA 3D', 'Turbo'),
            ('disease', 'Đặc trưng bệnh — PCA 3D',  'Plasma'),
        ]:
            pts3 = pca3_data.get(key, {}).get('points')
            if pts3:
                fig_3d = _scatter3d(np.array(pts3), label, palette=pal)
                if fig_3d:
                    st.plotly_chart(fig_3d, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — Feature Extraction
# ══════════════════════════════════════════════════════════════════════
with tabs[1]:
    meta = STAGE_META['stage2_feature_extraction']
    st.markdown(f"### {meta['icon']} {meta['title']}")
    st.info(meta['desc'])

    data2 = get_stage_result('stage2_feature_extraction', dataset)
    if 'error' in data2:
        st.warning(f'⏳ Chưa có kết quả cho {dataset}. Vui lòng chạy giai đoạn 2 từ sidebar.')
    else:
        odims = data2.get('output_dims', {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('GT Drug dim',     odims.get('gt_drug_dim', '-'))
        c2.metric('GT Disease dim',  odims.get('gt_disease_dim', '-'))
        c3.metric('HGT Drug dim',    odims.get('hgt_drug_dim', '-'))
        c4.metric('HGT Disease dim', odims.get('hgt_disease_dim', '-'))

        col_l, col_r = st.columns(2)

        with col_l:
            # Attention weights heatmap 2D
            attn = data2.get('attention_approx_50x50')
            if attn:
                attn_arr = np.array(attn)
                n = min(20, attn_arr.shape[0])
                fig_attn = go.Figure(go.Heatmap(
                    z=attn_arr[:n, :n].tolist(),
                    colorscale='RdBu', zmid=0,
                    colorbar=dict(thickness=12, outlinewidth=0),
                    hoverongaps=False,
                ))
                fig_attn.update_layout(
                    title=dict(text=f'Attention weights (GT Drug, {n}×{n} mẫu)',
                               font=dict(size=13)),
                    height=390,
                    template='plotly_white',
                    plot_bgcolor='rgba(248,250,252,1)',
                    margin=dict(l=30, r=30, t=45, b=20),
                )
                st.plotly_chart(fig_attn, use_container_width=True)

            # GT Drug stats
            gt_stats = data2.get('graph_transformer_stats', {})
            if gt_stats:
                st.markdown('#### 📈 Thống kê đặc trưng Graph Transformer')
                rows = []
                for k, v in gt_stats.items():
                    rows.append({'Module': k, 'Mean': round(v['mean'], 4),
                                 'Std': round(v['std'], 4),
                                 'Norm': round(v['norm'], 2),
                                 'Shape': str(v['shape'])})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with col_r:
            # t-SNE 2D: GT Drug, HGT Drug, GT Disease
            pca2d = data2.get('pca2d', {})
            for key, label, pal in [
                ('gt_drug',    'GT Drug Embeddings',    'Turbo'),
                ('hgt_drug',   'HGT Drug Embeddings',   'Bluered'),
                ('gt_disease', 'GT Disease Embeddings', 'Plasma'),
            ]:
                pts = pca2d.get(key, {}).get('points')
                if not pts:
                    continue
                method = _method_str(pca2d.get(key, {}).get('method', 'tsne'))
                fig_s = _scatter2d(np.array(pts), label, method=method, palette=pal, point_size=6)
                if fig_s:
                    st.plotly_chart(fig_s, use_container_width=True)

        # 3D embeddings — full-width
        pca3d = data2.get('pca3d', {})
        col_3a, col_3b = st.columns(2)
        for col_3, key, label, pal in [
            (col_3a, 'gt_drug',  'GT Drug — PCA 3D',  'Turbo'),
            (col_3b, 'hgt_drug', 'HGT Drug — PCA 3D', 'Electric'),
        ]:
            pts3 = pca3d.get(key, {}).get('points')
            if pts3:
                fig3d = _scatter3d(np.array(pts3), label, palette=pal)
                if fig3d:
                    with col_3:
                        st.plotly_chart(fig3d, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — Modality Interaction
# ══════════════════════════════════════════════════════════════════════
with tabs[2]:
    meta = STAGE_META['stage3_modality_interaction']
    st.markdown(f"### {meta['icon']} {meta['title']}")
    st.info(meta['desc'])

    data3 = get_stage_result('stage3_modality_interaction', dataset)
    if 'error' in data3:
        st.warning(f'⏳ Chưa có kết quả cho {dataset}. Vui lòng chạy giai đoạn 3 từ sidebar.')
    else:
        odims3 = data3.get('output_dims', {})
        col1, col2 = st.columns(2)
        col1.metric('Drug Final dim',    odims3.get('drug_final_dim', odims3.get('drug_trans_dim', '-')))
        col2.metric('Disease Final dim', odims3.get('disease_final_dim', odims3.get('disease_trans_dim', '-')))

        col_l, col_r = st.columns(2)

        with col_l:
            # Transformer stats
            tr_stats = data3.get('transformer_stats', {})
            if tr_stats:
                rows = [{'Module': k, 'Mean': round(v['mean'], 4),
                         'Std': round(v['std'], 4), 'Norm': round(v['norm'], 2)}
                        for k, v in tr_stats.items()]
                st.markdown('#### 📊 Thống kê Transformer Encoder')
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Pair interaction samples
            pairs = data3.get('pair_interaction_samples', [])
            if pairs:
                df_pairs = pd.DataFrame(pairs[:20])
                # use score if available, else cross norms
                xcol = 'cross_dr_norm' if 'cross_dr_norm' in df_pairs.columns else 'combined_drug_norm'
                ycol = 'cross_di_norm' if 'cross_di_norm' in df_pairs.columns else 'combined_disease_norm'
                if xcol in df_pairs.columns and ycol in df_pairs.columns:
                    fig_pairs = go.Figure(go.Scatter(
                        x=df_pairs[xcol], y=df_pairs[ycol],
                        mode='markers+text',
                        text=[f"D{r['drug_idx']}-I{r['disease_idx']}" for _, r in df_pairs.iterrows()],
                        textposition='top center',
                        textfont=dict(size=8, color='#64748b'),
                        marker=dict(size=10, color=list(range(len(df_pairs))), colorscale='Turbo',
                                    opacity=0.85, line=dict(width=0.5, color='white')),
                        hovertemplate=f'<b>Drug %{{text}}</b><br>{xcol}: %{{x:.3f}}<br>{ycol}: %{{y:.3f}}<extra></extra>',
                    ))
                    fig_pairs.update_layout(
                        title='Cross-modal pair interaction scores',
                        height=320, template='plotly_white',
                        plot_bgcolor='rgba(248,250,252,1)',
                        xaxis=dict(title=xcol, gridcolor='rgba(0,0,0,0.06)'),
                        yaxis=dict(title=ycol, gridcolor='rgba(0,0,0,0.06)'),
                    )
                    st.plotly_chart(fig_pairs, use_container_width=True)

        with col_r:
            # t-SNE 2D: final drug & disease embeddings
            pca2d3 = data3.get('pca2d', {})
            for key, label, pal in [
                ('drug',    'Drug — Biểu diễn cuối cùng',    'Turbo'),
                ('disease', 'Disease — Biểu diễn cuối cùng', 'Plasma'),
            ]:
                pts = pca2d3.get(key, {}).get('points')
                if not pts:
                    continue
                method = _method_str(pca2d3.get(key, {}).get('method', 'tsne'))
                fig_s = _scatter2d(np.array(pts), label, method=method, palette=pal, point_size=6)
                if fig_s:
                    st.plotly_chart(fig_s, use_container_width=True)

        # 3D final embeddings
        pca3d3 = data3.get('pca3d', {})
        col_3a, col_3b = st.columns(2)
        for col_3, key, label, pal in [
            (col_3a, 'drug',    'Drug Final — PCA 3D',    'Turbo'),
            (col_3b, 'disease', 'Disease Final — PCA 3D', 'Plasma'),
        ]:
            pts3 = pca3d3.get(key, {}).get('points')
            if pts3:
                fig3d = _scatter3d(np.array(pts3), label, palette=pal)
                if fig3d:
                    with col_3:
                        st.plotly_chart(fig3d, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 4 — Prediction & Evaluation
# ══════════════════════════════════════════════════════════════════════
with tabs[3]:
    meta = STAGE_META['stage4_prediction']
    st.markdown(f"### {meta['icon']} {meta['title']}")
    st.info(meta['desc'])

    # ── Section A: Full 10-fold results (from pre-computed CSV) ──────
    st.markdown('#### 📊 Kết quả đầy đủ — 10-fold cross-validation (AMNTDDA gốc)')
    full_res = get_training_results_ai(dataset=dataset, model='AMNTDDA')
    folds_data = full_res.get('folds', [])
    summary_data = full_res.get('summary', {})

    if folds_data:
        df_folds = pd.DataFrame(folds_data)
        num_cols = [c for c in METRICS_LIST if c in df_folds.columns]

        # Summary metrics from fold Mean row
        mean_row = summary_data.get('mean', {}) if summary_data else {}
        std_row  = summary_data.get('std',  {}) if summary_data else {}
        # Fallback: compute from fold data
        if not mean_row and num_cols:
            mean_row = df_folds[num_cols].mean().to_dict()
            std_row  = df_folds[num_cols].std().to_dict()

        m_cols = st.columns(len(num_cols))
        for col_m, k in zip(m_cols, num_cols):
            col_m.metric(
                k,
                f"{mean_row.get(k, 0):.4f}",
                delta=f"±{std_row.get(k, 0):.4f}",
                delta_color='off',
            )

        st.markdown('<br>', unsafe_allow_html=True)
        col_tbl, col_bar = st.columns([1, 1])

        with col_tbl:
            st.markdown('##### Kết quả từng fold')
            display_df = df_folds[['fold'] + num_cols].copy() if 'fold' in df_folds.columns else df_folds[num_cols].copy()
            display_df = display_df.round(4)
            st.dataframe(display_df, use_container_width=True, hide_index=True,
                         column_config={k: st.column_config.ProgressColumn(
                             k, min_value=0.0, max_value=1.0, format='%.4f')
                             for k in num_cols})

        with col_bar:
            st.markdown('##### Mean ± Std theo metric')
            if mean_row and num_cols:
                means = [round(float(mean_row.get(k, 0)), 4) for k in num_cols]
                stds  = [round(float(std_row.get(k, 0)), 4)  for k in num_cols]
                fig_full = go.Figure(go.Bar(
                    x=num_cols, y=means,
                    error_y=dict(type='data', array=stds, visible=True,
                                 color='rgba(30,41,59,0.5)', thickness=2, width=6),
                    text=[f'{v:.4f}' for v in means],
                    textposition='auto',
                    marker=dict(
                        color=means, colorscale='Turbo', showscale=False,
                        line=dict(width=0),
                    ),
                ))
                fig_full.update_layout(
                    title=dict(text=f'Mean ± Std — 10 fold ({dataset})', font=dict(size=13)),
                    yaxis=dict(range=[0, 1.05], title='Score',
                               gridcolor='rgba(0,0,0,0.06)'),
                    height=350, template='plotly_white',
                    plot_bgcolor='rgba(248,250,252,1)', bargap=0.35,
                )
                st.plotly_chart(fig_full, use_container_width=True)

        # Fold-by-fold line chart
        if num_cols and 'fold' in df_folds.columns:
            st.markdown('##### Biến động theo fold')
            fig_line = go.Figure()
            colors = ['#3B82F6', '#8B5CF6', '#10B981', '#F59E0B', '#EF4444', '#06B6D4', '#84CC16']
            for i, k in enumerate(num_cols):
                if k not in df_folds.columns:
                    continue
                fig_line.add_trace(go.Scatter(
                    x=df_folds['fold'].astype(str), y=df_folds[k].round(4),
                    mode='lines+markers', name=k,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=7),
                ))
            fig_line.update_layout(
                title=dict(text='Kết quả theo từng fold', font=dict(size=13)),
                xaxis_title='Fold', yaxis=dict(title='Score', range=[0, 1.05],
                                               gridcolor='rgba(0,0,0,0.06)'),
                height=340, template='plotly_white',
                plot_bgcolor='rgba(248,250,252,1)',
                legend=dict(orientation='h', y=1.02),
            )
            st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.warning(f'⏳ Chưa có kết quả 10-fold cho {dataset}/AMNTDDA trong thư mục results.')

    st.markdown('---')

    # ── Section B: Stage-4 demo run (1 fold, limited epochs) ────────
    with st.expander('🔬 Chi tiết chạy minh họa Stage 4 (1 fold, epoch giới hạn)', expanded=False):
        data4 = get_stage_result('stage4_prediction', dataset)
        if 'error' in data4:
            st.warning(f'⏳ Chưa có kết quả stage 4 cho {dataset}. Chạy từ sidebar để xem.')
        else:
            best_m = data4.get('best_metrics', {})
            lc = data4.get('learning_curve', {})
            epochs_run = data4.get('epochs_run', '?')
            st.caption(f'Chạy 1 fold × {epochs_run} epoch để minh họa quá trình huấn luyện.')

            col_l, col_r = st.columns(2)

            with col_l:
                if lc and lc.get('epochs'):
                    fig_lc = go.Figure()
                    fig_lc.add_trace(go.Scatter(
                        x=lc['epochs'], y=lc.get('auc', []),
                        name='AUC', fill='tozeroy',
                        line=dict(color='#3B82F6', width=2),
                        fillcolor='rgba(59,130,246,0.08)',
                    ))
                    fig_lc.add_trace(go.Scatter(
                        x=lc['epochs'], y=lc.get('aupr', []),
                        name='AUPR', line=dict(color='#8B5CF6', width=2),
                    ))
                    fig_lc.update_layout(
                        title=dict(text='Learning Curve — AUC & AUPR', font=dict(size=13)),
                        xaxis_title='Epoch',
                        yaxis=dict(title='Score', gridcolor='rgba(0,0,0,0.06)'),
                        height=300, template='plotly_white',
                        plot_bgcolor='rgba(248,250,252,1)',
                        legend=dict(orientation='h', y=1.02),
                    )
                    st.plotly_chart(fig_lc, use_container_width=True)

                if lc and lc.get('loss'):
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(
                        x=lc['epochs'], y=lc['loss'],
                        name='Loss', fill='tozeroy',
                        line=dict(color='#EF4444', width=2.5),
                        fillcolor='rgba(239,68,68,0.1)',
                    ))
                    if lc.get('auc'):
                        fig_loss.add_trace(go.Scatter(
                            x=lc['epochs'], y=lc['auc'],
                            name='AUC', yaxis='y2',
                            line=dict(color='#3B82F6', width=2, dash='dot'),
                        ))
                    fig_loss.update_layout(
                        title=dict(text='Loss Curve', font=dict(size=13)),
                        xaxis_title='Epoch',
                        yaxis=dict(title='Loss', gridcolor='rgba(0,0,0,0.06)'),
                        yaxis2=dict(title='AUC', overlaying='y', side='right',
                                    range=[0, 1], gridcolor='rgba(0,0,0,0)'),
                        height=280, template='plotly_white',
                        plot_bgcolor='rgba(248,250,252,1)',
                        legend=dict(orientation='h', y=1.02, x=0),
                    )
                    st.plotly_chart(fig_loss, use_container_width=True)

            with col_r:
                # 3D learning curve
                if lc and lc.get('epochs') and lc.get('auc'):
                    ep = np.array(lc['epochs'])
                    au = np.array(lc.get('auc', np.zeros_like(ep)))
                    ap = np.array(lc.get('aupr', np.zeros_like(ep)))
                    fig_3dlc = go.Figure()
                    fig_3dlc.add_trace(go.Scatter3d(
                        x=ep, y=au, z=np.zeros_like(ep),
                        mode='lines+markers', name='AUC',
                        marker=dict(size=3.5, color='#3B82F6', opacity=0.9),
                        line=dict(color='#3B82F6', width=5),
                    ))
                    fig_3dlc.add_trace(go.Scatter3d(
                        x=ep, y=ap, z=np.ones_like(ep),
                        mode='lines+markers', name='AUPR',
                        marker=dict(size=3.5, color='#8B5CF6', opacity=0.9),
                        line=dict(color='#8B5CF6', width=5),
                    ))
                    fig_3dlc.update_layout(
                        title=dict(text='3D Learning Curve', font=dict(size=13)),
                        scene=dict(
                            xaxis=dict(title='Epoch',
                                       backgroundcolor='rgba(238,242,255,0.7)'),
                            yaxis=dict(title='Score',
                                       backgroundcolor='rgba(238,242,255,0.7)'),
                            zaxis=dict(title='Metric',
                                       tickvals=[0, 1], ticktext=['AUC', 'AUPR'],
                                       backgroundcolor='rgba(238,242,255,0.7)'),
                            bgcolor='rgba(248,250,252,1)',
                            camera=dict(eye=dict(x=1.4, y=-1.4, z=0.8)),
                        ),
                        height=380, paper_bgcolor='rgba(255,255,255,1)',
                    )
                    st.plotly_chart(fig_3dlc, use_container_width=True)

                top_preds = data4.get('top_predictions', [])
                if top_preds:
                    df_top = pd.DataFrame(top_preds).rename(columns={
                        'drug_idx': 'Thuốc', 'disease_idx': 'Bệnh',
                        'prob': 'Điểm dự đoán', 'label': 'Nhãn thật',
                        'pred': 'Nhãn dự đoán',
                    })
                    st.markdown('**Top dự đoán mẫu**')
                    st.dataframe(df_top, use_container_width=True, hide_index=True,
                                 column_config={
                                     'Điểm dự đoán': st.column_config.ProgressColumn(
                                         'Điểm dự đoán', min_value=0.0, max_value=1.0,
                                         format='%.4f'),
                                 })
