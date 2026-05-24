"""
4_ablation.py — So sánh các phiên bản pipeline của mô hình (Ablation Study)

Hiển thị kết quả so sánh 7 biến thể kiến trúc để phân tích đóng góp
từng thành phần của mô hình AMNTDDA_Fuzzy (Full Model).

Nhóm biến thể:
  1. sim_only           — Chỉ dùng Similarity
  2. gcn_only           — Chỉ dùng GCN (Heterogeneous Network)
  3. sim_transformer    — Similarity + Transformer
  4. gcn_transformer    — Heterogeneous Graph + Transformer nâng cao
  5. sim_gcn            — Fusion: Similarity + GCN
  6. sim_transformer_gcn— Fusion: Similarity + Transformer + GCN
  7. full               — Full Model (Similarity + GCN + Modality Interaction)
"""

import os
import sys
import json
import subprocess
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.api_client import (
    get_ablation_comparison, get_ablation_all_variants, get_ablation_variant, _AI,
)

# ── Constants ─────────────────────────────────────────────────────────
VARIANT_ORDER = [
    'sim_only', 'gcn_only',
    'sim_transformer', 'gcn_transformer',
    'sim_gcn', 'sim_transformer_gcn',
    'full',
]

VARIANT_META = {
    'sim_only': dict(
        name_vi='Chỉ dùng Similarity',
        name_en='Similarity Only',
        group='Similarity Only',
        color='#3B82F6', icon='📐',
        components={'Similarity': True, 'GCN': False, 'Transformer': False, 'Cross-Modal': False},
        desc='Chỉ Graph Transformer trên đồ thị tương đồng thuốc–thuốc và bệnh–bệnh. '
             'Không dùng thông tin mạng lưới sinh hóa.',
    ),
    'gcn_only': dict(
        name_vi='Chỉ dùng GCN',
        name_en='Network (GCN) Only',
        group='Network Only',
        color='#10B981', icon='🕸️',
        components={'Similarity': False, 'GCN': True, 'Transformer': False, 'Cross-Modal': False},
        desc='HGT (Heterogeneous Graph Transformer) trên mạng dị thể thuốc–bệnh–protein. '
             'Không dùng ma trận tương đồng.',
    ),
    'sim_transformer': dict(
        name_vi='Similarity + Transformer',
        name_en='Similarity + Feature Extraction',
        group='Similarity FE',
        color='#8B5CF6', icon='🧬',
        components={'Similarity': True, 'GCN': False, 'Transformer': True, 'Cross-Modal': False},
        desc='Graph Transformer trên đồ thị tương đồng, sau đó TransformerEncoder '
             'self-attention để tinh chỉnh biểu diễn. Không có GCN.',
    ),
    'gcn_transformer': dict(
        name_vi='Heterogeneous Graph + Transformer nâng cao',
        name_en='Network + Advanced Feature Extraction',
        group='Network FE',
        color='#F59E0B', icon='🔬',
        components={'Similarity': False, 'GCN': True, 'Transformer': True, 'Cross-Modal': False},
        desc='HGT trên mạng dị thể, sau đó TransformerEncoder self-attention '
             'để nâng cao biểu diễn mạng lưới. Không dùng similarity.',
    ),
    'sim_gcn': dict(
        name_vi='Fusion: Similarity + GCN',
        name_en='Fusion (Sim + GCN)',
        group='Fusion',
        color='#EF4444', icon='🔗',
        components={'Similarity': True, 'GCN': True, 'Transformer': False, 'Cross-Modal': False},
        desc='Kết hợp trực tiếp Similarity (GT) và GCN (HGT) bằng concatenation. '
             'Không có transformer tương tác đa phương thức.',
    ),
    'sim_transformer_gcn': dict(
        name_vi='Fusion: Similarity + Transformer + GCN',
        name_en='Fusion (Sim + Transformer + GCN)',
        group='Fusion',
        color='#F97316', icon='⚡',
        components={'Similarity': True, 'GCN': True, 'Transformer': True, 'Cross-Modal': False},
        desc='Transformer tinh chỉnh similarity độc lập, sau đó ghép với GCN. '
             'Không có tương tác cross-modal giữa hai phương thức.',
    ),
    'full': dict(
        name_vi='Full Model: Similarity + GCN + Modality Interaction',
        name_en='Full Model (AMNTDDA_Fuzzy)',
        group='Full Model',
        color='#6366F1', icon='🎯',
        components={'Similarity': True, 'GCN': True, 'Transformer': True, 'Cross-Modal': True},
        desc='Mô hình đầy đủ: Similarity + HGT + Cross-Modal TransformerEncoder. '
             'Hai phương thức tương tác qua self-attention.',
    ),
}

METRICS   = ['AUC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC']
GROUPS    = {
    'Similarity Only': ['sim_only'],
    'Network Only':    ['gcn_only'],
    'Similarity FE':   ['sim_transformer'],
    'Network FE':      ['gcn_transformer'],
    'Fusion':          ['sim_gcn', 'sim_transformer_gcn'],
    'Full Model':      ['full'],
}

GROUP_COLORS = {
    'Similarity Only': '#3B82F6',
    'Network Only':    '#10B981',
    'Similarity FE':   '#8B5CF6',
    'Network FE':      '#F59E0B',
    'Fusion':          '#EF4444',
    'Full Model':      '#6366F1',
}

AMDGT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'AMDGT_main')
)
AI_ENGINE_SRC = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'AI_ENGINE', 'src')
)


# ══════════════════════════════════════════════════════════════════════
# Page setup
# ══════════════════════════════════════════════════════════════════════
st.title('🔬 Phân tích so sánh phiên bản Pipeline (Ablation Study)')
st.caption(
    'So sánh 7 biến thể kiến trúc để xem xét đóng góp của từng thành phần: '
    '**Similarity**, **GCN (Heterogeneous Network)**, **Transformer Feature Extraction**, '
    'và **Cross-Modal Modality Interaction**.'
)

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('### ⚙️ Cài đặt')
    dataset = st.selectbox('Dataset', ['B-dataset', 'C-dataset', 'F-dataset'],
                           index=1, key='ablation_dataset')

    st.markdown('---')
    st.markdown('### 📊 Hiển thị')
    selected_metrics = st.multiselect(
        'Metrics hiển thị',
        options=METRICS,
        default=['AUC', 'AUPR', 'F1'],
        key='ablation_metrics',
    )
    show_std = st.checkbox('Hiện sai số chuẩn (±std)', value=True)
    show_folds = st.checkbox('Hiện chi tiết từng fold', value=False)

    st.markdown('---')
    st.markdown('### 🚀 Huấn luyện ablation')
    variants_to_run = st.multiselect(
        'Chọn biến thể cần train',
        options=VARIANT_ORDER,
        default=VARIANT_ORDER,
        format_func=lambda v: f"{VARIANT_META[v]['icon']} {v}",
    )
    epochs_abl = st.number_input('Số epochs', min_value=50, max_value=2000,
                                  value=300, step=50, key='abl_epochs')
    run_btn = st.button('▶  Chạy huấn luyện', type='primary', use_container_width=True)
    st.markdown(
        '**Lưu ý:** Mỗi biến thể train đầy đủ k-fold.  \n'
        'Thời gian: ~30–120 phút tùy GPU/CPU.'
    )


# ── Run ablation training ─────────────────────────────────────────────
if run_btn:
    if not variants_to_run:
        st.warning('Chọn ít nhất một biến thể để train.')
    else:
        train_script = os.path.join(AI_ENGINE_SRC, 'train_DDA_ablation.py')
        python_exe   = sys.executable
        variants_arg = ','.join(variants_to_run)
        cmd = [
            python_exe, train_script,
            '--dataset', dataset,
            '--epochs', str(epochs_abl),
            '--variants', variants_arg,
        ]
        label = ', '.join(f"{VARIANT_META[v]['icon']} {v}" for v in variants_to_run)
        with st.status(f'Đang huấn luyện: {label} …', expanded=True) as status:
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True,
                    cwd=AI_ENGINE_SRC, timeout=7200,
                    encoding='utf-8', errors='replace',
                )
                if result.returncode == 0:
                    status.update(label='✅ Hoàn thành!', state='complete')
                    st.success('Huấn luyện xong. Làm mới trang để xem kết quả mới nhất.')
                    with st.expander('Xem log'):
                        tail = result.stdout[-4000:] if len(result.stdout) > 4000 else result.stdout
                        st.code(tail)
                else:
                    status.update(label='❌ Lỗi', state='error')
                    st.error('Có lỗi khi huấn luyện:')
                    st.code(result.stderr[-3000:])
            except subprocess.TimeoutExpired:
                status.update(label='⏱️ Timeout', state='error')
                st.error('Quá 2 giờ — tăng thời gian timeout hoặc giảm epochs.')
            except Exception as ex:
                status.update(label='❌ Exception', state='error')
                st.error(str(ex))

st.markdown('---')

# ── Load data ─────────────────────────────────────────────────────────
@st.cache_data(ttl=30, show_spinner=False)
def _load_ablation(dataset: str) -> dict:
    return get_ablation_all_variants(dataset)


with st.spinner('Đang tải kết quả ablation …'):
    abl_data = _load_ablation(dataset)

variants_info = abl_data.get('variants', {})
any_trained   = any(
    variants_info.get(v, {}).get('AUC_mean') is not None
    for v in VARIANT_ORDER
)

if not any_trained:
    st.info(
        '**Chưa có kết quả ablation.**  \n\n'
        'Sử dụng bảng điều khiển bên trái để chạy huấn luyện, '
        'hoặc chạy thủ công:\n\n'
        '```bash\npython AI_ENGINE/src/train_DDA_ablation.py '
        f'--dataset {dataset} --variants all --epochs 300\n```'
    )

# ── Pipeline Architecture Overview ────────────────────────────────────
st.subheader('🏗️ Kiến trúc các phiên bản Pipeline')

COMPONENT_COLS = ['Similarity', 'GCN', 'Transformer', 'Cross-Modal']
component_emojis = {'Similarity': '📐', 'GCN': '🕸️', 'Transformer': '🔄', 'Cross-Modal': '⚡'}

arch_rows = []
for variant in VARIANT_ORDER:
    meta = VARIANT_META[variant]
    row = {
        'Variant': f"{meta['icon']} {variant}",
        'Nhóm': meta['group'],
        'Tên tiếng Việt': meta['name_vi'],
    }
    for comp in COMPONENT_COLS:
        row[comp] = '✅' if meta['components'].get(comp) else '—'
    arch_rows.append(row)

arch_df = pd.DataFrame(arch_rows)
st.dataframe(
    arch_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        'Variant': st.column_config.TextColumn('Biến thể', width='medium'),
        'Nhóm': st.column_config.TextColumn('Nhóm', width='small'),
        'Tên tiếng Việt': st.column_config.TextColumn('Mô tả', width='large'),
        'Similarity': st.column_config.TextColumn('📐 Similarity', width='small'),
        'GCN': st.column_config.TextColumn('🕸️ GCN', width='small'),
        'Transformer': st.column_config.TextColumn('🔄 Transformer', width='small'),
        'Cross-Modal': st.column_config.TextColumn('⚡ Cross-Modal', width='small'),
    },
)

# ── Architecture flow diagram ──────────────────────────────────────────
with st.expander('📊 Sơ đồ đóng góp thành phần', expanded=False):
    st.markdown("""
```
Đầu vào dữ liệu (Drug / Disease / Protein)
    │
    ├─ [Similarity] GIP + Fingerprint → Graph Transformer ──────────────────┐
    │       ↓ dr_sim / di_sim (200-dim)                                       │
    │       ├─ [Transformer] TransformerEncoder self-attention               │
    │       │       (sim_transformer, sim_transformer_gcn)                   │
    │                                                                         │
    └─ [GCN] HGT Heterogeneous Graph (Drug+Disease+Protein) ─────────────────┤
            ↓ dr_hgt / di_hgt (200-dim)                                      │
            └─ [Transformer] TransformerEncoder self-attention               │
                    (gcn_transformer)                                         │
                                                                              │
    Fusion / Combination (400-dim embedding):                                 │
        sim_gcn          → cat(sim, hgt)                no transformer        │
        sim_transformer_gcn → cat(refined_sim, hgt)    independent           │
        full             → TransformerEncoder(sim, hgt) Cross-Modal ←────────┘
                                        ↓
                              MLP (400 → 1024 → 256 → 2)
                                        ↓
                              Dự đoán liên kết thuốc–bệnh
```
    """)

st.markdown('---')

# ══════════════════════════════════════════════════════════════════════
# Results section (only when trained data exists)
# ══════════════════════════════════════════════════════════════════════
if any_trained:

    # ── Build comparison DataFrame ─────────────────────────────────
    rows = []
    for variant in VARIANT_ORDER:
        info = variants_info.get(variant, {})
        auc  = info.get('AUC_mean')
        if auc is None:
            continue
        meta = VARIANT_META.get(variant, {})
        row  = {
            'Variant':  f"{meta.get('icon','')}\u2009{variant}",
            'Nhóm':     meta.get('group', ''),
            'color':    meta.get('color', '#888'),
        }
        for m in METRICS:
            row[f'{m}_mean'] = info.get(f'{m}_mean')
            row[f'{m}_std']  = info.get(f'{m}_std')
        rows.append(row)

    comp_df = pd.DataFrame(rows)

    # ── Summary table ──────────────────────────────────────────────
    st.subheader('📋 Bảng so sánh chỉ số')

    display_cols = {'Variant': 'Biến thể', 'Nhóm': 'Nhóm'}
    for m in selected_metrics or METRICS:
        display_cols[f'{m}_mean'] = f'{m} (Mean)'
        if show_std:
            display_cols[f'{m}_std'] = f'{m} (Std)'

    show_df = comp_df[[c for c in display_cols if c in comp_df.columns]].copy()
    show_df.rename(columns=display_cols, inplace=True)

    # Highlight best value per metric column
    def _style_best(s):
        """Green background for max, red for min (metrics where higher is better)."""
        if s.dtype not in (float, 'float64') or s.isna().all():
            return [''] * len(s)
        best  = s.max()
        worst = s.min()
        return [
            'background-color:#166534; color:#dcfce7; font-weight:bold' if v == best
            else 'background-color:#7f1d1d; color:#fee2e2' if v == worst
            else ''
            for v in s
        ]

    mean_cols = [c for c in show_df.columns if '(Mean)' in c]
    styled = show_df.style.apply(_style_best, subset=mean_cols)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Group-level summary ─────────────────────────────────────────
    st.subheader('📊 Tổng quan theo nhóm')
    group_cols = st.columns(len(GROUPS))
    for col, (group_name, group_variants) in zip(group_cols, GROUPS.items()):
        with col:
            group_color = GROUP_COLORS.get(group_name, '#888')
            best_auc = max(
                (variants_info.get(v, {}).get('AUC_mean') or 0.0
                 for v in group_variants), default=0.0
            )
            trained_in_group = [
                v for v in group_variants
                if variants_info.get(v, {}).get('AUC_mean') is not None
            ]
            st.markdown(
                f"""
                <div style="border:2px solid {group_color}; border-radius:10px;
                            padding:10px; text-align:center; background:#1e1e2e">
                    <div style="font-size:1.4rem">{VARIANT_META[group_variants[0]]['icon']}</div>
                    <div style="font-weight:600; color:{group_color}; font-size:.8rem">
                        {group_name}
                    </div>
                    <div style="font-size:1.1rem; font-weight:bold; margin-top:4px">
                        {"N/A" if best_auc == 0 else f"AUC {best_auc:.4f}"}
                    </div>
                    <div style="font-size:.7rem; color:#9ca3af">
                        {len(trained_in_group)}/{len(group_variants)} trained
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<br>', unsafe_allow_html=True)

    # ── Tabs: Bar | Radar | Per-Fold | Contribution ─────────────────
    tab_bar, tab_radar, tab_fold, tab_contrib = st.tabs([
        '📊 Bar Chart',
        '🕸️ Radar Chart',
        '📈 Per-Fold Details',
        '🔍 Phân tích đóng góp',
    ])

    # ─ Tab: Bar chart ──────────────────────────────────────────────
    with tab_bar:
        bar_metric = st.selectbox('Chọn metric', selected_metrics or ['AUC'],
                                   key='bar_metric')
        chart_variants = [v for v in VARIANT_ORDER
                          if variants_info.get(v, {}).get(f'{bar_metric}_mean') is not None]
        if chart_variants:
            means = [variants_info[v][f'{bar_metric}_mean'] for v in chart_variants]
            stds  = [variants_info[v].get(f'{bar_metric}_std', 0) or 0 for v in chart_variants]
            labels= [f"{VARIANT_META[v]['icon']}\u2009{v}" for v in chart_variants]
            colors= [VARIANT_META[v]['color'] for v in chart_variants]

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=labels, y=means,
                error_y=dict(type='data', array=stds, visible=show_std),
                marker_color=colors,
                text=[f'{m:.4f}' for m in means],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>'
                              f'{bar_metric}: %{{y:.4f}}<br>'
                              'Std: %{error_y.array:.4f}<extra></extra>',
            ))
            # Add full-model baseline line
            full_val = variants_info.get('full', {}).get(f'{bar_metric}_mean')
            if full_val and 'full' in chart_variants:
                fig_bar.add_hline(
                    y=full_val, line_dash='dash',
                    line_color='#6366F1',
                    annotation_text=f'Full Model: {full_val:.4f}',
                    annotation_position='top right',
                )
            fig_bar.update_layout(
                title=f'{bar_metric} — So sánh 7 biến thể pipeline',
                yaxis_title=bar_metric,
                xaxis_title='Pipeline Variant',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                margin=dict(t=50, b=80, l=40, r=40),
                yaxis=dict(range=[max(0, min(means) - 0.05), min(1.01, max(means) + 0.06)]),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info('Chưa có dữ liệu để vẽ biểu đồ.')

    # ─ Tab: Radar chart ────────────────────────────────────────────
    with tab_radar:
        radar_variants = st.multiselect(
            'Chọn biến thể hiển thị',
            options=[v for v in VARIANT_ORDER
                     if variants_info.get(v, {}).get('AUC_mean') is not None],
            default=[v for v in VARIANT_ORDER
                     if variants_info.get(v, {}).get('AUC_mean') is not None],
            key='radar_variants',
        )
        radar_metrics  = st.multiselect('Chọn metrics', METRICS, default=METRICS,
                                         key='radar_metrics_sel')

        if radar_variants and radar_metrics:
            cats = radar_metrics + [radar_metrics[0]]
            fig_radar = go.Figure()
            for variant in radar_variants:
                info = variants_info.get(variant, {})
                vals = [info.get(f'{m}_mean', 0) or 0 for m in radar_metrics]
                vals_closed = vals + [vals[0]]
                meta = VARIANT_META.get(variant, {})
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals_closed, theta=cats,
                    fill='toself', opacity=0.35,
                    line=dict(color=meta.get('color', '#888'), width=2),
                    name=f"{meta.get('icon','')}\u2009{variant}",
                    hovertemplate='<b>%{theta}</b>: %{r:.4f}<extra>' + variant + '</extra>',
                ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1],
                                    tickfont=dict(color='#9ca3af', size=9)),
                    angularaxis=dict(tickfont=dict(color='#e2e8f0')),
                    bgcolor='rgba(0,0,0,0)',
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                legend=dict(font=dict(size=11)),
                title='Radar: So sánh toàn diện các biến thể',
                margin=dict(t=60, b=40, l=40, r=40),
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info('Chọn ít nhất một biến thể và một metric.')

    # ─ Tab: Per-fold ───────────────────────────────────────────────
    with tab_fold:
        fold_variant = st.selectbox(
            'Chọn biến thể',
            options=[v for v in VARIANT_ORDER
                     if variants_info.get(v, {}).get('AUC_mean') is not None],
            format_func=lambda v: f"{VARIANT_META[v]['icon']} {v}",
            key='fold_variant_sel',
        )
        fold_metric = st.selectbox('Metric', METRICS, key='fold_metric_sel')

        @st.cache_data(ttl=30, show_spinner=False)
        def _load_variant_folds(dataset: str, variant: str) -> dict:
            return get_ablation_variant(dataset, variant)

        fold_data = _load_variant_folds(dataset, fold_variant)
        folds     = fold_data.get('folds', [])

        if folds:
            fold_df = pd.DataFrame(folds)
            numeric_fold_df = fold_df[pd.to_numeric(fold_df['fold'], errors='coerce').notna()].copy()
            numeric_fold_df['fold'] = numeric_fold_df['fold'].astype(int)

            if fold_metric in numeric_fold_df.columns:
                fig_fold = go.Figure()
                fig_fold.add_trace(go.Bar(
                    x=numeric_fold_df['fold'],
                    y=numeric_fold_df[fold_metric],
                    marker_color=VARIANT_META.get(fold_variant, {}).get('color', '#888'),
                    text=[f'{v:.4f}' for v in numeric_fold_df[fold_metric]],
                    textposition='outside',
                    name=fold_metric,
                    hovertemplate='Fold %{x}<br>' + fold_metric + ': %{y:.4f}<extra></extra>',
                ))
                mean_val = numeric_fold_df[fold_metric].mean()
                fig_fold.add_hline(
                    y=mean_val, line_dash='dash', line_color='#e2e8f0',
                    annotation_text=f'Mean: {mean_val:.4f}',
                )
                fig_fold.update_layout(
                    title=f'{fold_metric} per fold — {fold_variant}',
                    xaxis=dict(title='Fold', tickmode='linear'),
                    yaxis=dict(title=fold_metric,
                               range=[max(0, mean_val - 0.1), min(1.01, mean_val + 0.1)]),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0'),
                    margin=dict(t=50, b=40),
                )
                st.plotly_chart(fig_fold, use_container_width=True)

            st.markdown('#### Dữ liệu từng fold')
            st.dataframe(numeric_fold_df, use_container_width=True, hide_index=True)
        else:
            st.info(f'Chưa có dữ liệu fold cho biến thể **{fold_variant}**.')

    # ─ Tab: Contribution analysis ──────────────────────────────────
    with tab_contrib:
        st.markdown(
            '#### 🔍 Phân tích đóng góp từng thành phần\n\n'
            'So sánh theo cặp để đo lường lợi ích của từng thành phần bổ sung.'
        )

        contrib_metric = st.selectbox('Metric phân tích', METRICS, index=0,
                                       key='contrib_metric')

        def _delta(variant_new: str, variant_base: str, metric: str) -> str | None:
            v_new  = variants_info.get(variant_new, {}).get(f'{metric}_mean')
            v_base = variants_info.get(variant_base, {}).get(f'{metric}_mean')
            if v_new is None or v_base is None:
                return None
            delta = v_new - v_base
            sign  = '+' if delta >= 0 else ''
            arrow = '↑' if delta > 0.001 else ('↓' if delta < -0.001 else '→')
            return f'{arrow} {sign}{delta:.4f}'

        contrib_rows = [
            ('Sim → Sim+Trans',            'sim_transformer',    'sim_only',
             'Transformer có cải thiện Similarity không?'),
            ('GCN → GCN+Trans',            'gcn_transformer',    'gcn_only',
             'Transformer có cải thiện GCN không?'),
            ('Sim → Sim+GCN',              'sim_gcn',            'sim_only',
             'Thêm GCN vào Similarity mang lại gì?'),
            ('GCN → GCN+Sim',              'sim_gcn',            'gcn_only',
             'Thêm Similarity vào GCN mang lại gì?'),
            ('Sim+GCN → Sim+Trans+GCN',    'sim_transformer_gcn','sim_gcn',
             'Transformer (parallel) cải thiện Fusion?'),
            ('Sim+Trans+GCN → Full',       'full',               'sim_transformer_gcn',
             'Cross-modal interaction cải thiện gì?'),
            ('Baseline (sim_only) → Full', 'full',               'sim_only',
             'Tổng lợi ích từ Similarity-only lên Full Model'),
        ]

        contrib_data = []
        for label, new, base, desc in contrib_rows:
            delta_str = _delta(new, base, contrib_metric)
            new_val   = variants_info.get(new, {}).get(f'{contrib_metric}_mean')
            base_val  = variants_info.get(base, {}).get(f'{contrib_metric}_mean')
            contrib_data.append({
                'So sánh': label,
                'Mô tả': desc,
                'Biến thể mới': new,
                'Biến thể cơ sở': base,
                f'{contrib_metric} (mới)': f'{new_val:.4f}' if new_val else 'N/A',
                f'{contrib_metric} (cơ sở)': f'{base_val:.4f}' if base_val else 'N/A',
                'Δ': delta_str if delta_str else 'N/A',
            })

        contrib_df = pd.DataFrame(contrib_data)

        def _style_delta(val):
            if not isinstance(val, str) or val == 'N/A':
                return ''
            if val.startswith('↑'):
                return 'color:#4ade80; font-weight:bold'
            if val.startswith('↓'):
                return 'color:#f87171'
            return 'color:#9ca3af'

        styled_contrib = contrib_df.style.applymap(_style_delta, subset=['Δ'])
        st.dataframe(styled_contrib, use_container_width=True, hide_index=True)

        # ── Delta bar chart
        delta_vals = []
        delta_labels = []
        delta_colors = []
        for row in contrib_data:
            delta_str = row['Δ']
            if delta_str == 'N/A':
                continue
            try:
                val = float(delta_str.split()[-1])
                delta_vals.append(val)
                delta_labels.append(row['So sánh'])
                delta_colors.append('#4ade80' if val >= 0 else '#f87171')
            except ValueError:
                continue

        if delta_vals:
            fig_delta = go.Figure(go.Bar(
                x=delta_labels,
                y=delta_vals,
                marker_color=delta_colors,
                text=[f'{v:+.4f}' for v in delta_vals],
                textposition='outside',
                hovertemplate='%{x}<br>Δ ' + contrib_metric + ': %{y:+.4f}<extra></extra>',
            ))
            fig_delta.add_hline(y=0, line_color='#9ca3af', line_width=1)
            fig_delta.update_layout(
                title=f'Đóng góp thành phần (Δ {contrib_metric})',
                yaxis_title=f'Δ {contrib_metric}',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0'),
                margin=dict(t=50, b=80, l=40, r=40),
                xaxis=dict(tickangle=-25),
            )
            st.plotly_chart(fig_delta, use_container_width=True)

    # ── Variant detail cards ────────────────────────────────────────
    st.markdown('---')
    st.subheader('🃏 Chi tiết từng biến thể')

    card_cols = st.columns(3)
    col_idx   = 0
    for variant in VARIANT_ORDER:
        info = variants_info.get(variant, {})
        meta = VARIANT_META.get(variant, {})
        auc  = info.get('AUC_mean')
        aupr = info.get('AUPR_mean')
        f1   = info.get('F1_mean')
        color= meta.get('color', '#888')

        with card_cols[col_idx % 3]:
            trained_badge = (
                f'<span style="background:#166534;color:#dcfce7;padding:2px 6px;'
                f'border-radius:4px;font-size:.7rem">✅ Trained</span>'
                if auc is not None else
                f'<span style="background:#7f1d1d;color:#fee2e2;padding:2px 6px;'
                f'border-radius:4px;font-size:.7rem">⏳ Not trained</span>'
            )
            metrics_html = ''
            if auc is not None:
                metrics_html = (
                    f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:6px">'
                    f'<span style="background:#1e293b;padding:2px 6px;border-radius:4px;font-size:.8rem">'
                    f'AUC {auc:.4f}</span>'
                    f'<span style="background:#1e293b;padding:2px 6px;border-radius:4px;font-size:.8rem">'
                    f'AUPR {aupr:.4f}</span>'
                    f'<span style="background:#1e293b;padding:2px 6px;border-radius:4px;font-size:.8rem">'
                    f'F1 {f1:.4f}</span>'
                    f'</div>'
                )
            _icon_map = {"Similarity":"📐","GCN":"🕸️","Transformer":"🔄","Cross-Modal":"⚡"}
            comp_icons = ''.join(
                f'<span title="{k}" style="font-size:1rem;opacity:{"1" if v else "0.2"}">'
                f'{_icon_map.get(k, "❓")}</span>'
                for k, v in meta.get('components', {}).items()
            )
            st.markdown(
                f"""
                <div style="border:2px solid {color};border-radius:12px;
                            padding:14px;margin-bottom:12px;background:#0f172a">
                    <div style="display:flex;align-items:center;gap:8px">
                        <span style="font-size:1.5rem">{meta.get('icon','')}</span>
                        <div>
                            <div style="font-weight:700;color:{color};font-size:.9rem">{variant}</div>
                            <div style="font-size:.75rem;color:#9ca3af">{meta.get('name_vi','')}</div>
                        </div>
                        <div style="margin-left:auto">{trained_badge}</div>
                    </div>
                    <div style="margin-top:8px;display:flex;gap:4px">{comp_icons}</div>
                    <div style="font-size:.75rem;color:#6b7280;margin-top:6px">{meta.get('desc','')}</div>
                    {metrics_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
        col_idx += 1

else:
    # Not trained yet — show description cards only
    st.subheader('🃏 Mô tả các biến thể')
    card_cols = st.columns(3)
    for i, variant in enumerate(VARIANT_ORDER):
        meta  = VARIANT_META[variant]
        color = meta['color']
        _icon_map2 = {"Similarity":"📐","GCN":"🕸️","Transformer":"🔄","Cross-Modal":"⚡"}
        comp_icons = ''.join(
            f'<span title="{k}" style="font-size:1rem;opacity:{"1" if v else "0.2"}">'
            f'{_icon_map2.get(k, "❓")}</span>'
            for k, v in meta['components'].items()
        )
        with card_cols[i % 3]:
            st.markdown(
                f"""
                <div style="border:2px solid {color};border-radius:12px;
                            padding:14px;margin-bottom:12px;background:#0f172a">
                    <div style="font-size:1.5rem">{meta['icon']}</div>
                    <div style="font-weight:700;color:{color};font-size:.9rem;margin-top:4px">{variant}</div>
                    <div style="font-size:.75rem;color:#9ca3af">{meta['name_vi']}</div>
                    <div style="margin-top:6px;display:flex;gap:4px">{comp_icons}</div>
                    <div style="font-size:.75rem;color:#6b7280;margin-top:6px">{meta['desc']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
