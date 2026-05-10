"""
AMNTDDA_Ablation.py — Configurable ablation model for pipeline comparison.

Supports 7 pipeline variants to isolate the contribution of each component:

  1. sim_only           — Chỉ dùng Similarity (GIP + fingerprint → Graph Transformer)
  2. gcn_only           — Chỉ dùng GCN (Heterogeneous Network → HGT)
  3. sim_transformer    — Similarity + Transformer (self-attention on sim features)
  4. gcn_transformer    — Network + Transformer nâng cao (HGT + self-attention)
  5. sim_gcn            — Fusion: Similarity + GCN (no modality interaction)
  6. sim_transformer_gcn— Fusion: Similarity + Transformer + GCN (parallel enhancement)
  7. full               — Full model: Similarity + GCN + Cross-modal Interaction

All variants use the same MLP head (400 → 1024 → 1024 → 256 → 2) so
comparison is fair (equal prediction capacity, varying feature construction).
"""

import dgl
import dgl.nn.pytorch
import torch
import torch.nn as nn
from model import gt_net_drug, gt_net_disease

device_global = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Ablation variant definitions ──────────────────────────────────────
ABLATION_CONFIGS: dict[str, dict] = {
    'sim_only': dict(
        use_sim=True, use_gcn=False, use_trans=False, cross_modal=False,
        name_vi='Chỉ dùng Similarity',
        name_en='Similarity Only',
        group='similarity_only',
        desc='Chỉ dùng ma trận tương đồng (GIP + fingerprint) qua Graph Transformer. '
             'Không có thông tin mạng lưới dị thể.',
        color='#3B82F6',
        icon='📐',
    ),
    'gcn_only': dict(
        use_sim=False, use_gcn=True, use_trans=False, cross_modal=False,
        name_vi='Chỉ dùng GCN (Heterogeneous Network)',
        name_en='Network (GCN) Only',
        group='network_only',
        desc='Chỉ dùng mạng lưới dị thể thuốc–bệnh–protein qua HGT. '
             'Không có thông tin tương đồng.',
        color='#10B981',
        icon='🕸️',
    ),
    'sim_transformer': dict(
        use_sim=True, use_gcn=False, use_trans=True, cross_modal=False,
        name_vi='Similarity + Transformer',
        name_en='Similarity Feature Extraction',
        group='similarity_fe',
        desc='Similarity → Graph Transformer → self-attention TransformerEncoder '
             'để tinh chỉnh biểu diễn tương đồng. Không có GCN.',
        color='#8B5CF6',
        icon='🧬',
    ),
    'gcn_transformer': dict(
        use_sim=False, use_gcn=True, use_trans=True, cross_modal=False,
        name_vi='Heterogeneous Graph + Transformer nâng cao',
        name_en='Network Feature Extraction (Advanced)',
        group='network_fe',
        desc='HGT trên mạng dị thể + TransformerEncoder self-attention để '
             'nâng cao biểu diễn mạng lưới. Không có similarity.',
        color='#F59E0B',
        icon='🔬',
    ),
    'sim_gcn': dict(
        use_sim=True, use_gcn=True, use_trans=False, cross_modal=False,
        name_vi='Fusion: Similarity + GCN',
        name_en='Fusion (Sim + GCN)',
        group='fusion',
        desc='Kết hợp trực tiếp Similarity (Graph Transformer) và GCN (HGT) '
             'qua concatenation. Không có transformer tương tác đa phương thức.',
        color='#EF4444',
        icon='🔗',
    ),
    'sim_transformer_gcn': dict(
        use_sim=True, use_gcn=True, use_trans=True, cross_modal=False,
        name_vi='Fusion: Similarity + Transformer + GCN',
        name_en='Fusion (Sim + Transformer + GCN)',
        group='fusion',
        desc='Similarity được tinh chỉnh độc lập bằng Transformer self-attention, '
             'sau đó ghép nối với GCN (HGT). Không có tương tác cross-modal.',
        color='#F97316',
        icon='⚡',
    ),
    'full': dict(
        use_sim=True, use_gcn=True, use_trans=True, cross_modal=True,
        name_vi='Full Model: Similarity + GCN + Modality Interaction',
        name_en='Full Model (AMNTDDA_Fuzzy)',
        group='full_model',
        desc='Mô hình đầy đủ: Similarity + HGT + Cross-modal TransformerEncoder. '
             'Hai phương thức tương tác lẫn nhau qua attention.',
        color='#6366F1',
        icon='🎯',
    ),
}

VARIANT_ORDER = [
    'sim_only', 'gcn_only',
    'sim_transformer', 'gcn_transformer',
    'sim_gcn', 'sim_transformer_gcn',
    'full',
]


class AMNTDDA_Ablation(nn.Module):
    """
    Configurable ablation model for drug–disease association prediction.

    Parameters
    ----------
    args : argparse.Namespace
        Model hyper-parameters (same as AMNTDDA_Fuzzy).
    mode : str
        One of the keys in ABLATION_CONFIGS.
    """

    def __init__(self, args, mode: str = 'full'):
        super().__init__()
        if mode not in ABLATION_CONFIGS:
            raise ValueError(f"Unknown ablation mode '{mode}'. "
                             f"Choose from: {list(ABLATION_CONFIGS)}")
        self.args = args
        self.mode = mode
        cfg = ABLATION_CONFIGS[mode]
        self.use_sim    = cfg['use_sim']
        self.use_gcn    = cfg['use_gcn']
        self.use_trans  = cfg['use_trans']
        self.cross_modal = cfg['cross_modal']
        device = device_global

        # ── Similarity stream ─────────────────────────────────────────
        if self.use_sim:
            self.gt_drug = gt_net_drug.GraphTransformer(
                device, args.gt_layer, args.drug_number,
                args.gt_out_dim, args.gt_out_dim, args.gt_head, args.dropout)
            self.gt_disease = gt_net_disease.GraphTransformer(
                device, args.gt_layer, args.disease_number,
                args.gt_out_dim, args.gt_out_dim, args.gt_head, args.dropout)

        # ── Network stream (HGT) ──────────────────────────────────────
        if self.use_gcn:
            self.drug_linear    = nn.Linear(300, args.hgt_in_dim)
            self.protein_linear = nn.Linear(320, args.hgt_in_dim)

            hgt_mid = dgl.nn.pytorch.conv.HGTConv(
                args.hgt_in_dim, int(args.hgt_in_dim / args.hgt_head),
                args.hgt_head, 3, 3, args.dropout)
            hgt_last = dgl.nn.pytorch.conv.HGTConv(
                args.hgt_in_dim, args.hgt_head_dim,
                args.hgt_head, 3, 3, args.dropout)

            self.hgt = nn.ModuleList()
            for _ in range(args.hgt_layer - 1):
                self.hgt.append(hgt_mid)
            self.hgt.append(hgt_last)

        # ── Transformer stream ────────────────────────────────────────
        if self.use_trans:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=args.gt_out_dim, nhead=args.tr_head, batch_first=True)
            self.drug_trans    = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)
            self.disease_trans = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)

        # ── Prediction MLP (consistent 400-dim input for all variants) ─
        self.mlp = nn.Sequential(
            nn.Linear(args.gt_out_dim * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2),
        )

    # ── Forward ───────────────────────────────────────────────────────
    def forward(self, drdr_graph, didi_graph, drdipr_graph,
                drug_feature, disease_feature, protein_feature,
                sample, drug_topo=None, disease_topo=None):

        D  = self.args.drug_number
        Di = self.args.disease_number
        dim = self.args.gt_out_dim
        dev = drug_feature.device

        # ── 1. Similarity features ────────────────────────────────────
        if self.use_sim:
            dr_sim = self.gt_drug(drdr_graph)    # (D, dim)
            di_sim = self.gt_disease(didi_graph) # (Di, dim)

        # ── 2. Network (HGT) features ─────────────────────────────────
        if self.use_gcn:
            drug_proj    = self.drug_linear(drug_feature)
            protein_proj = self.protein_linear(protein_feature)
            feature_dict = {
                'drug':    drug_proj,
                'disease': disease_feature,
                'protein': protein_proj,
            }
            drdipr_graph.ndata['h'] = feature_dict
            g       = dgl.to_homogeneous(drdipr_graph, ndata='h')
            feature = torch.cat((drug_proj, disease_feature, protein_proj), dim=0)
            for layer in self.hgt:
                hgt_out = layer(g, feature, g.ndata['_TYPE'], g.edata['_TYPE'],
                                presorted=True)
                feature = hgt_out
            dr_hgt = hgt_out[:D, :]           # (D, dim)
            di_hgt = hgt_out[D:D + Di, :]     # (Di, dim)

        # ── 3. Construct 400-dim embedding per drug / disease ─────────
        #
        # Embedding rule (consistent 2*dim = 400 for all variants):
        #   sim_only           → cat(dr_sim, dr_sim)          no trans
        #   gcn_only           → cat(dr_hgt, dr_hgt)          no trans
        #   sim_transformer    → trans(stack(sim,sim))         cross=False
        #   gcn_transformer    → trans(stack(hgt,hgt))         cross=False
        #   sim_gcn            → cat(dr_sim, dr_hgt)           no trans
        #   sim_transformer_gcn→ cat(trans_sim[:0], dr_hgt)   independent trans on sim
        #   full               → trans(stack(sim, hgt))        cross-modal

        if self.use_sim and not self.use_gcn:
            # Similarity only (with or without transformer)
            dr_a, di_a = dr_sim, di_sim
            dr_b, di_b = dr_sim, di_sim  # b-slot repeated

        elif not self.use_sim and self.use_gcn:
            # Network only (with or without transformer)
            dr_a, di_a = dr_hgt, di_hgt
            dr_b, di_b = dr_hgt, di_hgt  # b-slot repeated

        else:
            # Both sim + gcn
            dr_a, di_a = dr_sim, di_sim
            dr_b, di_b = dr_hgt, di_hgt

        # ── 4. Apply transformer (if applicable) ──────────────────────
        if not self.use_trans:
            # No transformer: direct concatenation
            # sim_only / gcn_only duplicate: cat(a,a) = cat(sim,sim) or cat(hgt,hgt)
            # sim_gcn: cat(sim, hgt)
            if self.use_sim and not self.use_gcn:
                dr = torch.cat((dr_a, dr_a), dim=1)   # (D, 2*dim) repeated
                di = torch.cat((di_a, di_b), dim=1)   # b is di_sim for sim_only
            elif not self.use_sim and self.use_gcn:
                dr = torch.cat((dr_a, dr_b), dim=1)   # cat(hgt, hgt)
                di = torch.cat((di_a, di_b), dim=1)
            else:
                dr = torch.cat((dr_a, dr_b), dim=1)   # cat(sim, hgt)
                di = torch.cat((di_a, di_b), dim=1)

        elif self.cross_modal:
            # FULL: cross-modal transformer on (sim, hgt)
            dr_seq = torch.stack((dr_a, dr_b), dim=1)   # (D, 2, dim)
            di_seq = torch.stack((di_a, di_b), dim=1)
            dr = self.drug_trans(dr_seq).reshape(D, 2 * dim)
            di = self.disease_trans(di_seq).reshape(Di, 2 * dim)

        else:
            # Non-cross-modal transformer
            if self.use_sim and not self.use_gcn:
                # sim_transformer: self-attention on sim (stack sim,sim)
                dr_seq = torch.stack((dr_sim, dr_sim), dim=1)
                di_seq = torch.stack((di_sim, di_sim), dim=1)
                dr = self.drug_trans(dr_seq).reshape(D, 2 * dim)
                di = self.disease_trans(di_seq).reshape(Di, 2 * dim)

            elif not self.use_sim and self.use_gcn:
                # gcn_transformer: self-attention on hgt (stack hgt,hgt)
                dr_seq = torch.stack((dr_hgt, dr_hgt), dim=1)
                di_seq = torch.stack((di_hgt, di_hgt), dim=1)
                dr = self.drug_trans(dr_seq).reshape(D, 2 * dim)
                di = self.disease_trans(di_seq).reshape(Di, 2 * dim)

            else:
                # sim_transformer_gcn: refine sim independently via transformer,
                # then concatenate with gcn features (no cross-modal attention)
                dr_sim_seq = torch.stack((dr_sim, dr_sim), dim=1)   # (D, 2, dim)
                di_sim_seq = torch.stack((di_sim, di_sim), dim=1)
                dr_sim_refined = self.drug_trans(dr_sim_seq)[:, 0, :]    # (D, dim)
                di_sim_refined = self.disease_trans(di_sim_seq)[:, 0, :] # (Di, dim)
                dr = torch.cat((dr_sim_refined, dr_hgt), dim=1)   # (D, 2*dim)
                di = torch.cat((di_sim_refined, di_hgt), dim=1)

        # ── 5. Prediction ─────────────────────────────────────────────
        drdi_emb = torch.mul(dr[sample[:, 0]], di[sample[:, 1]])
        output   = self.mlp(drdi_emb)

        return dr, output
