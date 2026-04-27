"""
AMNTDDA_GCN.py — AMNTDDA backbone + GCN (GraphConv) encoder.

Thêm một lớp GCN (Graph Convolutional Network) trước HGT để học
biểu diễn nút trên đồ thị thuần nhất (homogeneous view) trước khi
HGT học quan hệ không đồng nhất (heterogeneous view).

Kiến trúc:
  Drug/Disease Sim. Matrices → GraphTransformer → HR_s, HD_s
  Drug-Disease-Protein Graph  → GCN → Embedding ban đầu
                              → HGT → HR_N, HD_N
  Combine (HR_s + HR_N) via TransformerEncoder → H_r
  Combine (HD_s + HD_N) via TransformerEncoder → H_d
  (H_r, H_d) → Dot Product + MLP → Prediction
"""

import dgl
import dgl.nn.pytorch
import torch
import torch.nn as nn
from model import gt_net_drug, gt_net_disease

device_global = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AMNTDDA_GCN(nn.Module):
    """AMNTDDA backbone + GCN pre-encoder trước HGT."""

    def __init__(self, args):
        super(AMNTDDA_GCN, self).__init__()
        self.args = args
        device = device_global

        # Projection layers
        self.drug_linear    = nn.Linear(300, args.hgt_in_dim)
        self.protein_linear = nn.Linear(320, args.hgt_in_dim)

        # ── Similarity Feature Extraction (Graph Transformer) ──────────
        self.gt_drug = gt_net_drug.GraphTransformer(
            device, args.gt_layer, args.drug_number,
            args.gt_out_dim, args.gt_out_dim, args.gt_head, args.dropout)
        self.gt_disease = gt_net_disease.GraphTransformer(
            device, args.gt_layer, args.disease_number,
            args.gt_out_dim, args.gt_out_dim, args.gt_head, args.dropout)

        # ── GCN encoder (trên đồ thị thuần nhất) ─────────────────────
        gcn_hidden = args.hgt_in_dim
        self.gcn_layer1 = dgl.nn.pytorch.GraphConv(
            args.hgt_in_dim, gcn_hidden, activation=torch.relu, allow_zero_in_degree=True)
        self.gcn_layer2 = dgl.nn.pytorch.GraphConv(
            gcn_hidden, args.hgt_in_dim, activation=None, allow_zero_in_degree=True)
        self.gcn_norm   = nn.LayerNorm(args.hgt_in_dim)

        # ── HGT encoder (Heterogeneous Graph Transformer) ─────────────
        self.hgt_dgl = dgl.nn.pytorch.conv.HGTConv(
            args.hgt_in_dim, int(args.hgt_in_dim / args.hgt_head),
            args.hgt_head, 3, 3, args.dropout)
        self.hgt_dgl_last = dgl.nn.pytorch.conv.HGTConv(
            args.hgt_in_dim, args.hgt_head_dim,
            args.hgt_head, 3, 3, args.dropout)

        self.hgt = nn.ModuleList()
        for _ in range(args.hgt_layer - 1):
            self.hgt.append(self.hgt_dgl)
        self.hgt.append(self.hgt_dgl_last)

        # ── Modality Interaction (Transformer Encoder) ─────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.gt_out_dim, nhead=args.tr_head, batch_first=False)
        self.drug_trans    = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)
        self.disease_trans = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)

        self.drug_tr = nn.Transformer(
            d_model=args.gt_out_dim, nhead=args.tr_head,
            num_encoder_layers=3, num_decoder_layers=3, batch_first=True)
        self.disease_tr = nn.Transformer(
            d_model=args.gt_out_dim, nhead=args.tr_head,
            num_encoder_layers=3, num_decoder_layers=3, batch_first=True)

        # ── Prediction MLP ─────────────────────────────────────────────
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

    def forward(self, drdr_graph, didi_graph, drdipr_graph,
                drug_feature, disease_feature, protein_feature,
                sample, drug_topo=None, disease_topo=None):
        # ── Stage 2a: Similarity Feature Extraction ────────────────────
        dr_sim = self.gt_drug(drdr_graph)    # (n_drug, gt_out_dim)
        di_sim = self.gt_disease(didi_graph) # (n_dis,  gt_out_dim)

        # Projection
        drug_feat_proj    = self.drug_linear(drug_feature)    # (n_drug, hgt_in_dim)
        protein_feat_proj = self.protein_linear(protein_feature)  # (n_prot, hgt_in_dim)

        # ── Stage 2b: GCN pre-encoder ──────────────────────────────────
        feature_dict = {
            'drug':    drug_feat_proj,
            'disease': disease_feature,
            'protein': protein_feat_proj,
        }
        drdipr_graph.ndata['h'] = feature_dict
        g_homo = dgl.to_homogeneous(drdipr_graph, ndata='h')
        feat_homo = torch.cat(
            (drug_feat_proj, disease_feature, protein_feat_proj), dim=0)

        gcn_out = self.gcn_layer1(g_homo, feat_homo)
        gcn_out = self.gcn_layer2(g_homo, gcn_out)
        gcn_out = self.gcn_norm(gcn_out + feat_homo)  # residual

        # ── Stage 2b cont.: HGT encoder ───────────────────────────────
        g_homo2 = dgl.to_homogeneous(drdipr_graph, ndata='h')
        for layer in self.hgt:
            hgt_out = layer(
                g_homo2, gcn_out,
                g_homo2.ndata['_TYPE'], g_homo2.edata['_TYPE'],
                presorted=True)
            gcn_out = hgt_out   # pass GCN-refined features through HGT

        dr_hgt = hgt_out[:self.args.drug_number, :]
        di_hgt = hgt_out[self.args.drug_number:
                         self.args.drug_number + self.args.disease_number, :]

        # ── Stage 3: Modality Interaction ─────────────────────────────
        dr = torch.stack((dr_sim, dr_hgt), dim=1)
        di = torch.stack((di_sim, di_hgt), dim=1)

        dr = self.drug_trans(dr)
        di = self.disease_trans(di)

        dr = dr.view(self.args.drug_number,    2 * self.args.gt_out_dim)
        di = di.view(self.args.disease_number, 2 * self.args.gt_out_dim)

        # ── Stage 4: Prediction ────────────────────────────────────────
        drug_embed    = dr[sample[:, 0]]
        disease_embed = di[sample[:, 1]]

        pair = torch.cat((drug_embed, disease_embed), dim=-1)
        score = self.mlp(pair)

        return dr, score
