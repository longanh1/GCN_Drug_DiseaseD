"""
AMNTDDA_Fuzzy.py — AMNTDDA backbone + Mamdani FIS interpretability layer.

The backbone is architecturally identical to base AMNTDDA.
The "Fuzzy" upgrade is the Mamdani FIS layer used at inference time
(see /predict/fuzzy_detail API endpoint) to explain each prediction
through fuzzy membership functions and IF-THEN rules.

Result: comparable or better AUC (fair comparison with same random seed),
plus rich explainability that base AMNTDDA does not provide.
"""

import dgl
import dgl.nn.pytorch
import torch
import torch.nn as nn
from model import gt_net_drug, gt_net_disease

device_global = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AMNTDDA_Fuzzy(nn.Module):
    """AMNTDDA backbone — identical architecture to base model for fair AUC comparison."""

    def __init__(self, args):
        super(AMNTDDA_Fuzzy, self).__init__()
        self.args = args
        device = device_global

        self.drug_linear    = nn.Linear(300, args.hgt_in_dim)
        self.protein_linear = nn.Linear(320, args.hgt_in_dim)
        # disease_feature passed raw (no projection) — same as base AMNTDDA

        self.gt_drug = gt_net_drug.GraphTransformer(
            device, args.gt_layer, args.drug_number,
            args.gt_out_dim, args.gt_out_dim, args.gt_head, args.dropout)
        self.gt_disease = gt_net_disease.GraphTransformer(
            device, args.gt_layer, args.disease_number,
            args.gt_out_dim, args.gt_out_dim, args.gt_head, args.dropout)

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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.gt_out_dim, nhead=args.tr_head)
        self.drug_trans    = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)
        self.disease_trans = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)

        self.drug_tr = nn.Transformer(
            d_model=args.gt_out_dim, nhead=args.tr_head,
            num_encoder_layers=3, num_decoder_layers=3, batch_first=True)
        self.disease_tr = nn.Transformer(
            d_model=args.gt_out_dim, nhead=args.tr_head,
            num_encoder_layers=3, num_decoder_layers=3, batch_first=True)

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
            nn.Linear(256, 2)
        )

    def forward(self, drdr_graph, didi_graph, drdipr_graph,
                drug_feature, disease_feature, protein_feature,
                sample, drug_topo=None, disease_topo=None):
        dr_sim = self.gt_drug(drdr_graph)
        di_sim = self.gt_disease(didi_graph)

        drug_feature_proj    = self.drug_linear(drug_feature)
        protein_feature_proj = self.protein_linear(protein_feature)

        feature_dict = {
            'drug':    drug_feature_proj,
            'disease': disease_feature,
            'protein': protein_feature_proj
        }
        drdipr_graph.ndata['h'] = feature_dict
        g = dgl.to_homogeneous(drdipr_graph, ndata='h')
        feature = torch.cat((drug_feature_proj, disease_feature, protein_feature_proj), dim=0)

        for layer in self.hgt:
            hgt_out = layer(g, feature, g.ndata['_TYPE'], g.edata['_TYPE'], presorted=True)
            feature = hgt_out

        dr_hgt = hgt_out[:self.args.drug_number, :]
        di_hgt = hgt_out[self.args.drug_number:self.args.drug_number + self.args.disease_number, :]

        dr = torch.stack((dr_sim, dr_hgt), dim=1)
        di = torch.stack((di_sim, di_hgt), dim=1)

        dr = self.drug_trans(dr)
        di = self.disease_trans(di)

        dr = dr.view(self.args.drug_number, 2 * self.args.gt_out_dim)
        di = di.view(self.args.disease_number, 2 * self.args.gt_out_dim)

        drdi_embedding = torch.mul(dr[sample[:, 0]], di[sample[:, 1]])
        output = self.mlp(drdi_embedding)

        return dr, output
