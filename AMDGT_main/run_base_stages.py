"""
run_base_stages.py â€” Cháº¡y tá»«ng giai Ä‘oáº¡n cá»§a mÃ´ hÃ¬nh AMNTDDA gá»‘c (khÃ´ng GCN, khÃ´ng Fuzzy)
vÃ  lÆ°u káº¿t quáº£ trung gian vÃ o AMDGT_main/Run_Base/stage*/dataset/result.json.

Má»—i láº§n cháº¡y sáº½ xá»­ lÃ½ 1 dataset táº¡i 1 thá»i Ä‘iá»ƒm (Ä‘Æ°á»£c truyá»n qua --dataset).
Káº¿t quáº£ Ä‘Æ°á»£c dÃ¹ng Ä‘á»ƒ so sÃ¡nh vá»›i mÃ´ hÃ¬nh nÃ¢ng cáº¥p trong tÆ°Æ¡ng lai â€” KHÃ”NG sá»­a Ä‘á»•i.

CÃ¡ch dÃ¹ng:
    python run_base_stages.py --dataset B-dataset
    python run_base_stages.py --dataset C-dataset --epochs 5
"""

import argparse
import os
import sys
import json
import time

# Fix Unicode output on Windows terminals (cp1252)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
import dgl

# â”€â”€ Compatibility patches â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import networkx as nx
if not hasattr(nx, 'from_numpy_matrix'):
    nx.from_numpy_matrix = nx.from_numpy_array
if not hasattr(nx, 'to_numpy_matrix'):
    nx.to_numpy_matrix = nx.to_numpy_array

import dgl.function as _dgl_fn
if not hasattr(_dgl_fn, 'src_mul_edge'):
    _dgl_fn.src_mul_edge = _dgl_fn.u_mul_e
if not hasattr(_dgl_fn, 'copy_edge'):
    _dgl_fn.copy_edge = _dgl_fn.copy_e

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_preprocess import get_data, data_processing, k_fold, dgl_similarity_graph, dgl_heterograph
from model.AMNTDDA import AMNTDDA
from metric import get_metric

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_BASE = os.path.join(THIS_DIR, 'Run_Base')

STAGE_FOLDERS = [
    'stage1_input_layer',
    'stage2_feature_extraction',
    'stage3_modality_interaction',
    'stage4_prediction',
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _save(stage_folder: str, dataset: str, payload: dict):
    out_dir = os.path.join(RUN_BASE, stage_folder, dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'result.json')
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f'  âœ“ ÄÃ£ lÆ°u: {out_path}')
    return out_path


def _matrix_stats(m: np.ndarray, name: str) -> dict:
    """Thá»‘ng kÃª cÆ¡ báº£n cá»§a má»™t ma tráº­n."""
    return {
        'name':   name,
        'shape':  list(m.shape),
        'min':    float(np.min(m)),
        'max':    float(np.max(m)),
        'mean':   float(np.mean(m)),
        'std':    float(np.std(m)),
        'sparsity': float(np.sum(m == 0) / m.size),
        'sample_5x5': m[:5, :5].tolist(),
    }


def _tensor_stats(t: torch.Tensor, name: str) -> dict:
    with torch.no_grad():
        arr = t.detach().cpu().numpy()
    return {
        'name':  name,
        'shape': list(arr.shape),
        'min':   float(arr.min()),
        'max':   float(arr.max()),
        'mean':  float(arr.mean()),
        'std':   float(arr.std()),
        'norm':  float(np.linalg.norm(arr)),
    }


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# STAGE 1 â€” Input Layer: Similarity Networks + Heterogeneous Network
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def run_stage1(data: dict, args) -> dict:
    print('\n[Stage 1] XÃ¢y dá»±ng máº¡ng lÆ°á»›i (Input Layer) ...')
    t0 = time.time()

    # Ma tráº­n Ä‘áº·c trÆ°ng
    drf = data['drf']   # Drug fingerprint
    drg = data['drg']   # Drug GIP
    dip = data['dip']   # Disease phenotypic similarity
    dig = data['dig']   # Disease GIP

    # Äáº·c trÆ°ng Ä‘áº§u vÃ o
    drug_feat    = data['drugfeature']      # Drug mol2vec (N_d Ã— 300)
    disease_feat = data['diseasefeature']   # Disease features (N_i Ã— D_i)
    protein_feat = data['proteinfeature']   # Protein ESM  (N_p Ã— 320)

    # LiÃªn káº¿t
    drdi = data['drdi']   # Drug-Disease associations
    drpr = data['drpr']   # Drug-Protein associations
    dipr = data['dipr']   # Disease-Protein associations

    # Sá»‘ lÆ°á»£ng nodes
    num_drugs    = int(data['drug_number'])
    num_diseases = int(data['disease_number'])
    num_proteins = int(data['protein_number'])

    # Thá»‘ng kÃª máº¡ng liÃªn káº¿t
    drdr_adj = drg  # dÃ¹ng GIP lÃ m drugâ€“drug similarity
    didi_adj = dig  # dÃ¹ng GIP lÃ m diseaseâ€“disease similarity

    # Sparsity cá»§a máº¡ng dá»‹ thá»ƒ
    total_edges = len(drdi) + len(drpr) + len(dipr)
    max_edges   = (num_drugs * num_diseases
                   + num_drugs * num_proteins
                   + num_diseases * num_proteins)
    hetero_sparsity = 1.0 - total_edges / max(max_edges, 1)

    # t-SNE 2D of input features (better visual separation than PCA)
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    def _tsne2(arr):
        n = min(200, arr.shape[0])
        perp = max(5, min(30, n // 3))
        return TSNE(n_components=2, perplexity=perp, random_state=42,
                    max_iter=500, init='pca').fit_transform(arr[:n]).tolist()

    # PCA 3D
    pca3 = PCA(n_components=3, random_state=42)
    drug_pca3    = pca3.fit_transform(drug_feat[:min(200, num_drugs)])
    disease_pca3 = pca3.fit_transform(disease_feat[:min(200, num_diseases)])

    result = {
        'stage': 'stage1_input_layer',
        'dataset': args.dataset,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_sec': round(time.time() - t0, 2),
        'network_stats': {
            'num_drugs':    num_drugs,
            'num_diseases': num_diseases,
            'num_proteins': num_proteins,
            'drug_disease_links':   int(len(drdi)),
            'drug_protein_links':   int(len(drpr)),
            'disease_protein_links': int(len(dipr)),
            'total_hetero_edges':   int(total_edges),
            'hetero_sparsity':      round(hetero_sparsity, 4),
        },
        'feature_dims': {
            'drug_input_dim':    int(drug_feat.shape[1]),
            'disease_input_dim': int(disease_feat.shape[1]),
            'protein_input_dim': int(protein_feat.shape[1]),
        },
        'similarity_matrices': [
            _matrix_stats(drg, 'DrugGIP'),
            _matrix_stats(dig, 'DiseaseGIP'),
            _matrix_stats(drf[:50, :50] if drf.shape[0] > 50 else drf, 'DrugFingerprint (50x50)'),
            _matrix_stats(dip[:50, :50] if dip.shape[0] > 50 else dip, 'DiseasePS (50x50)'),
        ],
        'feature_stats': [
            _matrix_stats(drug_feat, 'Drug_mol2vec'),
            _matrix_stats(disease_feat, 'DiseaseFeature'),
            _matrix_stats(protein_feat[:200], 'Protein_ESM (200 samples)'),
        ],
        'pca2d': {
            'drug':    {'points': _tsne2(drug_feat),    'method': 'tsne'},
            'disease': {'points': _tsne2(disease_feat), 'method': 'tsne'},
        },
        'pca3d': {
            'drug':    {'points': drug_pca3.tolist()},
            'disease': {'points': disease_pca3.tolist()},
        },
        'degree_distribution': {
            'drug_degrees':    sorted([sum(1 for (dr, _) in drdi.tolist() if dr == i) for i in range(num_drugs)])[:50],
            'disease_degrees': sorted([sum(1 for (_, di) in drdi.tolist() if di == i) for i in range(num_diseases)])[:50],
        },
    }
    return result


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# STAGE 2 â€” Feature Extraction: Graph Transformer + HGT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def run_stage2(data: dict, args, drdr_graph, didi_graph, drdipr_graph,
               drug_feat_t, disease_feat_t, protein_feat_t,
               model: AMNTDDA) -> dict:
    print('\n[Stage 2] Trich xuat dac trung (Feature Extraction) ...')
    t0 = time.time()
    model.eval()

    with torch.no_grad():
        # â”€â”€ Graph Transformer (reads ndata['drs'/'dis'] set by dgl_similarity_graph)
        gt_drug_out    = model.gt_drug(drdr_graph)    # (N_d, gt_out_dim)
        gt_disease_out = model.gt_disease(didi_graph) # (N_i, gt_out_dim)

        # â”€â”€ HGT: replicate model.forward() exactly â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        drug_lin    = model.drug_linear(drug_feat_t)       # (N_d, hgt_in_dim)
        protein_lin = model.protein_linear(protein_feat_t) # (N_p, hgt_in_dim)
        # disease_feat_t is already (N_i, hgt_in_dim=64)
        feature_dict = {'drug': drug_lin, 'disease': disease_feat_t, 'protein': protein_lin}
        drdipr_graph.ndata['h'] = feature_dict
        g_homo  = dgl.to_homogeneous(drdipr_graph, ndata='h')
        hgt_feat = torch.cat((drug_lin, disease_feat_t, protein_lin), dim=0)
        for hgt_layer in model.hgt:
            hgt_feat = hgt_layer(g_homo, hgt_feat,
                                 g_homo.ndata['_TYPE'], g_homo.edata['_TYPE'],
                                 presorted=True)
        hgt_drug    = hgt_feat[:args.drug_number, :]
        hgt_disease = hgt_feat[args.drug_number:args.drug_number + args.disease_number, :]

    # â”€â”€ Projections for visualization â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    gt_drug_np  = gt_drug_out.cpu().numpy()
    gt_dis_np   = gt_disease_out.cpu().numpy()
    hgt_drug_np = hgt_drug.cpu().numpy()
    hgt_dis_np  = hgt_disease.cpu().numpy()

    def _tsne2(arr):
        n = min(200, arr.shape[0])
        perp = max(5, min(30, n // 3))
        return TSNE(n_components=2, perplexity=perp, random_state=42,
                    max_iter=500, init='pca').fit_transform(arr[:n]).tolist()

    def _pca3(arr):
        n = min(200, arr.shape[0])
        return PCA(n_components=3, random_state=42).fit_transform(arr[:n]).tolist()

    # Cosine similarity heatmap 50Ã—50
    n = min(50, gt_drug_np.shape[0])
    norm = np.linalg.norm(gt_drug_np[:n], axis=1, keepdims=True) + 1e-8
    attn_approx = (gt_drug_np[:n] / norm @ (gt_drug_np[:n] / norm).T).tolist()

    result = {
        'stage': 'stage2_feature_extraction',
        'dataset': args.dataset,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_sec': round(time.time() - t0, 2),
        'graph_transformer_stats': {
            'drug':    _tensor_stats(gt_drug_out, 'GT_Drug'),
            'disease': _tensor_stats(gt_disease_out, 'GT_Disease'),
        },
        'hgt_stats': {
            'drug':    _tensor_stats(hgt_drug, 'HGT_Drug'),
            'disease': _tensor_stats(hgt_disease, 'HGT_Disease'),
        },
        'pca2d': {
            'gt_drug':    {'points': _tsne2(gt_drug_np),  'method': 'tsne'},
            'gt_disease': {'points': _tsne2(gt_dis_np),   'method': 'tsne'},
            'hgt_drug':   {'points': _tsne2(hgt_drug_np), 'method': 'tsne'},
            'hgt_disease':{'points': _tsne2(hgt_dis_np),  'method': 'tsne'},
        },
        'pca3d': {
            'gt_drug':    {'points': _pca3(gt_drug_np)},
            'hgt_drug':   {'points': _pca3(hgt_drug_np)},
        },
        'attention_approx_50x50': attn_approx,
        'output_dims': {
            'gt_drug_dim':    int(gt_drug_out.shape[1]),
            'gt_disease_dim': int(gt_disease_out.shape[1]),
            'hgt_drug_dim':   int(hgt_drug.shape[1]),
            'hgt_disease_dim':int(hgt_disease.shape[1]),
        },
    }
    return result


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# STAGE 3 â€” Modality Interaction Module (Transformer cross-attention)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def run_stage3(data: dict, args, drdr_graph, didi_graph, drdipr_graph,
               drug_feat_t, disease_feat_t, protein_feat_t,
               model: AMNTDDA, sample_pairs: list) -> dict:
    print('\n[Stage 3] Tuong tac da phuong thuc (Modality Interaction) ...')
    t0 = time.time()
    model.eval()

    with torch.no_grad():
        # Replicate model.forward() â€” GT + HGT
        gt_drug_out    = model.gt_drug(drdr_graph)
        gt_disease_out = model.gt_disease(didi_graph)
        drug_lin    = model.drug_linear(drug_feat_t)
        protein_lin = model.protein_linear(protein_feat_t)
        feature_dict = {'drug': drug_lin, 'disease': disease_feat_t, 'protein': protein_lin}
        drdipr_graph.ndata['h'] = feature_dict
        g_homo   = dgl.to_homogeneous(drdipr_graph, ndata='h')
        hgt_feat = torch.cat((drug_lin, disease_feat_t, protein_lin), dim=0)
        for hgt_layer in model.hgt:
            hgt_feat = hgt_layer(g_homo, hgt_feat,
                                 g_homo.ndata['_TYPE'], g_homo.edata['_TYPE'],
                                 presorted=True)
        hgt_drug    = hgt_feat[:args.drug_number, :]
        hgt_disease = hgt_feat[args.drug_number:args.drug_number + args.disease_number, :]

        # Stack + TransformerEncoder (match model.forward exactly)
        dr = torch.stack((gt_drug_out, hgt_drug), dim=1)       # (N_d, 2, gt_out_dim)
        di = torch.stack((gt_disease_out, hgt_disease), dim=1)  # (N_i, 2, gt_out_dim)
        dr_enc = model.drug_trans(dr)    # (N_d, 2, gt_out_dim)
        di_enc = model.disease_trans(di) # (N_i, 2, gt_out_dim)
        dr_flat = dr_enc.reshape(args.drug_number,    2 * args.gt_out_dim)  # final drug emb
        di_flat = di_enc.reshape(args.disease_number, 2 * args.gt_out_dim)  # final disease emb

        # Cross-modal pair interaction
        pair_scores = []
        for dr_idx, di_idx in sample_pairs[:20]:
            d_emb = dr_enc[dr_idx].unsqueeze(0)  # (1, 2, gt_out_dim)
            i_emb = di_enc[di_idx].unsqueeze(0)
            cross_dr = model.drug_tr(d_emb, i_emb).squeeze(0)
            cross_di = model.disease_tr(i_emb, d_emb).squeeze(0)
            score = float(torch.mul(dr_flat[dr_idx], di_flat[di_idx]).sum().item())
            pair_scores.append({
                'drug_idx': int(dr_idx), 'disease_idx': int(di_idx),
                'score': round(score, 4),
                'cross_dr_norm': round(float(cross_dr.norm().item()), 4),
                'cross_di_norm': round(float(cross_di.norm().item()), 4),
            })

    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    dr_np = dr_flat.cpu().numpy()
    di_np = di_flat.cpu().numpy()

    def _tsne2(arr):
        n = min(200, arr.shape[0])
        perp = max(5, min(30, n // 3))
        return TSNE(n_components=2, perplexity=perp, random_state=42,
                    max_iter=500, init='pca').fit_transform(arr[:n]).tolist()

    def _pca3(arr):
        n = min(200, arr.shape[0])
        return PCA(n_components=3, random_state=42).fit_transform(arr[:n]).tolist()

    result = {
        'stage': 'stage3_modality_interaction',
        'dataset': args.dataset,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_sec': round(time.time() - t0, 2),
        'transformer_stats': {
            'drug_trans':    _tensor_stats(dr_flat, 'DrugFinal'),
            'disease_trans': _tensor_stats(di_flat, 'DiseaseFinal'),
        },
        'pca2d': {
            'drug':    {'points': _tsne2(dr_np), 'method': 'tsne'},
            'disease': {'points': _tsne2(di_np), 'method': 'tsne'},
        },
        'pca3d': {
            'drug':    {'points': _pca3(dr_np)},
            'disease': {'points': _pca3(di_np)},
        },
        'pair_interaction_samples': pair_scores,
        'output_dims': {
            'drug_final_dim':    int(dr_flat.shape[1]),
            'disease_final_dim': int(di_flat.shape[1]),
        },
    }
    return result


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# STAGE 4 â€” Prediction Module (1 fold, limited epochs)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def run_stage4(data: dict, args, drdr_graph, didi_graph,
               drug_feat_t, disease_feat_t, protein_feat_t) -> dict:
    print('\n[Stage 4] Dá»± Ä‘oÃ¡n & Kiá»ƒm chá»©ng (Prediction + Case Study) ...')
    t0 = time.time()
    cross_entropy = nn.CrossEntropyLoss()

    # Cháº¡y 1 fold duy nháº¥t vá»›i sá»‘ epoch giá»›i háº¡n
    i = 0
    model = AMNTDDA(args).to(device)
    optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

    X_train = torch.LongTensor(data['X_train'][i]).to(device)
    Y_train = torch.LongTensor(data['Y_train'][i]).to(device)
    X_test  = torch.LongTensor(data['X_test'][i]).to(device)
    Y_test  = data['Y_test'][i].flatten()

    drdipr_graph, data = dgl_heterograph(data, data['X_train'][i], args)
    drdipr_graph = drdipr_graph.to(device)

    fold_epoch_log = []
    best_auc  = 0.0
    best_metrics = {}
    all_test_probs = []

    for epoch in range(args.epochs):
        model.train()
        _, train_score = model(drdr_graph, didi_graph, drdipr_graph,
                               drug_feat_t, disease_feat_t, protein_feat_t, X_train)
        loss = cross_entropy(train_score, torch.flatten(Y_train))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            _, test_score = model(drdr_graph, didi_graph, drdipr_graph,
                                  drug_feat_t, disease_feat_t, protein_feat_t, X_test)

        test_prob  = fn.softmax(test_score, dim=-1)[:, 1].cpu().numpy()
        test_label = torch.argmax(test_score, dim=-1).cpu().numpy()
        AUC, AUPR, acc, prec, rec, f1, mcc = get_metric(Y_test, test_label, test_prob)

        fold_epoch_log.append({
            'epoch': epoch + 1,
            'loss':  round(float(loss.item()), 6),
            'AUC':   round(float(AUC), 5),
            'AUPR':  round(float(AUPR), 5),
        })
        if AUC > best_auc:
            best_auc = AUC
            best_metrics = {
                'AUC': round(float(AUC), 5), 'AUPR': round(float(AUPR), 5),
                'Accuracy': round(float(acc), 5), 'Precision': round(float(prec), 5),
                'Recall': round(float(rec), 5), 'F1': round(float(f1), 5),
                'MCC': round(float(mcc), 5),
            }
            all_test_probs = [
                {'drug_idx': int(X_test[k, 0].item()),
                 'disease_idx': int(X_test[k, 1].item()),
                 'prob': round(float(test_prob[k]), 4),
                 'label': int(Y_test[k]),
                 'pred':  int(test_label[k])}
                for k in range(min(100, len(test_prob)))
            ]

    # Top predictions (sorted by score, novel = non-known)
    top_preds = sorted(all_test_probs, key=lambda x: x['prob'], reverse=True)[:30]

    result = {
        'stage': 'stage4_prediction',
        'dataset': args.dataset,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'elapsed_sec': round(time.time() - t0, 2),
        'fold_evaluated': 0,
        'epochs_run': args.epochs,
        'best_metrics': best_metrics,
        'epoch_log': fold_epoch_log,
        'top_predictions': top_preds,
        'learning_curve': {
            'epochs': [e['epoch'] for e in fold_epoch_log],
            'auc':    [e['AUC']   for e in fold_epoch_log],
            'aupr':   [e['AUPR']  for e in fold_epoch_log],
            'loss':   [e['loss']  for e in fold_epoch_log],
        },
    }
    return result


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MAIN
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def main():
    parser = argparse.ArgumentParser(description='Run Base AMNTDDA stages')
    parser.add_argument('--dataset',      default='B-dataset')
    parser.add_argument('--k_fold',       type=int,   default=10)
    parser.add_argument('--epochs',       type=int,   default=10,
                        help='Number of epochs for stage4 training (small value for demo)')
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--random_seed',  type=int,   default=1234)
    parser.add_argument('--neighbor',     type=int,   default=20)
    parser.add_argument('--negative_rate',type=float, default=1.0)
    parser.add_argument('--dropout',      type=float, default=0.2)
    parser.add_argument('--gt_layer',     type=int,   default=2)
    parser.add_argument('--gt_head',      type=int,   default=2)
    parser.add_argument('--gt_out_dim',   type=int,   default=200)
    parser.add_argument('--hgt_layer',    type=int,   default=2)
    parser.add_argument('--hgt_head',     type=int,   default=8)
    parser.add_argument('--hgt_in_dim',   type=int,   default=64)
    parser.add_argument('--hgt_head_dim', type=int,   default=25)
    parser.add_argument('--hgt_out_dim',  type=int,   default=200)
    parser.add_argument('--tr_layer',     type=int,   default=2)
    parser.add_argument('--tr_head',      type=int,   default=4)
    parser.add_argument('--stages',       nargs='+',  default=['1', '2', '3', '4'],
                        help='Which stages to run, e.g. --stages 1 2')
    args = parser.parse_args()

    args.data_dir   = os.path.join(THIS_DIR, 'data', args.dataset) + os.sep
    args.result_dir = os.path.join(THIS_DIR, 'Result', args.dataset, 'AMNTDDA') + os.sep
    os.makedirs(args.result_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'  AMNTDDA Run_Base  |  Dataset: {args.dataset}')
    print(f'  Device: {device}  |  Stages: {args.stages}')
    print(f'{"="*60}')

    # â”€â”€ Load & process data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print('\n[Data] Loading data ...')
    # Create fold directories
    for fi in range(args.k_fold):
        os.makedirs(os.path.join(args.data_dir, 'fold', str(fi)), exist_ok=True)

    data = get_data(args)
    args.drug_number    = data['drug_number']
    args.disease_number = data['disease_number']
    args.protein_number = data['protein_number']

    # Stage 1 (no model needed)
    if '1' in args.stages:
        result1 = run_stage1(data, args)
        _save('stage1_input_layer', args.dataset, result1)

    # Process data for stages 2â€“4
    data = data_processing(data, args)
    data = k_fold(data, args)
    drdr_graph, didi_graph, data = dgl_similarity_graph(data, args)
    drdr_graph = drdr_graph.to(device)
    didi_graph = didi_graph.to(device)

    drug_feat_t    = torch.FloatTensor(data['drugfeature']).to(device)
    disease_feat_t = torch.FloatTensor(data['diseasefeature']).to(device)
    protein_feat_t = torch.FloatTensor(data['proteinfeature']).to(device)

    # Need drdipr_graph for stages 2â€“3: use fold 0
    drdipr_graph, data = dgl_heterograph(data, data['X_train'][0], args)
    drdipr_graph = drdipr_graph.to(device)

    # Create a randomly initialised (untrained) model for stages 2â€“3
    model = AMNTDDA(args).to(device)

    if '2' in args.stages:
        result2 = run_stage2(data, args, drdr_graph, didi_graph, drdipr_graph,
                              drug_feat_t, disease_feat_t, protein_feat_t, model)
        _save('stage2_feature_extraction', args.dataset, result2)

    if '3' in args.stages:
        sample_pairs = data['X_train'][0][:50].tolist()
        result3 = run_stage3(data, args, drdr_graph, didi_graph, drdipr_graph,
                              drug_feat_t, disease_feat_t, protein_feat_t,
                              model, sample_pairs)
        _save('stage3_modality_interaction', args.dataset, result3)

    if '4' in args.stages:
        result4 = run_stage4(data, args, drdr_graph, didi_graph,
                              drug_feat_t, disease_feat_t, protein_feat_t)
        _save('stage4_prediction', args.dataset, result4)

    print(f'\n{"="*60}')
    print(f'  HoÃ n thÃ nh! Káº¿t quáº£ lÆ°u táº¡i: {RUN_BASE}/{args.dataset}/')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()
