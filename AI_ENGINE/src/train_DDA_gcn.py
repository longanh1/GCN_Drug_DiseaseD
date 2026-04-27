"""
train_DDA_gcn.py — Chạy model AMNTDDA_GCN (Có GCN + HGT, không có Fuzzy Logic)
và lưu kết quả vào AI_ENGINE/data/results/ để hiển thị trên web.

Thứ tự huấn luyện (3 bước):
  1. train_DDA_base.py  → AMNTDDA          (không GCN, không Fuzzy)
  2. train_DDA_gcn.py   → AMNTDDA_GCN      (có GCN,   không Fuzzy)  ← script này
  3. train_DDA_fuzzy.py → AMNTDDA_Fuzzy    (có GCN,   có Fuzzy)

Usage:
    python train_DDA_gcn.py --dataset C-dataset
"""

import sys
import os
import timeit
import argparse
import json
import numpy as np
import pandas as pd

# ── Compatibility patches ─────────────────────────────────────────────
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

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# ── Path setup ────────────────────────────────────────────────────────
THIS_DIR     = os.path.dirname(os.path.abspath(__file__))   # AI_ENGINE/src/
AIENGINE_DIR = os.path.dirname(THIS_DIR)                    # AI_ENGINE/
AMDGT_DIR    = os.path.abspath(os.path.join(AIENGINE_DIR, '..', 'AMDGT_main'))
DATA_OUT_DIR = os.path.join(AIENGINE_DIR, 'data')
RESULTS_DIR  = os.path.join(DATA_OUT_DIR, 'results')

sys.path.insert(0, AMDGT_DIR)
sys.path.insert(0, AIENGINE_DIR)
sys.path.insert(0, THIS_DIR)

from data_preprocess import get_data, data_processing, k_fold, dgl_similarity_graph, dgl_heterograph
from model.AMNTDDA_GCN import AMNTDDA_GCN
from metric import get_metric

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'AMNTDDA_GCN'


# ── Helpers ───────────────────────────────────────────────────────────
def _ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def save_fold_result(dataset: str, fold: int, fold_metrics: dict):
    path = os.path.join(RESULTS_DIR, f'{dataset}_{MODEL_NAME}_fold_results.csv')
    row  = {'fold': fold, **fold_metrics}
    df   = pd.DataFrame([row])
    df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)
    print(f"[fold {fold}] saved → {path}")


def compute_and_save_summary(dataset: str) -> dict:
    path = os.path.join(RESULTS_DIR, f'{dataset}_{MODEL_NAME}_fold_results.csv')
    df   = pd.read_csv(path)
    metrics = ['AUC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC']
    summary = {'dataset': dataset, 'model': MODEL_NAME, 'n_folds': len(df)}
    for m in metrics:
        summary[f'{m}_mean'] = round(float(df[m].mean()), 6)
        summary[f'{m}_std']  = round(float(df[m].std()),  6)

    out = os.path.join(RESULTS_DIR, f'{dataset}_{MODEL_NAME}_summary.json')
    with open(out, 'w') as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n{'='*60}")
    print(f"Summary  [{MODEL_NAME}]  on  {dataset}")
    print(f"{'='*60}")
    for m in metrics:
        print(f"  {m}: {summary[f'{m}_mean']:.4f} ± {summary[f'{m}_std']:.4f}")

    # Ghi Mean/Std vào cuối CSV
    all_metrics = ['AUC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC']
    mean_row = {'fold': 'Mean', **{m: round(float(df[m].mean()), 7) for m in all_metrics}}
    std_row  = {'fold': 'Std',  **{m: round(float(df[m].std()),  7) for m in all_metrics}}
    pd.DataFrame([mean_row, std_row]).to_csv(path, mode='a', header=False, index=False)
    return summary


def update_comparison(dataset: str):
    """Gộp summary AMNTDDA, AMNTDDA_GCN, AMNTDDA_Fuzzy vào comparison.json."""
    comparison = {'dataset': dataset}
    for model_name in ('AMNTDDA', MODEL_NAME, 'AMNTDDA_Fuzzy'):
        p = os.path.join(RESULTS_DIR, f'{dataset}_{model_name}_summary.json')
        if os.path.exists(p):
            with open(p) as fh:
                comparison[model_name] = json.load(fh)

    out = os.path.join(RESULTS_DIR, f'{dataset}_comparison.json')
    with open(out, 'w') as fh:
        json.dump(comparison, fh, indent=2)
    print(f"\nComparison updated → {out}")


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_fold',        type=int,   default=10)
    parser.add_argument('--epochs',        type=int,   default=1000)
    parser.add_argument('--lr',            type=float, default=1e-4)
    parser.add_argument('--weight_decay',  type=float, default=1e-3)
    parser.add_argument('--random_seed',   type=int,   default=1234)
    parser.add_argument('--neighbor',      type=int,   default=20)
    parser.add_argument('--negative_rate', type=float, default=1.0)
    parser.add_argument('--dataset',       default='C-dataset')
    parser.add_argument('--dropout',       type=float, default=0.2)
    parser.add_argument('--gt_layer',      type=int,   default=2)
    parser.add_argument('--gt_head',       type=int,   default=2)
    parser.add_argument('--gt_out_dim',    type=int,   default=200)
    parser.add_argument('--hgt_layer',     type=int,   default=2)
    parser.add_argument('--hgt_head',      type=int,   default=8)
    parser.add_argument('--hgt_in_dim',    type=int,   default=64)
    parser.add_argument('--hgt_head_dim',  type=int,   default=25)
    parser.add_argument('--hgt_out_dim',   type=int,   default=200)
    parser.add_argument('--tr_layer',      type=int,   default=2)
    parser.add_argument('--tr_head',       type=int,   default=4)
    args = parser.parse_args()

    args.data_dir   = os.path.join(AMDGT_DIR, 'data', args.dataset) + os.sep
    args.result_dir = os.path.join(AMDGT_DIR, 'Result', args.dataset, 'AMNTDDA_GCN') + os.sep

    _ensure_dirs(RESULTS_DIR, args.result_dir)
    for fi in range(args.k_fold):
        os.makedirs(os.path.join(args.data_dir, 'fold', str(fi)), exist_ok=True)

    # Xóa CSV cũ
    old_csv = os.path.join(RESULTS_DIR, f'{args.dataset}_{MODEL_NAME}_fold_results.csv')
    if os.path.exists(old_csv):
        os.remove(old_csv)

    print(f"Dataset: {args.dataset}")
    print(f"Model:   {MODEL_NAME} (GCN + HGT, không có Fuzzy Logic)")
    print(f"Device:  {device}")

    data = get_data(args)
    args.drug_number    = data['drug_number']
    args.disease_number = data['disease_number']
    args.protein_number = data['protein_number']

    data = data_processing(data, args)
    data = k_fold(data, args)

    drdr_graph, didi_graph, data = dgl_similarity_graph(data, args)
    drdr_graph = drdr_graph.to(device)
    didi_graph = didi_graph.to(device)

    drug_feature    = torch.FloatTensor(data['drugfeature']).to(device)
    disease_feature = torch.FloatTensor(data['diseasefeature']).to(device)
    protein_feature = torch.FloatTensor(data['proteinfeature']).to(device)

    cross_entropy = nn.CrossEntropyLoss()
    start = timeit.default_timer()

    header = ('Epoch\t\tTime\t\tAUC\t\tAUPR\t\t'
              'Accuracy\t\tPrecision\t\tRecall\t\tF1-score\t\tMcc')

    for i in range(args.k_fold):
        print(f'\nfold: {i}')
        print(header)

        model = AMNTDDA_GCN(args).to(device)
        optimizer = optim.Adam(model.parameters(),
                               weight_decay=args.weight_decay, lr=args.lr)

        best = dict(auc=0, aupr=0, accuracy=0, precision=0, recall=0, f1=0, mcc=0)

        X_train = torch.LongTensor(data['X_train'][i]).to(device)
        Y_train = torch.LongTensor(data['Y_train'][i]).to(device)
        X_test  = torch.LongTensor(data['X_test'][i]).to(device)
        Y_test  = data['Y_test'][i].flatten()

        drdipr_graph, data = dgl_heterograph(data, data['X_train'][i], args)
        drdipr_graph = drdipr_graph.to(device)

        for epoch in range(args.epochs):
            model.train()
            _, train_score = model(drdr_graph, didi_graph, drdipr_graph,
                                   drug_feature, disease_feature, protein_feature,
                                   X_train)
            train_loss = cross_entropy(train_score, torch.flatten(Y_train))
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                _, test_score = model(drdr_graph, didi_graph, drdipr_graph,
                                      drug_feature, disease_feature, protein_feature,
                                      X_test)

            test_prob = F.softmax(test_score, dim=-1)[:, 1].cpu().numpy()
            test_pred = torch.argmax(test_score, dim=-1).cpu().numpy()

            auc, aupr, acc, prec, rec, f1, mcc = get_metric(Y_test, test_pred, test_prob)

            elapsed = round(timeit.default_timer() - start, 2)
            row = [epoch+1, elapsed,
                   round(auc,5), round(aupr,5), round(acc,5),
                   round(prec,5), round(rec,5), round(f1,5), round(mcc,5)]
            print('\t\t'.join(map(str, row)))

            if auc > best['auc']:
                best = dict(auc=auc, aupr=aupr, accuracy=acc,
                            precision=prec, recall=rec, f1=f1, mcc=mcc)
                print(f'  ↑ AUC improved at epoch {epoch+1}: {auc:.5f}')

        save_fold_result(args.dataset, i, {
            'AUC':       round(best['auc'],       6),
            'AUPR':      round(best['aupr'],      6),
            'Accuracy':  round(best['accuracy'],  6),
            'Precision': round(best['precision'], 6),
            'Recall':    round(best['recall'],    6),
            'F1':        round(best['f1'],        6),
            'MCC':       round(best['mcc'],       6),
        })

    compute_and_save_summary(args.dataset)
    update_comparison(args.dataset)
    print("\nDone. Kết quả đã lưu vào AI_ENGINE/data/results/")
