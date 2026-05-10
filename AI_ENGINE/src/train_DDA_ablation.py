"""
train_DDA_ablation.py — Ablation study: train all 7 pipeline variants.

Trains each variant with the same hyper-parameters and data splits,
then saves per-fold CSVs, per-variant summary JSONs, and a master
ablation comparison JSON.

Output structure (AI_ENGINE/data/results/):
  {dataset}_ablation_{variant}_fold_results.csv   — per-fold metrics
  {dataset}_ablation_{variant}_summary.json       — mean ± std
  {dataset}_ablation_comparison.json              — all variants merged

Usage:
  python train_DDA_ablation.py --dataset C-dataset [--variants all|sim_only,gcn_only,...]
"""

import sys
import os
import timeit
import argparse
import json
from typing import Optional, List

# Force UTF-8 stdout to avoid UnicodeEncodeError on Windows cp932 terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# ── Path setup ────────────────────────────────────────────────────────
THIS_DIR     = os.path.dirname(os.path.abspath(__file__))      # AI_ENGINE/src/
AIENGINE_DIR = os.path.dirname(THIS_DIR)                       # AI_ENGINE/
AMDGT_DIR    = os.path.abspath(os.path.join(AIENGINE_DIR, '..', 'AMDGT_main'))
DATA_OUT_DIR = os.path.join(AIENGINE_DIR, 'data')

sys.path.insert(0, AMDGT_DIR)
sys.path.insert(0, AIENGINE_DIR)
sys.path.insert(0, THIS_DIR)

# NetworkX 3.x compat shim
import networkx as _nx
if not hasattr(_nx, 'from_numpy_matrix'):
    _nx.from_numpy_matrix = _nx.from_numpy_array  # type: ignore[attr-defined]
if not hasattr(_nx, 'to_numpy_matrix'):
    _nx.to_numpy_matrix = _nx.to_numpy_array  # type: ignore[attr-defined]

from data_preprocess import get_data, data_processing, k_fold, dgl_similarity_graph, dgl_heterograph, get_adj
from metric import get_metric, metrics_to_dict, print_metric_header, print_metric_row
from model.AMNTDDA_Ablation import AMNTDDA_Ablation, ABLATION_CONFIGS, VARIANT_ORDER
from topo_analysis import compute_topo_features

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# ── Helpers ───────────────────────────────────────────────────────────

def _ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def _csv_path(results_dir: str, dataset: str, variant: str) -> str:
    return os.path.join(results_dir, f'{dataset}_ablation_{variant}_fold_results.csv')


def save_fold_result(results_dir: str, dataset: str, variant: str,
                     fold: int, fold_metrics: dict):
    path = _csv_path(results_dir, dataset, variant)
    row  = {'fold': fold, **fold_metrics}
    df   = pd.DataFrame([row])
    df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)


def compute_and_save_summary(results_dir: str, dataset: str,
                              variant: str) -> Optional[dict]:
    path = _csv_path(results_dir, dataset, variant)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df[pd.to_numeric(df['fold'], errors='coerce').notna()]
    metrics = ['AUC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC']
    cfg     = ABLATION_CONFIGS[variant]
    summary = {
        'dataset':  dataset,
        'variant':  variant,
        'trained':  True,
        'name_vi':  cfg['name_vi'],
        'name_en':  cfg['name_en'],
        'group':    cfg['group'],
        'desc':     cfg['desc'],
        'color':    cfg['color'],
        'icon':     cfg['icon'],
        'n_folds':  len(df),
    }
    for m in metrics:
        if m in df.columns:
            summary[f'{m}_mean'] = round(float(df[m].mean()), 6)
            summary[f'{m}_std']  = round(float(df[m].std()),  6)
        else:
            summary[f'{m}_mean'] = 0.0
            summary[f'{m}_std']  = 0.0

    out = os.path.join(results_dir, f'{dataset}_ablation_{variant}_summary.json')
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    return summary


def build_ablation_comparison(results_dir: str, dataset: str,
                               variants: List[str]) -> dict:
    """Merge all variant summaries into a master comparison JSON."""
    comparison = {
        'dataset':  dataset,
        'variants': {},
        'variant_order': [v for v in VARIANT_ORDER if v in variants],
    }
    for variant in variants:
        p = os.path.join(results_dir, f'{dataset}_ablation_{variant}_summary.json')
        if os.path.exists(p):
            with open(p, encoding='utf-8') as fh:
                comparison['variants'][variant] = json.load(fh)

    out = os.path.join(results_dir, f'{dataset}_ablation_comparison.json')
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(comparison, fh, indent=2, ensure_ascii=False)
    print(f"\nAblation comparison saved → {out}")
    return comparison


def _print_summary_table(summaries: dict):
    metrics = ['AUC', 'AUPR', 'Accuracy', 'F1', 'MCC']
    header  = f"  {'Variant':<24}  " + "  ".join(f"{m:>8}" for m in metrics)
    print(f"\n{'='*80}")
    print("ABLATION COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(header)
    print(f"  {'-'*76}")
    for variant in VARIANT_ORDER:
        if variant not in summaries:
            continue
        s   = summaries[variant]
        row = f"  {variant:<24}  " + "  ".join(
            f"{s.get(f'{m}_mean', 0.0):>8.4f}" for m in metrics)
        print(row)
    print(f"{'='*80}")


# ── Train one variant for all folds ──────────────────────────────────

def variant_is_trained(results_dir: str, dataset: str, variant: str) -> bool:
    """Return True if a summary JSON already exists for this variant."""
    summary_path = os.path.join(results_dir, f'{dataset}_ablation_{variant}_summary.json')
    return os.path.exists(summary_path)


def load_existing_summary(results_dir: str, dataset: str, variant: str) -> dict:
    """Load and return an existing summary JSON."""
    p = os.path.join(results_dir, f'{dataset}_ablation_{variant}_summary.json')
    with open(p, encoding='utf-8') as fh:
        return json.load(fh)


def train_variant(variant: str, args, data: dict,
                  drdr_graph, didi_graph,
                  drug_feature: torch.Tensor,
                  disease_feature: torch.Tensor,
                  protein_feature: torch.Tensor,
                  drug_topo: torch.Tensor,
                  disease_topo: torch.Tensor,
                  results_dir: str,
                  models_dir: str,
                  force: bool = False) -> dict:
    """Train one ablation variant for k_fold folds. Returns summary dict."""

    cfg = ABLATION_CONFIGS[variant]
    print(f"\n{'#'*70}")
    print(f"  Variant: {variant}  ({cfg['name_vi']})")
    print(f"  Group:   {cfg['group']}")
    print(f"  Config:  use_sim={cfg['use_sim']}  use_gcn={cfg['use_gcn']}  "
          f"use_trans={cfg['use_trans']}  cross_modal={cfg['cross_modal']}")
    print(f"{'#'*70}")

    # Reset fold CSV for fresh run (only when forced)
    old_csv = _csv_path(results_dir, args.dataset, variant)
    if force and os.path.exists(old_csv):
        os.remove(old_csv)

    # Pre-create checkpoint directory
    ckpt_dir = os.path.join(models_dir, variant)
    os.makedirs(ckpt_dir, exist_ok=True)

    cross_entropy = nn.CrossEntropyLoss()
    start         = timeit.default_timer()

    for fold_idx in range(args.k_fold):
        print(f"\n  {'─'*50}\n  fold: {fold_idx}  [{variant}]")
        print_metric_header()

        torch.manual_seed(args.random_seed + fold_idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.random_seed + fold_idx)

        model     = AMNTDDA_Ablation(args, mode=variant).to(device)
        optimizer = optim.Adam(model.parameters(),
                               weight_decay=args.weight_decay, lr=args.lr)

        X_train = torch.LongTensor(data['X_train'][fold_idx]).to(device)
        Y_train = torch.LongTensor(data['Y_train'][fold_idx]).to(device)
        X_test  = torch.LongTensor(data['X_test'][fold_idx]).to(device)
        Y_test  = data['Y_test'][fold_idx].flatten()

        drdipr_graph, data = dgl_heterograph(data, data['X_train'][fold_idx], args)
        drdipr_graph = drdipr_graph.to(device)

        best_auc     = 0.0
        best_metrics: dict = {}

        for epoch in range(args.epochs):
            # --- train ---
            model.train()
            _, train_score = model(
                drdr_graph, didi_graph, drdipr_graph,
                drug_feature, disease_feature, protein_feature,
                X_train, drug_topo, disease_topo)
            loss = cross_entropy(train_score, torch.flatten(Y_train))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- eval ---
            with torch.no_grad():
                model.eval()
                _, test_score = model(
                    drdr_graph, didi_graph, drdipr_graph,
                    drug_feature, disease_feature, protein_feature,
                    X_test, drug_topo, disease_topo)

            test_prob = F.softmax(test_score, dim=-1)[:, 1].cpu().numpy()
            test_pred = torch.argmax(test_score, dim=-1).cpu().numpy()
            AUC, AUPR, acc, prec, rec, f1, mcc = get_metric(Y_test, test_pred, test_prob)
            print_metric_row(epoch + 1, timeit.default_timer() - start,
                             AUC, AUPR, acc, prec, rec, f1, mcc)

            if AUC > best_auc:
                best_auc     = AUC
                best_metrics = metrics_to_dict(AUC, AUPR, acc, prec, rec, f1, mcc)
                # save model checkpoint (ckpt_dir already created above)
                torch.save(model.state_dict(),
                           os.path.join(ckpt_dir, f'fold{fold_idx}.pt'))
                print(f'    ↑ AUC improved at epoch {epoch+1}: {best_auc:.5f}')

        save_fold_result(results_dir, args.dataset, variant, fold_idx, best_metrics)

        # Running mean after each fold
        csv_so_far = _csv_path(results_dir, args.dataset, variant)
        if os.path.exists(csv_so_far):
            df_sf = pd.read_csv(csv_so_far)
            df_sf = df_sf[pd.to_numeric(df_sf['fold'], errors='coerce').notna()]
            print(f"  → Running avg ({len(df_sf)} folds): "
                  f"AUC={df_sf['AUC'].mean():.5f}  "
                  f"AUPR={df_sf['AUPR'].mean():.5f}  "
                  f"F1={df_sf['F1'].mean():.5f}")

    summary = compute_and_save_summary(results_dir, args.dataset, variant)
    return summary or {}


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k_fold',       type=int,   default=10)
    parser.add_argument('--epochs',       type=int,   default=300,
                        help='Fewer epochs for ablation (same seed → comparable)')
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--random_seed',  type=int,   default=1234)
    parser.add_argument('--neighbor',     type=int,   default=20)
    parser.add_argument('--negative_rate',type=float, default=1.0)
    parser.add_argument('--dataset',      default='C-dataset')
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
    parser.add_argument(
        '--variants', default='all',
        help='Comma-separated variant names or "all". '
             f'Available: {",".join(VARIANT_ORDER)}')
    parser.add_argument(
        '--force', action='store_true', default=False,
        help='Re-train even if results already exist for a variant')
    parser.add_argument('--num_threads', type=int, default=4,
        help='Number of CPU threads for PyTorch intraop parallelism')
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(max(1, args.num_threads // 2))
    print(f"CPU threads: {args.num_threads} (intraop) / {max(1, args.num_threads // 2)} (interop)")

    # Resolve variants list
    if args.variants.strip().lower() == 'all':
        run_variants = VARIANT_ORDER
    else:
        run_variants = [v.strip() for v in args.variants.split(',')]
        for v in run_variants:
            if v not in ABLATION_CONFIGS:
                raise ValueError(f"Unknown variant '{v}'. "
                                 f"Available: {list(ABLATION_CONFIGS)}")

    print(f"Running ablation study on: {args.dataset}")
    print(f"Variants: {run_variants}")
    print(f"Device:   {device}")

    # Paths
    args.data_dir = os.path.join(AMDGT_DIR, 'data', args.dataset) + os.sep
    results_dir   = os.path.join(DATA_OUT_DIR, 'results')
    models_dir    = os.path.join(DATA_OUT_DIR, 'models', args.dataset)
    _ensure_dirs(results_dir, models_dir)

    # ── Load data once ────────────────────────────────────────────────
    data = get_data(args)
    args.drug_number    = data['drug_number']
    args.disease_number = data['disease_number']
    args.protein_number = data['protein_number']
    args.disease_feature_dim = int(np.array(data['diseasefeature']).shape[1])

    for fi in range(args.k_fold):
        os.makedirs(os.path.join(args.data_dir, 'fold', str(fi)), exist_ok=True)

    data = data_processing(data, args)
    data = k_fold(data, args)

    drdr_graph, didi_graph, data = dgl_similarity_graph(data, args)
    drdr_graph = drdr_graph.to(device)
    didi_graph = didi_graph.to(device)

    drug_feature    = torch.FloatTensor(data['drugfeature']).to(device)
    disease_feature = torch.FloatTensor(data['diseasefeature']).to(device)
    protein_feature = torch.FloatTensor(data['proteinfeature']).to(device)

    # Topological features (shared across all variants for consistent input)
    print("\nComputing topological features …")
    drdi_mat = get_adj(data['drdi'], (args.drug_number, args.disease_number)).numpy().astype(float)
    drpr_mat = get_adj(data['drpr'], (args.drug_number, args.protein_number)).numpy().astype(float)
    dipr_mat = get_adj(data['dipr'], (args.disease_number, args.protein_number)).numpy().astype(float)

    drug_centrality, disease_centrality = compute_topo_features(drdi_mat, drpr_mat, dipr_mat)
    drug_topo    = torch.FloatTensor(drug_centrality).to(device)
    disease_topo = torch.FloatTensor(disease_centrality).to(device)

    # ── Train each variant ────────────────────────────────────────────
    all_summaries: dict = {}
    for variant in run_variants:
        # ── Skip if already trained and --force not set ───────────────
        if not args.force and variant_is_trained(results_dir, args.dataset, variant):
            print(f"\n[SKIP] {variant}: results already exist "
                  f"(use --force to retrain)")
            all_summaries[variant] = load_existing_summary(
                results_dir, args.dataset, variant)
            continue

        summary = train_variant(
            variant=variant,
            args=args,
            data=data,
            drdr_graph=drdr_graph,
            didi_graph=didi_graph,
            drug_feature=drug_feature,
            disease_feature=disease_feature,
            protein_feature=protein_feature,
            drug_topo=drug_topo,
            disease_topo=disease_topo,
            results_dir=results_dir,
            models_dir=models_dir,
            force=args.force,
        )
        if summary:
            all_summaries[variant] = summary

    # ── Build comparison JSON ─────────────────────────────────────────
    comparison = build_ablation_comparison(results_dir, args.dataset, run_variants)
    _print_summary_table(all_summaries)

    print(f"\nAll ablation results saved → {results_dir}")
