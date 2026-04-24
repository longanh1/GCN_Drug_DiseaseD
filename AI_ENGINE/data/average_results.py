"""
average_results.py — Compute and display cross-fold averages from saved results.

Run after train_DDA.py or train_DDA_fuzzy.py to aggregate fold results and
generate comparison JSON files.

Usage:
  python average_results.py --dataset C-dataset
  python average_results.py --dataset C-dataset --model AMNTDDA_Fuzzy
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────
THIS_DIR     = os.path.dirname(os.path.abspath(__file__))    # AI_ENGINE/data/
DATA_DIR     = THIS_DIR                                       # same folder
RESULTS_DIR  = os.path.join(DATA_DIR, 'results')


METRICS = ['AUC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC']


# ── Helpers ───────────────────────────────────────────────────────────
def compute_average(dataset: str, model_name: str) -> dict | None:
    """Read fold CSV, compute mean±std, save JSON summary."""
    csv_path = os.path.join(RESULTS_DIR, f'{dataset}_{model_name}_fold_results.csv')
    if not os.path.exists(csv_path):
        print(f"[WARN] File not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    n  = len(df)
    summary = {
        'dataset':  dataset,
        'model':    model_name,
        'n_folds':  n,
        'folds':    df.to_dict(orient='records'),
    }

    print(f"\n{'='*65}")
    print(f"Dataset: {dataset}   Model: {model_name}   Folds: {n}")
    print(f"{'='*65}")
    header = f"{'Metric':<12} {'Mean':>10} {'Std':>10}"
    print(header)
    print('-' * len(header))

    for m in METRICS:
        if m in df.columns:
            mean_val = float(df[m].mean())
            std_val  = float(df[m].std())
            summary[f'{m}_mean'] = round(mean_val, 6)
            summary[f'{m}_std']  = round(std_val,  6)
            print(f"{m:<12} {mean_val:>10.4f} {std_val:>10.4f}")

    # Save summary JSON
    out_path = os.path.join(RESULTS_DIR, f'{dataset}_{model_name}_summary.json')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n→ Summary saved: {out_path}")
    return summary


def compare_models(dataset: str) -> dict:
    """Merge summaries for all available models into a comparison JSON."""
    model_names = ['AMNTDDA', 'AMNTDDA_Fuzzy']
    comparison  = {'dataset': dataset, 'models': {}}

    for name in model_names:
        p = os.path.join(RESULTS_DIR, f'{dataset}_{name}_summary.json')
        if os.path.exists(p):
            with open(p) as fh:
                comparison['models'][name] = json.load(fh)

    if len(comparison['models']) >= 2:
        # Delta GCN → Fuzzy
        base   = comparison['models'].get('AMNTDDA', {})
        fuzzy  = comparison['models'].get('AMNTDDA_Fuzzy', {})
        deltas = {}
        for m in METRICS:
            bk = f'{m}_mean'
            if bk in base and bk in fuzzy:
                deltas[m] = round(fuzzy[bk] - base[bk], 6)
        comparison['delta_fuzzy_vs_gcn'] = deltas

        print(f"\n{'='*65}")
        print(f"Model Comparison  |  {dataset}")
        print(f"{'='*65}")
        print(f"{'Metric':<12}  {'AMNTDDA':>10}  {'Fuzzy':>10}  {'Δ':>10}")
        print('-' * 50)
        for m in METRICS:
            bv = base.get(f'{m}_mean', float('nan'))
            fv = fuzzy.get(f'{m}_mean', float('nan'))
            dv = deltas.get(m, float('nan'))
            sign = '+' if dv >= 0 else ''
            print(f"{m:<12}  {bv:>10.4f}  {fv:>10.4f}  {sign}{dv:>9.4f}")

    out_path = os.path.join(RESULTS_DIR, f'{dataset}_comparison.json')
    with open(out_path, 'w') as fh:
        json.dump(comparison, fh, indent=2)
    print(f"\n→ Comparison saved: {out_path}")
    return comparison


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute average results from k-fold training.')
    parser.add_argument('--dataset', default='C-dataset',
                        help='Dataset name (B-dataset / C-dataset / F-dataset)')
    parser.add_argument('--model',   default=None,
                        help='Model name to average; leave blank for all')
    args = parser.parse_args()

    models_to_process = (
        [args.model] if args.model else ['AMNTDDA', 'AMNTDDA_Fuzzy']
    )

    for model_name in models_to_process:
        compute_average(args.dataset, model_name)

    compare_models(args.dataset)
