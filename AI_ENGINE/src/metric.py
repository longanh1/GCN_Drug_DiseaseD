"""
metric.py — Upgraded metric module for AI_ENGINE.

Extends the original AMDGT_main/metric.py with:
- CSV export of per-epoch and per-fold results
- Formatted print utilities
"""

import numpy as np
import pandas as pd
import os
from sklearn.metrics import (
    auc, accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, precision_recall_curve,
    roc_curve, roc_auc_score
)


def get_metric(y_true, y_pred, y_prob):
    """Compute all classification metrics. Same interface as AMDGT_main/metric.py."""
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    Auc = auc(fpr, tpr)

    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
    Aupr = auc(rec_curve, prec_curve)

    return Auc, Aupr, accuracy, precision, recall, f1, mcc


def metrics_to_dict(AUC, AUPR, accuracy, precision, recall, f1, mcc):
    """Convert metric values to a named dictionary."""
    return {
        'AUC': round(float(AUC), 6),
        'AUPR': round(float(AUPR), 6),
        'Accuracy': round(float(accuracy), 6),
        'Precision': round(float(precision), 6),
        'Recall': round(float(recall), 6),
        'F1': round(float(f1), 6),
        'MCC': round(float(mcc), 6),
    }


def print_metric_header():
    print('Epoch\t\tTime\t\tAUC\t\tAUPR\t\tAccuracy\t\tPrecision\t\tRecall\t\tF1-score\t\tMcc')


def print_metric_row(epoch, time_elapsed, AUC, AUPR, accuracy, precision, recall, f1, mcc):
    show = [epoch, round(time_elapsed, 2), round(AUC, 5), round(AUPR, 5),
            round(accuracy, 5), round(precision, 5), round(recall, 5),
            round(f1, 5), round(mcc, 5)]
    print('\t\t'.join(map(str, show)))


def append_epoch_to_csv(csv_path: str, fold: int, epoch: int, time_elapsed: float,
                         AUC: float, AUPR: float, accuracy: float,
                         precision: float, recall: float, f1: float, mcc: float):
    """Append one epoch's metrics to a CSV file."""
    row = {
        'fold': fold, 'epoch': epoch, 'time': round(time_elapsed, 2),
        'AUC': round(AUC, 6), 'AUPR': round(AUPR, 6),
        'Accuracy': round(accuracy, 6), 'Precision': round(precision, 6),
        'Recall': round(recall, 6), 'F1': round(f1, 6), 'MCC': round(mcc, 6),
    }
    df = pd.DataFrame([row])
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode='a', header=header, index=False)


def summarize_folds(fold_metrics: list, model_name: str = '') -> dict:
    """Compute mean and std across folds from a list of metric dicts."""
    keys = ['AUC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC']
    summary = {'model': model_name, 'n_folds': len(fold_metrics)}
    for k in keys:
        vals = [m[k] for m in fold_metrics]
        summary[f'{k}_mean'] = round(float(np.mean(vals)), 6)
        summary[f'{k}_std'] = round(float(np.std(vals)), 6)
    return summary
