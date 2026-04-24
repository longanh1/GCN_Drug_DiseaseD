import timeit
import argparse
import os
import json
import numpy as np
import pandas as pd
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as fn

# ── Compatibility patches ─────────────────────────────────────────────
import networkx as nx
if not hasattr(nx, 'from_numpy_matrix'):
    nx.from_numpy_matrix = nx.from_numpy_array  # type: ignore[attr-defined]
if not hasattr(nx, 'to_numpy_matrix'):
    nx.to_numpy_matrix = nx.to_numpy_array  # type: ignore[attr-defined]

import dgl.function as _dgl_fn
if not hasattr(_dgl_fn, 'src_mul_edge'):
    _dgl_fn.src_mul_edge = _dgl_fn.u_mul_e  # type: ignore[attr-defined]
if not hasattr(_dgl_fn, 'copy_edge'):
    _dgl_fn.copy_edge = _dgl_fn.copy_e  # type: ignore[attr-defined]

from data_preprocess import *
from model.AMNTDDA import AMNTDDA

# ── Đường dẫn lưu kết quả vào AI_ENGINE ──────────────────────────────
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_AIENGINE_DIR = os.path.join(_THIS_DIR, '..', 'AI_ENGINE')
_RESULTS_DIR  = os.path.normpath(os.path.join(_AIENGINE_DIR, 'data', 'results'))
os.makedirs(_RESULTS_DIR, exist_ok=True)
from metric import *

device = torch.device('cuda')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--k_fold', type=int, default=10, help='k-fold cross validation')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight_decay')
    parser.add_argument('--random_seed', type=int, default=1234, help='random seed')
    parser.add_argument('--neighbor', type=int, default=20, help='neighbor')
    parser.add_argument('--negative_rate', type=float, default=1.0, help='negative_rate')
    parser.add_argument('--dataset', default='C-dataset', help='dataset')
    parser.add_argument('--dropout', default='0.2', type=float, help='dropout')
    parser.add_argument('--gt_layer', default='2', type=int, help='graph transformer layer')
    parser.add_argument('--gt_head', default='2', type=int, help='graph transformer head')
    parser.add_argument('--gt_out_dim', default='200', type=int, help='graph transformer output dimension')
    parser.add_argument('--hgt_layer', default='2', type=int, help='heterogeneous graph transformer layer')
    parser.add_argument('--hgt_head', default='8', type=int, help='heterogeneous graph transformer head')
    parser.add_argument('--hgt_in_dim', default='64', type=int, help='heterogeneous graph transformer input dimension')
    parser.add_argument('--hgt_head_dim', default='25', type=int, help='heterogeneous graph transformer head dimension')
    parser.add_argument('--hgt_out_dim', default='200', type=int, help='heterogeneous graph transformer output dimension')
    parser.add_argument('--tr_layer', default='2', type=int, help='transformer layer')
    parser.add_argument('--tr_head', default='4', type=int, help='transformer head')

    args = parser.parse_args()
    args.data_dir = 'data/' + args.dataset + '/'
    args.result_dir = 'Result/' + args.dataset + '/AMNTDDA/'

    # Tạo thư mục fold nếu chưa có
    for _fi in range(args.k_fold):
        os.makedirs(os.path.join(args.data_dir, 'fold', str(_fi)), exist_ok=True)

    # Xóa CSV cũ để tránh append chồng dữ liệu
    _old_csv = os.path.join(_RESULTS_DIR, f'{args.dataset}_AMNTDDA_fold_results.csv')
    if os.path.exists(_old_csv):
        os.remove(_old_csv)
        print(f'Đã xóa CSV cũ: {_old_csv}')

    data = get_data(args)
    args.drug_number = data['drug_number']
    args.disease_number = data['disease_number']
    args.protein_number = data['protein_number']

    data = data_processing(data, args)
    data = k_fold(data, args)

    drdr_graph, didi_graph, data = dgl_similarity_graph(data, args)

    drdr_graph = drdr_graph.to(device)
    didi_graph = didi_graph.to(device)

    drug_feature = torch.FloatTensor(data['drugfeature']).to(device)
    disease_feature = torch.FloatTensor(data['diseasefeature']).to(device)
    protein_feature = torch.FloatTensor(data['proteinfeature']).to(device)
    all_sample = torch.tensor(data['all_drdi']).long()

    start = timeit.default_timer()

    cross_entropy = nn.CrossEntropyLoss()

    Metric = ('Epoch\t\tTime\t\tAUC\t\tAUPR\t\tAccuracy\t\tPrecision\t\tRecall\t\tF1-score\t\tMcc')
    AUCs, AUPRs = [], []

    print('Dataset:', args.dataset)

    for i in range(args.k_fold):

        print('fold:', i)
        print(Metric)

        model = AMNTDDA(args)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

        best_auc, best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = 0, 0, 0, 0, 0, 0, 0
        _fold_metrics = {}
        X_train = torch.LongTensor(data['X_train'][i]).to(device)
        Y_train = torch.LongTensor(data['Y_train'][i]).to(device)
        X_test = torch.LongTensor(data['X_test'][i]).to(device)
        Y_test = data['Y_test'][i].flatten()

        drdipr_graph, data = dgl_heterograph(data, data['X_train'][i], args)
        drdipr_graph = drdipr_graph.to(device)

        for epoch in range(args.epochs):
            model.train()
            _, train_score = model(drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature, X_train)
            train_loss = cross_entropy(train_score, torch.flatten(Y_train))
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                dr_representation, test_score = model(drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature, X_test)

            test_prob = fn.softmax(test_score, dim=-1)
            test_score = torch.argmax(test_score, dim=-1)

            test_prob = test_prob[:, 1]
            test_prob = test_prob.cpu().numpy()

            test_score = test_score.cpu().numpy()

            AUC, AUPR, accuracy, precision, recall, f1, mcc = get_metric(Y_test, test_score, test_prob)

            end = timeit.default_timer()
            time = end - start
            show = [epoch + 1, round(time, 2), round(float(AUC), 5), round(float(AUPR), 5), round(float(accuracy), 5),
                       round(float(precision), 5), round(float(recall), 5), round(float(f1), 5), round(float(mcc), 5)]
            print('\t\t'.join(map(str, show)))
            if AUC > best_auc:
                best_epoch = epoch + 1
                best_auc = AUC
                best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = AUPR, accuracy, precision, recall, f1, mcc
                _fold_metrics = dict(AUC=AUC, AUPR=AUPR, Accuracy=accuracy,
                                     Precision=precision, Recall=recall, F1=f1, MCC=mcc)
                print('AUC improved at epoch ', best_epoch, ';\tbest_auc:', best_auc)

        AUCs.append(best_auc)
        AUPRs.append(best_aupr)

        # ── Lưu fold result vào AI_ENGINE ────────────────────────────
        _csv_path = os.path.join(_RESULTS_DIR, f'{args.dataset}_AMNTDDA_fold_results.csv')
        _row = pd.DataFrame([{'fold': i, **{k: round(float(v), 6) for k, v in _fold_metrics.items()}}])
        _row.to_csv(_csv_path, mode='a', header=not os.path.exists(_csv_path), index=False)
        print(f'[fold {i}] saved → {_csv_path}')

        # ── In trung bình tích lũy sau mỗi fold ──────────────────────
        _df_so_far = pd.read_csv(_csv_path)
        print(f'  → Trung bình {len(_df_so_far)} fold(s): '
              f'AUC={_df_so_far["AUC"].mean():.5f}  '
              f'AUPR={_df_so_far["AUPR"].mean():.5f}  '
              f'Acc={_df_so_far["Accuracy"].mean():.5f}  '
              f'F1={_df_so_far["F1"].mean():.5f}')

    print('AUC:', AUCs)
    AUC_mean = np.mean(AUCs)
    AUC_std = np.std(AUCs)
    print('Mean AUC:', AUC_mean, '(', AUC_std, ')')

    print('AUPR:', AUPRs)
    AUPR_mean = np.mean(AUPRs)
    AUPR_std = np.std(AUPRs)
    print('Mean AUPR:', AUPR_mean, '(', AUPR_std, ')')

    # ── Bảng tổng kết đầy đủ tất cả metrics ─────────────────────────
    _csv_final = os.path.join(_RESULTS_DIR, f'{args.dataset}_AMNTDDA_fold_results.csv')
    if os.path.exists(_csv_final):
        _df_final = pd.read_csv(_csv_final)
        _all_metrics = ['AUC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC']
        print(f"\n{'='*70}")
        print(f"TỔNG KẾT  |  AMNTDDA  |  {args.dataset}  |  {len(_df_final)} folds")
        print(f"{'='*70}")
        print(f"  {'Metric':<12}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
        print(f"  {'-'*48}")
        for _m in _all_metrics:
            if _m in _df_final.columns:
                print(f"  {_m:<12}  {_df_final[_m].mean():>8.5f}  "
                      f"{_df_final[_m].std():>8.5f}  "
                      f"{_df_final[_m].min():>8.5f}  "
                      f"{_df_final[_m].max():>8.5f}")
        print(f"{'='*70}")

        # Ghi dòng Mean và Std vào cuối CSV
        _mean_row = {'fold': 'Mean', **{_m: round(float(_df_final[_m].mean()), 10) for _m in _all_metrics if _m in _df_final.columns}}
        _std_row  = {'fold': 'Std',  **{_m: round(float(_df_final[_m].std()),  10) for _m in _all_metrics if _m in _df_final.columns}}
        pd.DataFrame([_mean_row, _std_row]).to_csv(_csv_final, mode='a', header=False, index=False)
        print(f'Mean/Std rows appended → {_csv_final}')

    # ── Tạo summary + comparison JSON ────────────────────────────────
    _csv_path = os.path.join(_RESULTS_DIR, f'{args.dataset}_AMNTDDA_fold_results.csv')
    if os.path.exists(_csv_path):
        _df = pd.read_csv(_csv_path)
        _metrics = ['AUC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC']
        _summary = {'dataset': args.dataset, 'model': 'AMNTDDA', 'n_folds': len(_df)}
        for _m in _metrics:
            if _m in _df.columns:
                _summary[f'{_m}_mean'] = round(float(_df[_m].mean()), 6)
                _summary[f'{_m}_std']  = round(float(_df[_m].std()),  6)
        _sum_path = os.path.join(_RESULTS_DIR, f'{args.dataset}_AMNTDDA_summary.json')
        with open(_sum_path, 'w') as _fh:
            json.dump(_summary, _fh, indent=2)
        print(f'Summary saved → {_sum_path}')

        # Cập nhật comparison.json
        _comp = {'dataset': args.dataset}
        for _mn in ('AMNTDDA', 'AMNTDDA_Fuzzy'):
            _p = os.path.join(_RESULTS_DIR, f'{args.dataset}_{_mn}_summary.json')
            if os.path.exists(_p):
                with open(_p) as _fh:
                    _comp[_mn] = json.load(_fh)
        _comp_path = os.path.join(_RESULTS_DIR, f'{args.dataset}_comparison.json')
        with open(_comp_path, 'w') as _fh:
            json.dump(_comp, _fh, indent=2)
        print(f'Comparison updated → {_comp_path}')



