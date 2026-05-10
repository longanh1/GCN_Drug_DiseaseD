import torch
import numpy as np
import pandas as pd
import argparse
import os
import sys
from graph_vae import build_vgae, generate_new_edges

# --- CẤU HÌNH ---
EPOCHS = 500
LR = 0.01

THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
AMDGT_DIR  = os.path.abspath(os.path.join(THIS_DIR, '..', '..', 'AMDGT_main'))
OUTPUT_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'data', 'results'))


def load_graph_data(dataset: str = 'B-dataset'):
    """Đọc dữ liệu từ CSV files trong AMDGT_main/data/{dataset}."""
    base = os.path.join(AMDGT_DIR, 'data', dataset)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Dataset không tồn tại: {base}")

    # ── Drug features: DrugFingerprint + DrugGIP → concat (dim = 269+269 = 538) ──
    fp  = pd.read_csv(os.path.join(base, 'DrugFingerprint.csv'), index_col=0).values.astype(float)
    gip = pd.read_csv(os.path.join(base, 'DrugGIP.csv'),         index_col=0).values.astype(float)
    x_drug = np.concatenate([fp, gip], axis=1)   # (n_drug, 538)

    # ── Protein features: Protein_ESM (dim = 320), pad đến 538 ──
    esm  = pd.read_csv(os.path.join(base, 'Protein_ESM.csv'), index_col=0).values.astype(float)
    pad  = np.zeros((esm.shape[0], x_drug.shape[1] - esm.shape[1]), dtype=float)
    x_prot = np.concatenate([esm, pad], axis=1)  # (n_prot, 538)

    x = torch.tensor(np.vstack([x_drug, x_prot]), dtype=torch.float)

    # ── Drug-Protein edges ──
    dp = pd.read_csv(os.path.join(base, 'DrugProteinAssociationNumber.csv'), index_col=0)
    n_drug   = x_drug.shape[0]
    n_prot   = x_prot.shape[0]
    offset_p = n_drug

    drug_ids = dp.index.tolist()
    prot_ids = dp[dp.columns[0]].tolist()

    # Chuẩn hoá về 0-based nếu cần
    drug_min = min(drug_ids) if drug_ids else 0
    prot_min = min(prot_ids) if prot_ids else 0
    drug_ids = [d - drug_min for d in drug_ids]
    prot_ids = [p - prot_min for p in prot_ids]

    # Lọc bỏ các edge có index ngoài bounds (lỗi dữ liệu trong CSV)
    valid = [(d, p) for d, p in zip(drug_ids, prot_ids) if d < n_drug and p < n_prot]
    if not valid:
        raise ValueError(f"Không có edge hợp lệ nào trong {dataset}")
    drug_ids, prot_ids = zip(*valid)

    edge_index = torch.tensor(
        [[d, p + offset_p] for d, p in zip(drug_ids, prot_ids)],
        dtype=torch.long
    ).t().contiguous()

    print(f"Dataset   : {dataset}")
    print(f"Drugs     : {n_drug}  |  Proteins: {x_prot.shape[0]}")
    print(f"Edges     : {edge_index.shape[1]}")
    print(f"Feature dim: {x.shape[1]}")

    return x, edge_index, n_drug, offset_p

def train_vgae(dataset: str = 'B-dataset'):
    x, edge_index, n_drug, offset_p = load_graph_data(dataset)
    input_dim = x.size(1)
    model = build_vgae(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"--- Đang huấn luyện GraphVAE trên Victus ---")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(x, edge_index)
        loss = model.recon_loss(z, edge_index) + (1 / x.size(0)) * model.kl_loss()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch: {epoch:03d}, Loss: {loss.item():.4f}")

    # --- GIAI ĐOẠN SINH DỮ LIỆU ---
    model.eval()
    with torch.no_grad():
        z = model.encode(x, edge_index)
        # Tăng threshold lên một chút để lấy "tinh hoa"
        new_edges, _ = generate_new_edges(model, z, threshold=0.95)
        
        # Lọc bỏ các cạnh trùng với dữ liệu cũ (chỉ lấy cái thực sự mới)
        existing_edges = set(map(tuple, edge_index.t().tolist()))
        truly_new = []
        
        print(f"\n🔍 KẾT QUẢ GIẢI MÃ LIÊN KẾT TẠO SINH:")
        for edge in new_edges.tolist():
            u, v = edge
            if tuple(edge) not in existing_edges and (u < n_drug and v >= offset_p):
                # Tính toán ID thực tế: ID = index + 1
                drug_id = u + 1
                prot_id = (v - offset_p) + 1
                print(f"   [!] Phát hiện: Drug ID {drug_id} --(+)--> Protein ID {prot_id}")
                truly_new.append(edge)

        if truly_new:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            out_path = os.path.join(OUTPUT_DIR, f'{dataset}_generated_edges.pt')
            torch.save(torch.tensor(truly_new), out_path)
            print(f"✅ Đã lưu {len(truly_new)} liên kết vào {out_path}")
        else:
            print("⚠️ Không tìm thấy liên kết mới nào đủ tin cậy.")
            
        return truly_new, z

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='B-dataset',
                        choices=['B-dataset', 'C-dataset', 'F-dataset'],
                        help='Dataset để huấn luyện VGAE')
    args = parser.parse_args()
    train_vgae(args.dataset)