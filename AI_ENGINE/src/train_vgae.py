import torch
import psycopg2
import numpy as np
import time
import os
from graph_vae import build_vgae, generate_new_edges

# --- CẤU HÌNH ---
DB_URL = "postgresql://postgres:a@localhost:5432/GCH_ThuocBenh"
EPOCHS = 500
LR = 0.01

def load_graph_data():
    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()
    
    # 1. Padding vector
    all_lengths = []
    for table in ["drug", "protein"]:
        cursor.execute(f"SELECT array_length(embedding, 1) FROM {table}")
        res = cursor.fetchall()
        all_lengths.extend([r[0] for r in res if r[0] is not None])
    
    TARGET_LEN = max(all_lengths) if all_lengths else 774

    def fetch_and_pad(table):
        cursor.execute(f"SELECT embedding FROM {table} ORDER BY {table}_id")
        rows = cursor.fetchall()
        return [list(r[0]) + [0.0] * (TARGET_LEN - len(r[0])) for r in rows]

    x_drug = fetch_and_pad("drug")
    x_prot = fetch_and_pad("protein")
    x = torch.tensor(x_drug + x_prot, dtype=torch.float)
    
    # 2. Lấy cạnh hiện tại
    cursor.execute("SELECT drug_id, protein_id FROM drug_protein")
    edges_raw = cursor.fetchall()
    
    offset_p = len(x_drug)
    edge_index = torch.tensor([[d-1, p-1+offset_p] for d, p in edges_raw], dtype=torch.long).t().contiguous()
    
    conn.close()
    return x, edge_index, len(x_drug), offset_p

def train_vgae():
    x, edge_index, n_drug, offset_p = load_graph_data()
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
            torch.save(torch.tensor(truly_new), 'src/generated_edges.pt')
            print(f"✅ Đã lưu {len(truly_new)} liên kết 'vàng' vào generated_edges.pt")
        else:
            print("⚠️ Không tìm thấy liên kết mới nào đủ tin cậy.")
            
        return truly_new, z

if __name__ == "__main__":
    train_vgae()