import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, VGAE

class VGAEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(VGAEEncoder, self).__init__()
        # Lớp 1: Trích xuất đặc trưng bậc thấp (tăng lên 256)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        
        # Lớp 2: Trích xuất đặc trưng bậc cao (Deep Learning)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Nhánh tính Mean và Log-Variance (tăng lên 128)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Đi qua lớp 1 + Activation + Dropout để chống Overfitting
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Đi qua lớp 2
        x = self.conv2(x, edge_index).relu()
        
        # Trả về mu và logstd cho không gian ẩn
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

def build_vgae(input_dim):
    # Nâng cấp thông số: hidden=256, out=128
    # Điều này giúp không gian ẩn "rộng" hơn để chứa nhiều tri thức hóa học hơn
    encoder = VGAEEncoder(in_channels=input_dim, hidden_channels=256, out_channels=128)
    model = VGAE(encoder)
    return model

# --- LOGIC TẠO SINH CẢI TIẾN ---
def generate_new_edges(model, z, threshold=0.90):
    """
    Sử dụng Dot-product Decoder để tái cấu trúc ma trận kề từ z.
    """
    # Công thức cốt lõi của VGAE: p(A|Z) = sigmoid(ZZ^T)
    adj_prob = torch.sigmoid(torch.matmul(z, z.t()))
    
    # Chỉ lấy các cạnh vượt ngưỡng tin cậy
    indices = (adj_prob > threshold).nonzero()
    
    return indices, adj_prob