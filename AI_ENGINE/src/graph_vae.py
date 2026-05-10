"""
graph_vae.py – VGAE thuần PyTorch, không cần torch_geometric.
Kiến trúc: GCN Encoder (2 lớp) → Variational latent → Dot-product Decoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Chuẩn hoá ma trận kề D^{-1/2} A D^{-1/2} ─────────────────────────────
def _normalize_adj(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Trả về sparse COO tensor của ma trận kề chuẩn hoá (với self-loop)."""
    device = edge_index.device

    # Thêm self-loop
    self_loop = torch.arange(num_nodes, dtype=torch.long, device=device).unsqueeze(0).expand(2, -1)
    ei = torch.cat([edge_index, self_loop], dim=1)

    row, col = ei
    deg = torch.zeros(num_nodes, device=device)
    deg.scatter_add_(0, row, torch.ones(row.size(0), device=device))

    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0

    values = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    adj = torch.sparse_coo_tensor(ei, values, (num_nodes, num_nodes))
    return adj.coalesce()


# ── Lớp GCN (thay thế GCNConv của torch_geometric) ────────────────────────
class GCNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(adj, x @ self.weight)


# ── Encoder ────────────────────────────────────────────────────────────────
class VGAEEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1     = GCNLayer(in_channels,     hidden_channels)
        self.conv2     = GCNLayer(hidden_channels,  hidden_channels)
        self.conv_mu   = GCNLayer(hidden_channels,  out_channels)
        self.conv_logstd = GCNLayer(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        # Lớp 1
        h = F.relu(self.conv1(x, adj))
        h = F.dropout(h, p=0.3, training=self.training)
        # Lớp 2
        h = F.relu(self.conv2(h, adj))
        return self.conv_mu(h, adj), self.conv_logstd(h, adj)


# ── VGAE (thay thế VGAE của torch_geometric) ──────────────────────────────
class VGAE(nn.Module):
    def __init__(self, encoder: VGAEEncoder):
        super().__init__()
        self.encoder = encoder
        self._mu     = None
        self._logstd = None

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        adj = _normalize_adj(edge_index, num_nodes)
        mu, logstd = self.encoder(x, adj)
        logstd = logstd.clamp(max=10)
        self._mu     = mu
        self._logstd = logstd
        # Reparameterisation trick
        z = mu + torch.randn_like(mu) * logstd.exp()
        return z

    def recon_loss(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """BCE reconstruction loss với negative sampling."""
        pos_pred = torch.sigmoid((z[edge_index[0]] * z[edge_index[1]]).sum(dim=1))
        pos_loss = -torch.log(pos_pred + 1e-8).mean()

        num_nodes = z.size(0)
        neg_idx = torch.randint(0, num_nodes, edge_index.shape, device=z.device)
        neg_pred = torch.sigmoid((z[neg_idx[0]] * z[neg_idx[1]]).sum(dim=1))
        neg_loss = -torch.log(1 - neg_pred + 1e-8).mean()
        return pos_loss + neg_loss

    def kl_loss(self) -> torch.Tensor:
        return -0.5 * torch.mean(
            1 + 2 * self._logstd - self._mu.pow(2) - (2 * self._logstd).exp()
        )


# ── Factory ────────────────────────────────────────────────────────────────
def build_vgae(input_dim: int) -> VGAE:
    encoder = VGAEEncoder(in_channels=input_dim, hidden_channels=256, out_channels=128)
    return VGAE(encoder)


# ── Sinh liên kết mới ─────────────────────────────────────────────────────
def generate_new_edges(model: VGAE, z: torch.Tensor, threshold: float = 0.90):
    """
    Tái cấu trúc ma trận kề từ z: p(A|Z) = sigmoid(Z Z^T).
    Trả về (indices, adj_prob) — tương thích với torch_geometric API cũ.
    """
    adj_prob = torch.sigmoid(torch.matmul(z, z.t()))
    indices  = (adj_prob > threshold).nonzero()
    return indices, adj_prob
