"""
topo_analysis.py — Upgraded topological feature extraction.

Removes database dependency. Accepts numpy adjacency matrices directly.
Builds a NetworkX graph from drug-disease(-protein) connections and
computes degree centrality for each drug and disease node.
"""

import numpy as np
import networkx as nx


def compute_topo_features(
        drug_disease_matrix: np.ndarray,
        drug_protein_matrix: np.ndarray = None,
        disease_protein_matrix: np.ndarray = None,
) -> tuple:
    """
    Build a heterogeneous graph from adjacency matrices and compute
    degree centrality for each drug and disease node.

    Parameters
    ----------
    drug_disease_matrix   : ndarray (num_drugs, num_diseases)
    drug_protein_matrix   : ndarray (num_drugs, num_proteins), optional
    disease_protein_matrix: ndarray (num_diseases, num_proteins), optional

    Returns
    -------
    drug_centrality    : ndarray (num_drugs,)   values in [0, 1]
    disease_centrality : ndarray (num_diseases,) values in [0, 1]
    """
    num_drugs    = drug_disease_matrix.shape[0]
    num_diseases = drug_disease_matrix.shape[1]

    G = nx.Graph()
    G.add_nodes_from([f'dr_{i}' for i in range(num_drugs)])
    G.add_nodes_from([f'di_{j}' for j in range(num_diseases)])

    # Drug-disease edges
    rows, cols = np.where(drug_disease_matrix > 0)
    G.add_edges_from([(f'dr_{r}', f'di_{c}') for r, c in zip(rows, cols)])

    # Drug-protein edges
    if drug_protein_matrix is not None:
        num_proteins = drug_protein_matrix.shape[1]
        G.add_nodes_from([f'pr_{p}' for p in range(num_proteins)])
        pr_rows, pr_cols = np.where(drug_protein_matrix > 0)
        G.add_edges_from([(f'dr_{r}', f'pr_{c}') for r, c in zip(pr_rows, pr_cols)])

    # Disease-protein edges
    if disease_protein_matrix is not None:
        dp_rows, dp_cols = np.where(disease_protein_matrix > 0)
        G.add_edges_from([(f'di_{r}', f'pr_{c}') for r, c in zip(dp_rows, dp_cols)])

    print(f"Topo graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    centrality = nx.degree_centrality(G)

    drug_c    = np.array([centrality.get(f'dr_{i}', 0.0) for i in range(num_drugs)],    dtype=np.float32)
    disease_c = np.array([centrality.get(f'di_{j}', 0.0) for j in range(num_diseases)], dtype=np.float32)

    # Normalise to [0, 1]
    if drug_c.max() > 0:
        drug_c = drug_c / drug_c.max()
    if disease_c.max() > 0:
        disease_c = disease_c / disease_c.max()

    print(">>> ĐÃ CẬP NHẬT ĐẶC TRƯNG TOPO VÀO EMBEDDING!")
    return drug_c, disease_c


# ── legacy entry-point ────────────────────────────────────────────────
def calculate_topo():
    """Demo with random matrices."""
    rng = np.random.default_rng(42)
    dd = (rng.random((50, 40)) > 0.95).astype(float)
    dp = (rng.random((50, 100)) > 0.97).astype(float)
    dis_p = (rng.random((40, 100)) > 0.97).astype(float)
    drug_c, disease_c = compute_topo_features(dd, dp, dis_p)
    print(f"Drug centrality sample:    {drug_c[:5]}")
    print(f"Disease centrality sample: {disease_c[:5]}")


if __name__ == "__main__":
    calculate_topo()
