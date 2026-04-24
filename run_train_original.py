"""
Wrapper để chạy AMDGT_main/train_DDA.py mà không sửa file gốc.
Áp dụng các compatibility patches cho NetworkX 3.x và DGL mới trước khi import.

Usage:
    python run_train_original.py --dataset C-dataset
"""

import sys
import os

# ── 1. NetworkX 3.x shim ─────────────────────────────────────────────
import networkx as nx
if not hasattr(nx, 'from_numpy_matrix'):
    nx.from_numpy_matrix = nx.from_numpy_array
if not hasattr(nx, 'to_numpy_matrix'):
    nx.to_numpy_matrix = nx.to_numpy_array

# ── 2. DGL legacy function shim ──────────────────────────────────────
import dgl.function as _dgl_fn
if not hasattr(_dgl_fn, 'src_mul_edge'):
    _dgl_fn.src_mul_edge = _dgl_fn.u_mul_e
if not hasattr(_dgl_fn, 'copy_edge'):
    _dgl_fn.copy_edge = _dgl_fn.copy_e

# ── 3. Chuyển thư mục và chạy train_DDA.py gốc ──────────────────────
AMDGT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AMDGT_main')
os.chdir(AMDGT_DIR)
sys.path.insert(0, AMDGT_DIR)

import runpy
runpy.run_path(os.path.join(AMDGT_DIR, 'train_DDA.py'), run_name='__main__')
