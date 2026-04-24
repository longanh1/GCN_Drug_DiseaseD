"""
api.py — FastAPI ML prediction server for AI_ENGINE.

Serves drug/disease data from AMDGT_main CSV files and provides
prediction endpoints.  Runs on port 8000.

Start:  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Paths ─────────────────────────────────────────────────────────────
THIS_DIR     = os.path.dirname(os.path.abspath(__file__))   # AI_ENGINE/
AMDGT_DIR    = os.path.abspath(os.path.join(THIS_DIR, '..', 'AMDGT_main'))
DATA_OUT_DIR = os.path.join(THIS_DIR, 'data')
SRC_DIR      = os.path.join(THIS_DIR, 'src')

sys.path.insert(0, SRC_DIR)    # fuzzy_weight
from fuzzy_weight import MamdaniFIS

# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(title="AI_ENGINE PharmaLink API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global cache ──────────────────────────────────────────────────────
_cache: dict = {}
_mamdani = MamdaniFIS()


# ── Data loading helpers ──────────────────────────────────────────────
def _data_path(dataset: str, filename: str) -> str:
    return os.path.join(AMDGT_DIR, 'data', dataset, filename)


def _load_dataset(dataset: str) -> dict:
    if dataset in _cache:
        return _cache[dataset]

    base = os.path.join(AMDGT_DIR, 'data', dataset)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Dataset folder not found: {base}")

    d = {}

    # Drug information
    drug_info_path = os.path.join(base, 'DrugInformation.csv')
    if os.path.exists(drug_info_path):
        df = pd.read_csv(drug_info_path)
        d['drug_df'] = df
        d['drugs'] = df.to_dict(orient='records')
    else:
        d['drugs'] = []

    # Disease IDs come from the first column of DiseaseFeature.csv
    disease_feat_path = os.path.join(base, 'DiseaseFeature.csv')
    if os.path.exists(disease_feat_path):
        df_dis = pd.read_csv(disease_feat_path, header=None)
        disease_ids = df_dis.iloc[:, 0].astype(str).tolist()
        d['disease_ids'] = disease_ids
        d['diseases'] = [{'id': did, 'name': did.replace('D', 'OMIM:D')} for did in disease_ids]
        d['disease_features'] = df_dis.iloc[:, 1:].to_numpy(dtype=np.float32)
    else:
        d['disease_ids'] = []
        d['diseases'] = []

    # Protein information
    prot_info_path = os.path.join(base, 'ProteinInformation.csv')
    if os.path.exists(prot_info_path):
        d['proteins'] = pd.read_csv(prot_info_path).to_dict(orient='records')
    else:
        d['proteins'] = []

    # Known drug-disease associations
    assoc_path = os.path.join(base, 'DrugDiseaseAssociationNumber.csv')
    if os.path.exists(assoc_path):
        assoc = pd.read_csv(assoc_path, dtype=int)
        d['assoc'] = assoc
        d['assoc_set'] = set(zip(assoc['drug'].tolist(), assoc['disease'].tolist()))
    else:
        d['assoc'] = pd.DataFrame(columns=['drug', 'disease'])
        d['assoc_set'] = set()

    # Drug features for similarity
    fingerprint_path = os.path.join(base, 'DrugFingerprint.csv')
    if os.path.exists(fingerprint_path):
        df_fp = pd.read_csv(fingerprint_path)
        d['drug_features'] = df_fp.iloc[:, 1:].to_numpy(dtype=np.float32)
    else:
        d['drug_features'] = None

    # GIP similarity matrices for neighbour features
    drug_gip_path = os.path.join(base, 'DrugGIP.csv')
    dis_gip_path  = os.path.join(base, 'DiseaseGIP.csv')
    if os.path.exists(drug_gip_path):
        d['drug_gip'] = pd.read_csv(drug_gip_path).iloc[:, 1:].to_numpy(dtype=np.float32)
    if os.path.exists(dis_gip_path):
        d['dis_gip'] = pd.read_csv(dis_gip_path).iloc[:, 1:].to_numpy(dtype=np.float32)

    _cache[dataset] = d
    return d


def _neighbor_sim(gip: np.ndarray, node_idx: int, K: int = 20) -> float:
    """Mean similarity to top-K neighbours of a given node."""
    if gip is None or node_idx >= gip.shape[0]:
        return 0.3
    row = gip[node_idx].copy()
    row[node_idx] = 0.0
    top_k = np.sort(row)[-K:]
    return float(np.mean(top_k))


def _score_drug_all_diseases(dataset_data: dict, drug_idx: int) -> np.ndarray:
    """
    Score a drug against all diseases using GIP-weighted association aggregation.
    For each disease, score = weighted sum of known drug-disease links from
    top-K similar drugs (using DrugGIP similarity).
    Falls back to random baseline if data unavailable.
    """
    num_diseases = len(dataset_data['diseases'])
    assoc_set    = dataset_data.get('assoc_set', set())
    drug_gip     = dataset_data.get('drug_gip')

    if drug_gip is None or drug_idx >= drug_gip.shape[0]:
        rng = np.random.default_rng(drug_idx)
        return rng.random(num_diseases).astype(np.float32)

    # GIP similarity row for query drug
    sim_row = drug_gip[drug_idx].copy()
    sim_row[drug_idx] = 0.0  # exclude self

    # Build disease score: sum of similarity-weighted known links
    scores = np.zeros(num_diseases, dtype=np.float32)
    for other_drug_idx in range(sim_row.shape[0]):
        w = float(sim_row[other_drug_idx])
        if w <= 0:
            continue
        for di_idx in range(num_diseases):
            if (other_drug_idx, di_idx) in assoc_set:
                scores[di_idx] += w

    # Normalise to [0, 1]
    lo, hi = scores.min(), scores.max()
    if hi - lo > 1e-8:
        scores = (scores - lo) / (hi - lo)
    return scores.astype(np.float32)


# ── Schemas ───────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    dataset: str = "C-dataset"
    drug_idx: int
    direction: str = "drug->disease"    # or disease->drug
    model: str = "AMNTDDA_Fuzzy"
    top_k: int = 10


class MatrixRequest(BaseModel):
    dataset: str = "C-dataset"
    drug_indices: List[int]
    disease_indices: List[int]
    model: str = "AMNTDDA_Fuzzy"


# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "AI_ENGINE"}


@app.get("/datasets")
def list_datasets():
    base = os.path.join(AMDGT_DIR, 'data')
    datasets = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    return {"datasets": sorted(datasets)}


@app.get("/stats")
def stats(dataset: str = Query("C-dataset")):
    d = _load_dataset(dataset)
    return {
        "dataset":    dataset,
        "num_drugs":    len(d.get('drugs', [])),
        "num_diseases": len(d.get('diseases', [])),
        "num_proteins": len(d.get('proteins', [])),
        "num_known_links": len(d.get('assoc_set', set())),
    }


@app.get("/drugs")
def get_drugs(dataset: str = Query("C-dataset"),
              search: Optional[str] = Query(None),
              limit: int = Query(100)):
    d = _load_dataset(dataset)
    drugs = d.get('drugs', [])
    if search:
        sl = search.lower()
        drugs = [dr for dr in drugs if sl in str(dr.get('name', '')).lower()
                                       or sl in str(dr.get('id', '')).lower()]
    # Return with index
    result = [{'idx': i, **dr} for i, dr in enumerate(drugs[:limit])]
    return {"drugs": result, "total": len(drugs)}


@app.get("/diseases")
def get_diseases(dataset: str = Query("C-dataset"),
                 search: Optional[str] = Query(None),
                 limit: int = Query(100)):
    d = _load_dataset(dataset)
    diseases = d.get('diseases', [])
    if search:
        sl = search.lower()
        diseases = [di for di in diseases if sl in str(di.get('id', '')).lower()
                                              or sl in str(di.get('name', '')).lower()]
    result = [{'idx': i, **di} for i, di in enumerate(diseases[:limit])]
    return {"diseases": result, "total": len(diseases)}


@app.get("/proteins")
def get_proteins(dataset: str = Query("C-dataset"), limit: int = Query(50)):
    d = _load_dataset(dataset)
    return {"proteins": d.get('proteins', [])[:limit], "total": len(d.get('proteins', []))}


@app.post("/predict/single")
def predict_single(req: PredictRequest):
    d = _load_dataset(req.dataset)
    drugs    = d.get('drugs', [])
    diseases = d.get('diseases', [])

    if req.drug_idx < 0 or req.drug_idx >= len(drugs):
        raise HTTPException(400, f"drug_idx {req.drug_idx} out of range (0–{len(drugs)-1})")

    scores = _score_drug_all_diseases(d, req.drug_idx)

    # Apply fuzzy refinement if requested
    if 'Fuzzy' in req.model:
        drug_gip = d.get('drug_gip')
        dis_gip  = d.get('dis_gip')
        src_nb   = _neighbor_sim(drug_gip, req.drug_idx)
        results  = []
        for di_idx, raw_score in enumerate(scores):
            tgt_nb     = _neighbor_sim(dis_gip, di_idx)
            fuz_score  = _mamdani.compute(float(raw_score), src_nb, tgt_nb)
            results.append((di_idx, float(raw_score), fuz_score))
    else:
        results = [(i, float(s), float(s)) for i, s in enumerate(scores)]

    # Sort by fuzzy/final score
    results.sort(key=lambda x: x[2], reverse=True)

    output = []
    for rank, (di_idx, raw, fuz) in enumerate(results[:req.top_k]):
        is_known = (req.drug_idx, di_idx) in d.get('assoc_set', set())
        dis_info = diseases[di_idx] if di_idx < len(diseases) else {'id': str(di_idx)}
        output.append({
            "rank":       rank + 1,
            "disease_idx": di_idx,
            "disease_id":  dis_info.get('id', str(di_idx)),
            "disease_name": dis_info.get('name', dis_info.get('id', str(di_idx))),
            "gcn_score":   round(raw, 4),
            "fuzzy_score": round(fuz, 4),
            "is_known":    is_known,
        })

    drug_info = drugs[req.drug_idx] if req.drug_idx < len(drugs) else {}
    return {
        "drug_idx":   req.drug_idx,
        "drug_name":  drug_info.get('name', drug_info.get('id', str(req.drug_idx))),
        "drug_smiles": drug_info.get('smiles', ''),
        "model":      req.model,
        "top_k":      req.top_k,
        "results":    output,
    }


@app.post("/predict/fuzzy_detail")
def fuzzy_detail(dataset: str = Query("C-dataset"),
                 drug_idx: int = Query(...),
                 disease_idx: int = Query(...)):
    d = _load_dataset(dataset)
    scores   = _score_drug_all_diseases(d, drug_idx)
    raw      = float(scores[disease_idx]) if disease_idx < len(scores) else 0.3
    drug_gip = d.get('drug_gip')
    dis_gip  = d.get('dis_gip')
    src_nb   = _neighbor_sim(drug_gip, drug_idx)
    tgt_nb   = _neighbor_sim(dis_gip, disease_idx)
    details  = _mamdani.get_memberships(raw, src_nb, tgt_nb)

    drugs    = d.get('drugs', [])
    diseases = d.get('diseases', [])
    details['drug_name']    = drugs[drug_idx].get('name', str(drug_idx)) if drug_idx < len(drugs) else str(drug_idx)
    details['disease_name'] = diseases[disease_idx].get('id', str(disease_idx)) if disease_idx < len(diseases) else str(disease_idx)
    return details


@app.post("/predict/matrix")
def predict_matrix(req: MatrixRequest):
    d = _load_dataset(req.dataset)
    drugs    = d.get('drugs', [])
    diseases = d.get('diseases', [])
    drug_gip = d.get('drug_gip')
    dis_gip  = d.get('dis_gip')

    cells = []
    for dr_idx in req.drug_indices:
        scores = _score_drug_all_diseases(d, dr_idx)
        for di_idx in req.disease_indices:
            raw = float(scores[di_idx]) if di_idx < len(scores) else 0.0
            if 'Fuzzy' in req.model:
                src_nb = _neighbor_sim(drug_gip, dr_idx)
                tgt_nb = _neighbor_sim(dis_gip, di_idx)
                fuz    = _mamdani.compute(raw, src_nb, tgt_nb)
            else:
                fuz = raw
            is_known = (dr_idx, di_idx) in d.get('assoc_set', set())
            cells.append({
                "drug_idx":    dr_idx,
                "drug_name":   drugs[dr_idx].get('name', str(dr_idx)) if dr_idx < len(drugs) else str(dr_idx),
                "disease_idx": di_idx,
                "disease_name": diseases[di_idx].get('id', str(di_idx)) if di_idx < len(diseases) else str(di_idx),
                "gcn_score":   round(raw, 4),
                "fuzzy_score": round(fuz, 4),
                "delta":       round(fuz - raw, 4),
                "is_known":    is_known,
            })
    return {"cells": cells, "dataset": req.dataset, "model": req.model}


@app.get("/results/training")
def get_training_results(dataset: str = Query("C-dataset"),
                         model: str = Query("AMNTDDA_Fuzzy")):
    results_dir = os.path.join(DATA_OUT_DIR, 'results')

    # Summary
    summary_path = os.path.join(results_dir, f'{dataset}_{model}_summary.json')
    summary = None
    if os.path.exists(summary_path):
        with open(summary_path) as fh:
            summary = json.load(fh)

    # Per-fold CSV
    csv_path = os.path.join(results_dir, f'{dataset}_{model}_fold_results.csv')
    folds = []
    if os.path.exists(csv_path):
        folds = pd.read_csv(csv_path).to_dict(orient='records')

    return {"dataset": dataset, "model": model, "summary": summary, "folds": folds}


@app.get("/results/comparison")
def get_comparison(dataset: str = Query("C-dataset")):
    results_dir   = os.path.join(DATA_OUT_DIR, 'results')
    comparison_path = os.path.join(results_dir, f'{dataset}_comparison.json')
    if not os.path.exists(comparison_path):
        # Return empty comparison
        return {"dataset": dataset, "models": {}}
    with open(comparison_path) as fh:
        return json.load(fh)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
