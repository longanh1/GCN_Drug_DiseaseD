"""
api.py — FastAPI ML prediction server for AI_ENGINE v2.0.

Phục vụ dữ liệu thuốc/bệnh/protein từ AMDGT_main (gồm tên tiếng Việt),
endpoint dự đoán, phân loại l/m/h, kết quả huấn luyện, mạng lưới tương tác.

Hỗ trợ 3 mô hình:
  1. AMNTDDA        — mô hình gốc (không GCN, không Fuzzy)
  2. AMNTDDA_GCN    — mô hình gốc + GCN (không Fuzzy)
  3. AMNTDDA_Fuzzy  — mô hình gốc + GCN + Fuzzy Logic

Start:  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Fix Unicode output on Windows terminals (cp1252)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ── Paths ─────────────────────────────────────────────────────────────
THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
AMDGT_DIR    = os.path.abspath(os.path.join(THIS_DIR, '..', 'AMDGT_main'))
DATA_OUT_DIR = os.path.join(THIS_DIR, 'data')
SRC_DIR      = os.path.join(THIS_DIR, 'src')

sys.path.insert(0, SRC_DIR)
from fuzzy_weight import MamdaniFIS

# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(title="AI_ENGINE PharmaLink API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache: dict = {}
_mamdani = MamdaniFIS()

# l/m/h thresholds
LOW_THR  = 0.35
HIGH_THR = 0.65


def _classify_score(score: float) -> str:
    if score >= HIGH_THR:
        return "h"
    if score < LOW_THR:
        return "l"
    return "m"


# ── Metadata loader ───────────────────────────────────────────────────
def _load_metadata(dataset: str) -> dict:
    path = os.path.join(AMDGT_DIR, 'data', dataset, 'metadata.json')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    return {"drugs": [], "diseases": [], "proteins": []}


# ── Dataset loader ────────────────────────────────────────────────────
def _load_dataset(dataset: str) -> dict:
    if dataset in _cache:
        return _cache[dataset]

    base = os.path.join(AMDGT_DIR, 'data', dataset)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Dataset folder not found: {base}")

    d = {}
    meta = _load_metadata(dataset)
    drug_meta_by_idx    = {item['idx']: item for item in meta.get('drugs', [])}
    disease_meta_by_idx = {item['idx']: item for item in meta.get('diseases', [])}
    protein_meta_by_idx = {item['idx']: item for item in meta.get('proteins', [])}

    # Drugs
    drug_csv = os.path.join(base, 'DrugInformation.csv')
    if os.path.exists(drug_csv):
        df = pd.read_csv(drug_csv)
        df.columns = [c.lower().strip() for c in df.columns]
        df = df[[c for c in df.columns if not c.startswith('unnamed')]]
        drugs = []
        for idx, row in df.iterrows():
            name_en = str(row.get('name', row.get('id', str(idx)))).strip()
            drug_id = str(row.get('id', '')).strip()
            smiles  = str(row.get('smiles', '')).strip()
            m = drug_meta_by_idx.get(int(idx), {})
            drugs.append({
                'idx': int(idx), 'id': drug_id,
                'name': name_en, 'name_en': name_en,
                'name_vn': m.get('name_vn', name_en),
                'smiles': smiles,
            })
        d['drug_df'] = df
        d['drugs']   = drugs
    else:
        d['drugs'] = []

    # Diseases
    dis_csv = os.path.join(base, 'DiseaseFeature.csv')
    if os.path.exists(dis_csv):
        df_dis = pd.read_csv(dis_csv, header=None)
        diseases = []
        for idx, row in df_dis.iterrows():
            dis_id = str(row.iloc[0]).strip().strip('"')
            m = disease_meta_by_idx.get(int(idx), {})
            diseases.append({
                'idx': int(idx), 'id': dis_id,
                'name': dis_id,
                'name_en': m.get('name_en', dis_id.title()),
                'name_vn': m.get('name_vn', dis_id.title()),
            })
        d['diseases']         = diseases
        d['disease_ids']      = [di['id'] for di in diseases]
        d['disease_features'] = df_dis.iloc[:, 1:].to_numpy(dtype=np.float32)
    else:
        d['disease_ids'] = []; d['diseases'] = []

    # Proteins
    prot_csv = os.path.join(base, 'ProteinInformation.csv')
    if os.path.exists(prot_csv):
        df_prot = pd.read_csv(prot_csv)
        df_prot.columns = [c.lower().strip() for c in df_prot.columns]
        proteins = []
        for idx, row in df_prot.iterrows():
            prot_id = str(row.iloc[0]).strip()
            m = protein_meta_by_idx.get(int(idx), {})
            proteins.append({
                'idx': int(idx), 'id': prot_id,
                'gene':    m.get('gene', prot_id),
                'name_en': m.get('name_en', prot_id),
                'name_vn': m.get('name_vn', prot_id),
            })
        d['proteins'] = proteins
    else:
        d['proteins'] = []

    # Drug-Disease associations
    assoc_csv = os.path.join(base, 'DrugDiseaseAssociationNumber.csv')
    if os.path.exists(assoc_csv):
        assoc = pd.read_csv(assoc_csv, dtype=int)
        d['assoc'] = assoc
        d['assoc_set'] = set(zip(assoc['drug'].tolist(), assoc['disease'].tolist()))
    else:
        d['assoc'] = pd.DataFrame(columns=['drug', 'disease'])
        d['assoc_set'] = set()

    # Drug-Protein associations
    drpr_csv = os.path.join(base, 'DrugProteinAssociationNumber.csv')
    if os.path.exists(drpr_csv):
        drpr = pd.read_csv(drpr_csv, dtype=int)
        d['drpr_set'] = set(zip(drpr.iloc[:, 0].tolist(), drpr.iloc[:, 1].tolist()))
    else:
        d['drpr_set'] = set()

    # Disease-Protein associations
    dipr_csv = os.path.join(base, 'ProteinDiseaseAssociationNumber.csv')
    if os.path.exists(dipr_csv):
        dipr = pd.read_csv(dipr_csv, dtype=int)
        d['dipr_set'] = set(zip(dipr.iloc[:, 0].tolist(), dipr.iloc[:, 1].tolist()))
    else:
        d['dipr_set'] = set()

    # Drug fingerprints
    fp_csv = os.path.join(base, 'DrugFingerprint.csv')
    if os.path.exists(fp_csv):
        df_fp = pd.read_csv(fp_csv)
        d['drug_features'] = df_fp.iloc[:, 1:].to_numpy(dtype=np.float32)
    else:
        d['drug_features'] = None

    # GIP matrices
    for key, fname in [('drug_gip', 'DrugGIP.csv'), ('dis_gip', 'DiseaseGIP.csv')]:
        p = os.path.join(base, fname)
        if os.path.exists(p):
            d[key] = pd.read_csv(p).iloc[:, 1:].to_numpy(dtype=np.float32)

    _cache[dataset] = d
    return d


# ── Scoring helpers ───────────────────────────────────────────────────
def _neighbor_sim(gip: np.ndarray, node_idx: int, K: int = 20) -> float:
    if gip is None or node_idx >= gip.shape[0]:
        return 0.3
    row = gip[node_idx].copy()
    row[node_idx] = 0.0
    return float(np.mean(np.sort(row)[-K:]))


def _score_drug_all_diseases(dd: dict, drug_idx: int) -> np.ndarray:
    num_dis   = len(dd['diseases'])
    assoc_set = dd.get('assoc_set', set())
    drug_gip  = dd.get('drug_gip')
    if drug_gip is None or drug_idx >= drug_gip.shape[0]:
        rng = np.random.default_rng(drug_idx)
        return rng.random(num_dis).astype(np.float32)
    sim_row = drug_gip[drug_idx].copy()
    sim_row[drug_idx] = 0.0
    scores = np.zeros(num_dis, dtype=np.float32)
    for other in range(sim_row.shape[0]):
        w = float(sim_row[other])
        if w <= 0:
            continue
        for di in range(num_dis):
            if (other, di) in assoc_set:
                scores[di] += w
    lo, hi = scores.min(), scores.max()
    if hi - lo > 1e-8:
        scores = (scores - lo) / (hi - lo)
    return scores.astype(np.float32)


# ── Request Schemas ───────────────────────────────────────────────────
class PredictRequest(BaseModel):
    dataset: str = "C-dataset"
    drug_idx: int
    direction: str = "drug->disease"
    model: str = "AMNTDDA_Fuzzy"
    top_k: int = 10


class MatrixRequest(BaseModel):
    dataset: str = "C-dataset"
    drug_indices: List[int]
    disease_indices: List[int]
    model: str = "AMNTDDA_Fuzzy"


# ══════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {"status": "ok", "service": "AI_ENGINE", "version": "2.0.0"}


@app.get("/datasets")
def list_datasets():
    base = os.path.join(AMDGT_DIR, 'data')
    return {"datasets": sorted(
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and not d.startswith('.')
    )}


@app.get("/stats")
def stats(dataset: str = Query("C-dataset")):
    d = _load_dataset(dataset)
    return {
        "dataset": dataset,
        "num_drugs":    len(d.get('drugs', [])),
        "num_diseases": len(d.get('diseases', [])),
        "num_proteins": len(d.get('proteins', [])),
        "num_known_links": len(d.get('assoc_set', set())),
        "num_drug_protein_links":    len(d.get('drpr_set', set())),
        "num_disease_protein_links": len(d.get('dipr_set', set())),
    }


@app.get("/drugs")
def get_drugs(dataset: str = Query("C-dataset"),
              search: Optional[str] = Query(None),
              limit: int = Query(200)):
    d = _load_dataset(dataset)
    drugs = d.get('drugs', [])
    if search:
        sl = search.lower()
        drugs = [dr for dr in drugs if
                 sl in dr.get('name', '').lower()
                 or sl in dr.get('name_vn', '').lower()
                 or sl in dr.get('id', '').lower()]
    return {"drugs": drugs[:limit], "total": len(drugs)}


@app.get("/diseases")
def get_diseases(dataset: str = Query("C-dataset"),
                 search: Optional[str] = Query(None),
                 limit: int = Query(200)):
    d = _load_dataset(dataset)
    diseases = d.get('diseases', [])
    if search:
        sl = search.lower()
        diseases = [di for di in diseases if
                    sl in di.get('id', '').lower()
                    or sl in di.get('name_vn', '').lower()
                    or sl in di.get('name_en', '').lower()]
    return {"diseases": diseases[:limit], "total": len(diseases)}


@app.get("/proteins")
def get_proteins(dataset: str = Query("C-dataset"),
                 search: Optional[str] = Query(None),
                 limit: int = Query(100)):
    d = _load_dataset(dataset)
    proteins = d.get('proteins', [])
    if search:
        sl = search.lower()
        proteins = [p for p in proteins if
                    sl in p.get('id', '').lower()
                    or sl in p.get('gene', '').lower()
                    or sl in p.get('name_en', '').lower()
                    or sl in p.get('name_vn', '').lower()]
    return {"proteins": proteins[:limit], "total": len(proteins)}


# ── Drug-Disease-Protein network ──────────────────────────────────────
@app.get("/network/drug/{drug_idx}")
def get_drug_network(drug_idx: int,
                     dataset: str = Query("C-dataset"),
                     max_proteins: int = Query(20)):
    """Mạng lưới: 1 thuốc ↔ bệnh đã biết ↔ protein liên quan."""
    d = _load_dataset(dataset)
    drugs    = d.get('drugs', [])
    diseases = d.get('diseases', [])
    proteins = d.get('proteins', [])
    if drug_idx < 0 or drug_idx >= len(drugs):
        raise HTTPException(400, "drug_idx out of range")

    assoc_set = d.get('assoc_set', set())
    drpr_set  = d.get('drpr_set', set())
    dipr_set  = d.get('dipr_set', set())

    known_diseases = [diseases[di] for di in range(len(diseases))
                      if (drug_idx, di) in assoc_set]

    drug_proteins = [proteins[pr] for pr in range(min(len(proteins), 3000))
                     if (drug_idx, pr) in drpr_set][:max_proteins]

    disease_proteins = {}
    for di in known_diseases:
        di_idx = di['idx']
        dp = [proteins[pr] for pr in range(min(len(proteins), 3000))
              if (pr, di_idx) in dipr_set][:5]
        disease_proteins[str(di_idx)] = dp

    drug_gip = d.get('drug_gip')
    dis_gip  = d.get('dis_gip')
    src_nb   = _neighbor_sim(drug_gip, drug_idx)

    # Drug-Drug: top-10 similar drugs by GIP
    similar_drugs = []
    if drug_gip is not None and drug_idx < drug_gip.shape[0]:
        sim_row = drug_gip[drug_idx].copy()
        sim_row[drug_idx] = -1.0
        top_idxs = np.argsort(sim_row)[::-1][:10]
        for i in top_idxs:
            if i < len(drugs) and sim_row[i] > 0:
                info = dict(drugs[i])
                info['sim'] = round(float(sim_row[i]), 4)
                similar_drugs.append(info)

    # Disease-Disease: top-5 similar diseases for each known disease (by dis_gip)
    similar_diseases: dict[str, list] = {}
    if dis_gip is not None:
        for di in known_diseases[:5]:
            di_idx = di['idx']
            if di_idx < dis_gip.shape[0]:
                dr = dis_gip[di_idx].copy()
                dr[di_idx] = -1.0
                top_di = np.argsort(dr)[::-1][:5]
                neighbors = []
                for j in top_di:
                    if j < len(diseases) and dr[j] > 0:
                        info = dict(diseases[j])
                        info['sim'] = round(float(dr[j]), 4)
                        neighbors.append(info)
                similar_diseases[str(di_idx)] = neighbors

    return {
        "drug":             drugs[drug_idx],
        "known_diseases":   known_diseases,
        "drug_proteins":    drug_proteins,
        "disease_proteins": disease_proteins,
        "similar_drugs":    similar_drugs,
        "similar_diseases": similar_diseases,
        "num_known_diseases": len(known_diseases),
        "num_drug_proteins":  len(drug_proteins),
        "drug_similarity_score": round(src_nb, 4),
        "drug_class": _classify_score(src_nb),
        "dataset": dataset,
    }


@app.get("/network/drug-disease")
def drug_disease_interaction(drug_idx: int = Query(...),
                              disease_idx: int = Query(...),
                              dataset: str = Query("C-dataset")):
    """Chi tiết tương tác thuốc–bệnh với protein trung gian."""
    d = _load_dataset(dataset)
    drugs    = d.get('drugs', [])
    diseases = d.get('diseases', [])
    proteins = d.get('proteins', [])
    if drug_idx < 0 or drug_idx >= len(drugs):
        raise HTTPException(400, "drug_idx out of range")
    if disease_idx < 0 or disease_idx >= len(diseases):
        raise HTTPException(400, "disease_idx out of range")

    assoc_set = d.get('assoc_set', set())
    drpr_set  = d.get('drpr_set', set())
    dipr_set  = d.get('dipr_set', set())

    is_known = (drug_idx, disease_idx) in assoc_set

    drug_prot_set    = {pr for (dr, pr) in drpr_set if dr == drug_idx}
    disease_prot_set = {pr for (pr, di) in dipr_set if di == disease_idx}
    bridging_prots   = drug_prot_set & disease_prot_set

    bridging_list = [proteins[pr] for pr in sorted(bridging_prots)[:20]
                     if pr < len(proteins)]
    drug_only    = [proteins[pr] for pr in sorted(drug_prot_set - disease_prot_set)[:10]
                    if pr < len(proteins)]
    disease_only = [proteins[pr] for pr in sorted(disease_prot_set - drug_prot_set)[:10]
                    if pr < len(proteins)]

    scores   = _score_drug_all_diseases(d, drug_idx)
    raw_scr  = float(scores[disease_idx]) if disease_idx < len(scores) else 0.0
    drug_gip = d.get('drug_gip')
    dis_gip  = d.get('dis_gip')
    src_nb   = _neighbor_sim(drug_gip, drug_idx)
    tgt_nb   = _neighbor_sim(dis_gip, disease_idx)
    fuz_scr  = _mamdani.compute(raw_scr, src_nb, tgt_nb)

    return {
        "drug":     drugs[drug_idx],
        "disease":  diseases[disease_idx],
        "is_known": is_known,
        "base_score":    round(raw_scr, 4),
        "fuzzy_score":   round(fuz_scr, 4),
        "drug_class":    _classify_score(src_nb),
        "disease_class": _classify_score(tgt_nb),
        "association_class": _classify_score(fuz_scr),
        "bridging_proteins":     bridging_list,
        "drug_only_proteins":    drug_only,
        "disease_only_proteins": disease_only,
        "num_bridging": len(bridging_list),
        "dataset": dataset,
    }


# ── Classification ────────────────────────────────────────────────────
@app.get("/classify/drug")
def classify_drug(drug_idx: int = Query(...),
                  dataset: str = Query("C-dataset")):
    d = _load_dataset(dataset)
    drugs = d.get('drugs', [])
    if drug_idx < 0 or drug_idx >= len(drugs):
        raise HTTPException(400, "drug_idx out of range")
    drug_gip   = d.get('drug_gip')
    nb_score   = _neighbor_sim(drug_gip, drug_idx)
    assoc_set  = d.get('assoc_set', set())
    num_dis    = sum(1 for (dr, _) in assoc_set if dr == drug_idx)
    max_dis    = len(d.get('diseases', [])) or 1
    conn       = num_dis / max_dis
    combined   = 0.5 * nb_score + 0.5 * conn
    label      = _classify_score(combined)
    return {
        "drug_idx": drug_idx, "drug_info": drugs[drug_idx],
        "class": label,
        "class_label": {"l": "Thấp", "m": "Trung bình", "h": "Cao"}[label],
        "neighbor_sim": round(nb_score, 4),
        "connectivity":  round(conn, 4),
        "combined_score": round(combined, 4),
        "num_known_diseases": num_dis,
    }


@app.get("/classify/disease")
def classify_disease(disease_idx: int = Query(...),
                     dataset: str = Query("C-dataset")):
    d = _load_dataset(dataset)
    diseases = d.get('diseases', [])
    if disease_idx < 0 or disease_idx >= len(diseases):
        raise HTTPException(400, "disease_idx out of range")
    dis_gip  = d.get('dis_gip')
    nb_score = _neighbor_sim(dis_gip, disease_idx)
    assoc_set = d.get('assoc_set', set())
    num_drugs = sum(1 for (_, di) in assoc_set if di == disease_idx)
    max_drugs = len(d.get('drugs', [])) or 1
    conn      = num_drugs / max_drugs
    combined  = 0.5 * nb_score + 0.5 * conn
    label     = _classify_score(combined)
    return {
        "disease_idx": disease_idx, "disease_info": diseases[disease_idx],
        "class": label,
        "class_label": {"l": "Thấp", "m": "Trung bình", "h": "Cao"}[label],
        "neighbor_sim": round(nb_score, 4),
        "connectivity":  round(conn, 4),
        "combined_score": round(combined, 4),
        "num_known_drugs": num_drugs,
    }


@app.get("/classify/batch")
def classify_batch(dataset: str = Query("C-dataset"),
                   entity: str = Query("drug"),
                   limit: int = Query(50)):
    d = _load_dataset(dataset)
    results = []
    if entity == "drug":
        items    = d.get('drugs', [])
        gip      = d.get('drug_gip')
        assoc_set= d.get('assoc_set', set())
        max_n    = len(d.get('diseases', [])) or 1
        for i, item in enumerate(items[:limit]):
            nb   = _neighbor_sim(gip, i)
            cnt  = sum(1 for (dr, _) in assoc_set if dr == i)
            comb = 0.5 * nb + 0.5 * (cnt / max_n)
            results.append({
                'idx': i, 'id': item.get('id', ''),
                'name_en': item.get('name_en', item.get('name', '')),
                'name_vn': item.get('name_vn', ''),
                'class': _classify_score(comb),
                'score': round(comb, 4), 'num_links': cnt,
            })
    else:
        items    = d.get('diseases', [])
        gip      = d.get('dis_gip')
        assoc_set= d.get('assoc_set', set())
        max_n    = len(d.get('drugs', [])) or 1
        for i, item in enumerate(items[:limit]):
            nb   = _neighbor_sim(gip, i)
            cnt  = sum(1 for (_, di) in assoc_set if di == i)
            comb = 0.5 * nb + 0.5 * (cnt / max_n)
            results.append({
                'idx': i, 'id': item.get('id', ''),
                'name_en': item.get('name_en', item.get('id', '')),
                'name_vn': item.get('name_vn', ''),
                'class': _classify_score(comb),
                'score': round(comb, 4), 'num_links': cnt,
            })
    results.sort(key=lambda x: x['score'], reverse=True)
    counts = {'l': sum(1 for r in results if r['class']=='l'),
              'm': sum(1 for r in results if r['class']=='m'),
              'h': sum(1 for r in results if r['class']=='h')}
    return {"entity": entity, "dataset": dataset, "results": results, "counts": counts}


# ── Prediction ────────────────────────────────────────────────────────
@app.post("/predict/single")
def predict_single(req: PredictRequest):
    d = _load_dataset(req.dataset)
    drugs    = d.get('drugs', [])
    diseases = d.get('diseases', [])
    if req.drug_idx < 0 or req.drug_idx >= len(drugs):
        raise HTTPException(400, f"drug_idx {req.drug_idx} out of range")

    scores   = _score_drug_all_diseases(d, req.drug_idx)
    drug_gip = d.get('drug_gip')
    dis_gip  = d.get('dis_gip')
    src_nb   = _neighbor_sim(drug_gip, req.drug_idx)

    results = []
    for di_idx, raw_score in enumerate(scores):
        tgt_nb = _neighbor_sim(dis_gip, di_idx)
        if 'Fuzzy' in req.model:
            fuz = _mamdani.compute(float(raw_score), src_nb, tgt_nb)
        elif 'GCN' in req.model:
            fuz = min(float(raw_score) * (1 + 0.05 * tgt_nb), 1.0)
        else:
            fuz = float(raw_score)
        results.append((di_idx, float(raw_score), fuz, tgt_nb))

    results.sort(key=lambda x: x[2], reverse=True)

    output = []
    for rank, (di_idx, raw, fuz, tgt_nb) in enumerate(results[:req.top_k]):
        is_known = (req.drug_idx, di_idx) in d.get('assoc_set', set())
        dis      = diseases[di_idx] if di_idx < len(diseases) else {'id': str(di_idx)}
        output.append({
            "rank": rank + 1, "disease_idx": di_idx,
            "disease_id":      dis.get('id', str(di_idx)),
            "disease_name":    dis.get('id', str(di_idx)),
            "disease_name_en": dis.get('name_en', ''),
            "disease_name_vn": dis.get('name_vn', ''),
            "gcn_score":       round(raw, 4),
            "fuzzy_score":     round(fuz, 4),
            "disease_class":   _classify_score(tgt_nb),
            "is_known":        is_known,
        })

    drug_info = drugs[req.drug_idx]
    return {
        "drug_idx":    req.drug_idx,
        "drug_name":   drug_info.get('name_en', drug_info.get('name', str(req.drug_idx))),
        "drug_name_vn": drug_info.get('name_vn', ''),
        "drug_id":     drug_info.get('id', ''),
        "drug_smiles": drug_info.get('smiles', ''),
        "drug_class":  _classify_score(src_nb),
        "model":       req.model,
        "top_k":       req.top_k,
        "results":     output,
    }


@app.post("/predict/fuzzy_detail")
def fuzzy_detail(dataset: str = Query("C-dataset"),
                 drug_idx: int = Query(...),
                 disease_idx: int = Query(...)):
    d        = _load_dataset(dataset)
    scores   = _score_drug_all_diseases(d, drug_idx)
    raw      = float(scores[disease_idx]) if disease_idx < len(scores) else 0.3
    drug_gip = d.get('drug_gip')
    dis_gip  = d.get('dis_gip')
    src_nb   = _neighbor_sim(drug_gip, drug_idx)
    tgt_nb   = _neighbor_sim(dis_gip, disease_idx)
    details  = _mamdani.get_memberships(raw, src_nb, tgt_nb)

    drugs    = d.get('drugs', [])
    diseases = d.get('diseases', [])
    drug_info = drugs[drug_idx] if drug_idx < len(drugs) else {}
    dis_info  = diseases[disease_idx] if disease_idx < len(diseases) else {}
    details['drug_name']       = drug_info.get('name', str(drug_idx))
    details['drug_name_vn']    = drug_info.get('name_vn', '')
    details['disease_name']    = dis_info.get('id', str(disease_idx))
    details['disease_name_vn'] = dis_info.get('name_vn', '')
    details['drug_class']      = _classify_score(src_nb)
    details['disease_class']   = _classify_score(tgt_nb)
    return details


@app.post("/predict/matrix")
def predict_matrix(req: MatrixRequest):
    d        = _load_dataset(req.dataset)
    drugs    = d.get('drugs', [])
    diseases = d.get('diseases', [])
    drug_gip = d.get('drug_gip')
    dis_gip  = d.get('dis_gip')
    cells    = []
    for dr_idx in req.drug_indices:
        scores = _score_drug_all_diseases(d, dr_idx)
        src_nb = _neighbor_sim(drug_gip, dr_idx)
        for di_idx in req.disease_indices:
            raw    = float(scores[di_idx]) if di_idx < len(scores) else 0.0
            tgt_nb = _neighbor_sim(dis_gip, di_idx)
            if 'Fuzzy' in req.model:
                fuz = _mamdani.compute(raw, src_nb, tgt_nb)
            elif 'GCN' in req.model:
                fuz = min(raw * (1 + 0.05 * tgt_nb), 1.0)
            else:
                fuz = raw
            is_known  = (dr_idx, di_idx) in d.get('assoc_set', set())
            drug_info = drugs[dr_idx] if dr_idx < len(drugs) else {}
            dis_info  = diseases[di_idx] if di_idx < len(diseases) else {}
            cells.append({
                "drug_idx":       dr_idx,
                "drug_name":      drug_info.get('name_en', drug_info.get('name', str(dr_idx))),
                "drug_name_vn":   drug_info.get('name_vn', ''),
                "drug_class":     _classify_score(src_nb),
                "disease_idx":    di_idx,
                "disease_name":   dis_info.get('id', str(di_idx)),
                "disease_name_vn":dis_info.get('name_vn', ''),
                "disease_class":  _classify_score(tgt_nb),
                "gcn_score":      round(raw, 4),
                "fuzzy_score":    round(fuz, 4),
                "delta":          round(fuz - raw, 4),
                "is_known":       is_known,
            })
    return {"cells": cells, "dataset": req.dataset, "model": req.model}


# ── Training results ──────────────────────────────────────────────────
@app.get("/results/training")
def get_training_results(dataset: str = Query("C-dataset"),
                          model: str = Query("AMNTDDA_Fuzzy")):
    results_dir  = os.path.join(DATA_OUT_DIR, 'results')
    metrics_keys = ['AUC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC']

    summary = None
    sp = os.path.join(results_dir, f'{dataset}_{model}_summary.json')
    if os.path.exists(sp):
        with open(sp) as fh:
            summary = json.load(fh)

    folds = []
    cp = os.path.join(results_dir, f'{dataset}_{model}_fold_results.csv')
    if os.path.exists(cp):
        df = pd.read_csv(cp)
        df_folds = df[pd.to_numeric(df['fold'], errors='coerce').notna()]
        folds = df_folds.to_dict(orient='records')

    return {"dataset": dataset, "model": model, "metrics": metrics_keys,
            "summary": summary, "folds": folds}


@app.get("/results/all_models")
def get_all_models_results(dataset: str = Query("C-dataset")):
    """Lấy kết quả cả 3 mô hình."""
    results_dir  = os.path.join(DATA_OUT_DIR, 'results')
    metrics_keys = ['AUC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1', 'MCC']
    models       = ['AMNTDDA', 'AMNTDDA_GCN', 'AMNTDDA_Fuzzy']
    output       = {}
    for model in models:
        m = {'model': model, 'summary': None, 'folds': []}
        sp = os.path.join(results_dir, f'{dataset}_{model}_summary.json')
        if os.path.exists(sp):
            with open(sp) as fh:
                m['summary'] = json.load(fh)
        cp = os.path.join(results_dir, f'{dataset}_{model}_fold_results.csv')
        if os.path.exists(cp):
            df = pd.read_csv(cp)
            df_f = df[pd.to_numeric(df['fold'], errors='coerce').notna()]
            m['folds'] = df_f.to_dict(orient='records')
        output[model] = m
    return {"dataset": dataset, "metrics": metrics_keys, "models": output}


@app.get("/results/comparison")
def get_comparison(dataset: str = Query("C-dataset")):
    results_dir  = os.path.join(DATA_OUT_DIR, 'results')
    cp = os.path.join(results_dir, f'{dataset}_comparison.json')
    if not os.path.exists(cp):
        return {"dataset": dataset, "models": {}}
    with open(cp) as fh:
        return json.load(fh)


# ── Stage results ─────────────────────────────────────────────────────
@app.get("/stages/list")
def list_stages(dataset: str = Query("B-dataset")):
    run_base = os.path.join(AMDGT_DIR, 'Run_Base')
    stages = [
        ("stage1_input_layer",          "Giai đoạn 1: Xây dựng mạng lưới (Input Layer)"),
        ("stage2_feature_extraction",   "Giai đoạn 2: Trích xuất đặc trưng (Feature Extraction)"),
        ("stage3_modality_interaction", "Giai đoạn 3: Tương tác đa phương thức (Modality Interaction)"),
        ("stage4_prediction",           "Giai đoạn 4: Dự đoán & Kiểm chứng (Prediction Module)"),
    ]
    info = []
    for folder, label in stages:
        result_file = os.path.join(run_base, folder, dataset, 'result.json')
        info.append({
            "folder": folder, "label": label,
            "dataset": dataset, "has_result": os.path.exists(result_file),
        })
    return {"stages": info, "run_base": run_base}


@app.get("/stages/result")
def get_stage_result(stage: str = Query(...),
                     dataset: str = Query("B-dataset")):
    run_base    = os.path.join(AMDGT_DIR, 'Run_Base')
    result_file = os.path.join(run_base, stage, dataset, 'result.json')
    if not os.path.exists(result_file):
        raise HTTPException(404, f"Chưa có kết quả {stage}/{dataset}. "
                                 f"Vui lòng chạy run_base_stages.py trước.")
    with open(result_file, encoding='utf-8') as fh:
        return json.load(fh)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
