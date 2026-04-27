import os
import requests
from typing import Optional, List, Dict, Any

_BACKEND = os.getenv("BACKEND_URL", "http://localhost:3000/api")
_AI      = os.getenv("AI_ENGINE_URL", "http://localhost:8000")


def _get(url: str, params: dict = None, timeout: int = 10) -> Optional[Dict]:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _post(url: str, body: dict = None, timeout: int = 30) -> Optional[Dict]:
    try:
        r = requests.post(url, json=body or {}, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ── Stats ─────────────────────────────────────────────────────────────
def get_stats(dataset: str = "C-dataset") -> dict:
    return _get(f"{_BACKEND}/stats", {"dataset": dataset}) or {}

def get_global_stats() -> dict:
    return _get(f"{_BACKEND}/stats/global") or {}

def get_datasets() -> List[str]:
    d = _get(f"{_BACKEND}/stats/datasets")
    return d.get("datasets", []) if d else []


# ── Drugs ─────────────────────────────────────────────────────────────
def search_drugs(dataset: str, query: str = "", limit: int = 100) -> List[dict]:
    params = {"dataset": dataset, "limit": limit}
    if query:
        params["search"] = query
    d = _get(f"{_BACKEND}/drugs", params)
    return d.get("drugs", []) if d else []


# ── Diseases ──────────────────────────────────────────────────────────
def search_diseases(dataset: str, query: str = "", limit: int = 100) -> List[dict]:
    params = {"dataset": dataset, "limit": limit}
    if query:
        params["search"] = query
    d = _get(f"{_BACKEND}/diseases", params)
    return d.get("diseases", []) if d else []


# ── Proteins ──────────────────────────────────────────────────────────
def get_proteins(dataset: str, limit: int = 50) -> List[dict]:
    d = _get(f"{_BACKEND}/proteins", {"dataset": dataset, "limit": limit})
    return d.get("proteins", []) if d else []


# ── Predictions ───────────────────────────────────────────────────────
def predict_single(dataset: str, drug_idx: int, model: str = "AMNTDDA_Fuzzy",
                   top_k: int = 10) -> dict:
    return _post(f"{_BACKEND}/predictions/single", {
        "dataset": dataset, "drug_idx": drug_idx,
        "model": model, "top_k": top_k,
    }) or {}


def get_fuzzy_detail(dataset: str, drug_idx: int, disease_idx: int) -> dict:
    return _post(
        f"{_BACKEND}/predictions/fuzzy-detail",
        {},
        # pass as query params via requests
    ) or _post(
        f"{_AI}/predict/fuzzy_detail?dataset={dataset}&drug_idx={drug_idx}&disease_idx={disease_idx}",
        {},
    ) or {}


def predict_matrix(dataset: str, drug_indices: List[int],
                   disease_indices: List[int], model: str = "AMNTDDA_Fuzzy") -> dict:
    return _post(f"{_BACKEND}/predictions/matrix", {
        "dataset": dataset,
        "drug_indices": drug_indices,
        "disease_indices": disease_indices,
        "model": model,
    }) or {}


# ── Comparison ────────────────────────────────────────────────────────
def get_comparison(dataset: str = "C-dataset") -> dict:
    return _get(f"{_BACKEND}/comparison", {"dataset": dataset}) or {}

def compare_matrix(dataset: str, drug_indices: List[int],
                   disease_indices: List[int]) -> dict:
    return _post(f"{_BACKEND}/comparison/matrix", {
        "dataset": dataset,
        "drug_indices": drug_indices,
        "disease_indices": disease_indices,
    }) or {}


# ── Training results ──────────────────────────────────────────────────
def get_training_results(dataset: str = "C-dataset",
                          model: str = "AMNTDDA_Fuzzy") -> dict:
    return _get(f"{_BACKEND}/predictions/results",
                {"dataset": dataset, "model": model}) or {}


# ── History ───────────────────────────────────────────────────────────
def get_history(limit: int = 50) -> List[dict]:
    d = _get(f"{_BACKEND}/history", {"limit": limit})
    return d.get("history", []) if d else []

def save_history(entry: dict) -> dict:
    return _post(f"{_BACKEND}/history", entry) or {}

def clear_history() -> dict:
    try:
        r = requests.delete(f"{_BACKEND}/history", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ── Stage results (Run_Base) — calls AI_ENGINE directly ──────────────
def list_stages(dataset: str = "B-dataset") -> dict:
    return _get(f"{_AI}/stages/list", {"dataset": dataset}) or {}

def get_stage_result(stage: str, dataset: str = "B-dataset") -> dict:
    return _get(f"{_AI}/stages/result", {"stage": stage, "dataset": dataset}) or {}

def get_all_models_results(dataset: str = "C-dataset") -> dict:
    return _get(f"{_AI}/results/all_models", {"dataset": dataset}) or {}

def get_training_results_ai(dataset: str = "C-dataset",
                             model: str = "AMNTDDA_Fuzzy") -> dict:
    return _get(f"{_AI}/results/training", {"dataset": dataset, "model": model}) or {}

def get_drug_network(drug_idx: int, dataset: str = "C-dataset") -> dict:
    return _get(f"{_AI}/network/drug/{drug_idx}", {"dataset": dataset}) or {}

def get_drug_disease_interaction(drug_idx: int, disease_idx: int,
                                  dataset: str = "C-dataset") -> dict:
    return _get(f"{_AI}/network/drug-disease",
                {"drug_idx": drug_idx, "disease_idx": disease_idx, "dataset": dataset}) or {}

def classify_batch(dataset: str, entity: str = "drug", limit: int = 50) -> dict:
    return _get(f"{_AI}/classify/batch",
                {"dataset": dataset, "entity": entity, "limit": limit}) or {}
