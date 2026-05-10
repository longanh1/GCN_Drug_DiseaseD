"""
generate_metadata.py — Tạo tệp metadata.json cho mỗi dataset trong AMDGT_main.
Bổ sung tên tiếng Anh, tiếng Việt cho Thuốc, Bệnh, Protein.

Chạy:
    python generate_metadata.py
"""

import os
import json
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data')

# ── Từ điển tên tiếng Việt cho thuốc (tên tiếng Anh → tên tiếng Việt) ──────
DRUG_VN: dict = {
    # Anticonvulsants / Antiepileptics
    "clobazam":         "Clobazam (chống co giật)",
    "lamotrigine":      "Lamotrigine (chống động kinh)",
    # Anti-cancer
    "vinorelbine":      "Vinorelbine (chống ung thư)",
    "gemcitabine":      "Gemcitabine (chống ung thư)",
    "docetaxel":        "Docetaxel (chống ung thư)",
    "gefitinib":        "Gefitinib (ung thư phổi)",
    "sorafenib":        "Sorafenib (ung thư gan/thận)",
    "sunitinib":        "Sunitinib (chống ung thư)",
    "imatinib mesylate":"Imatinib (CML/GIST)",
    "dasatinib":        "Dasatinib (CML)",
    "erlotinib hydrochloride": "Erlotinib (ung thư phổi)",
    "alitretinoin":     "Alitretinoin (sarcom Kaposi)",
    "amsacrine":        "Amsacrin (bệnh bạch cầu)",
    # Cardiovascular
    "carvedilol":       "Carvedilol (suy tim/tăng huyết áp)",
    "benazepril":       "Benazepril (hạ huyết áp)",
    "trandolapril":     "Trandolapril (hạ huyết áp)",
    "clopidogrel":      "Clopidogrel (kháng tiểu cầu)",
    "candesartan":      "Candesartan (hạ huyết áp)",
    "telmisartan":      "Telmisartan (hạ huyết áp)",
    "valsartan":        "Valsartan (hạ huyết áp)",
    "bosentan":         "Bosentan (tăng áp phổi)",
    "sildenafil citrate":"Sildenafil (rối loạn cương/tăng áp phổi)",
    "acebutolol":       "Acebutolol (chẹn beta)",
    "amiodarone":       "Amiodarone (rối loạn nhịp tim)",
    # Statins
    "fluvastatin":      "Fluvastatin (hạ mỡ máu)",
    "cerivastatin":     "Cerivastatin (hạ mỡ máu)",
    "rosuvastatin calcium": "Rosuvastatin (hạ mỡ máu)",
    "atorvastatin calcium": "Atorvastatin (hạ mỡ máu)",
    "ezetimibe":        "Ezetimibe (hạ cholesterol)",
    # Anti-inflammatory / NSAID
    "leflunomide":      "Leflunomide (viêm khớp dạng thấp)",
    "meloxicam":        "Meloxicam (chống viêm NSAID)",
    "celecoxib":        "Celecoxib (chống viêm COX-2)",
    "rofecoxib":        "Rofecoxib (chống viêm COX-2)",
    "aspirin":          "Aspirin (giảm đau, kháng đông)",
    # Antidiabetics
    "troglitazone":     "Troglitazone (tiểu đường typ 2)",
    "pioglitazone":     "Pioglitazone (tiểu đường typ 2)",
    "rosiglitazone":    "Rosiglitazone (tiểu đường typ 2)",
    "sitagliptin phosphate": "Sitagliptin (tiểu đường typ 2)",
    # Antihistamines / Leukotriene modulators
    "zafirlukast":      "Zafirlukast (hen suyễn)",
    "montelukast":      "Montelukast (hen suyễn)",
    "fexofenadine":     "Fexofenadine (chống dị ứng)",
    "desloratadine":    "Desloratadine (chống dị ứng)",
    # Immunosuppressants
    "mycophenolate mofetil": "Mycophenolate mofetil (ức chế miễn dịch)",
    "cyclosporine":     "Cyclosporin (ức chế miễn dịch)",
    # Antidepressants / Antipsychotics
    "nefazodone":       "Nefazodone (chống trầm cảm)",
    "olanzapine":       "Olanzapine (chống loạn thần)",
    "venlafaxine hydrochloride": "Venlafaxine (chống trầm cảm)",
    "quetiapine fumarate": "Quetiapine (chống loạn thần)",
    "amitriptyline":    "Amitriptyline (chống trầm cảm ba vòng)",
    # Antiparkinsonian
    "amantadine":       "Amantadine (Parkinson/cúm)",
    "biperiden":        "Biperiden (Parkinson)",
    # Antimicrobials
    "acetazolamide":    "Acetazolamide (lợi tiểu/tăng nhãn áp)",
    # Hormones / Vitamins
    "goserelin":        "Goserelin (ung thư tuyến tiền liệt/vú)",
    "desmopressin":     "Desmopressin (đái tháo nhạt)",
    "cyclosporine":     "Cyclosporin (ức chế miễn dịch)",
    "calcitriol":       "Calcitriol (vitamin D3)",
    "ergocalciferol":   "Ergocalciferol (vitamin D2)",
    "cholecalciferol":  "Cholecalciferol (vitamin D3)",
    "riboflavin":       "Riboflavin (vitamin B2)",
    "folic acid":       "Acid folic (vitamin B9)",
    "thiamine":         "Thiamine (vitamin B1)",
    "pyridoxine":       "Pyridoxine (vitamin B6)",
    "ascorbic acid":    "Acid ascorbic (vitamin C)",
    "vitamin a":        "Vitamin A (retinol)",
    "vitamin e":        "Vitamin E (tocopherol)",
    "arginine":         "Arginine (amino acid)",
    "alanine":          "Alanine (amino acid)",
    "isoleucine":       "Isoleucine (amino acid)",
    # Other
    "acetaminophen":    "Acetaminophen/Paracetamol (giảm đau)",
    "allopurinol":      "Allopurinol (gout)",
    "bezafibrate":      "Bezafibrate (hạ mỡ máu)",
    "adenosine":        "Adenosine (rối loạn nhịp tim)",
    "amiloride":        "Amiloride (lợi tiểu)",
    "amphetamine":      "Amphetamine (ADHD/narcolepsy)",
    "antipyrine":       "Antipyrin (giảm đau/hạ sốt)",
    "betamethasone":    "Betamethasone (corticosteroid)",
    "leuprolide":       "Leuprolide (ung thư tuyến tiền liệt/vú)",
    "sermorelin":       "Sermorelin (thiếu GH)",
    "octreotide":       "Octreotide (to đầu chi/ung thư thần kinh nội tiết)",
    "vasopressin":      "Vasopressin (đái tháo nhạt)",
    "choline":          "Choline (bổ sung dinh dưỡng)",
    "pravastatin":      "Pravastatin (hạ mỡ máu)",
    "fluvoxamine":      "Fluvoxamine (chống trầm cảm SSRI)",
    "ramipril":         "Ramipril (hạ huyết áp)",
    "masoprocol":       "Masoprocol (kem bôi da)",
    "flunisolide":      "Flunisolide (corticosteroid hít)",
}

# ── Từ điển tên tiếng Việt cho bệnh ────────────────────────────────────────
DISEASE_VN: dict = {
    "anxiety disorders":                    "Rối loạn lo âu",
    "ataxia":                               "Mất điều hòa vận động",
    "constriction, pathologic":             "Co thắt bệnh lý",
    "epilepsy, tonic-clonic":              "Động kinh toàn thể co giật",
    "glycosuria":                           "Tiểu đường niệu",
    "disorders of excessive somnolence":    "Rối loạn buồn ngủ quá mức",
    "hypothyroidism":                       "Suy giáp",
    "myoclonus":                            "Giật cơ",
    "paresis":                              "Liệt nhẹ",
    "proteinuria":                          "Protein niệu",
    "seizures":                             "Cơn động kinh",
    "stevens-johnson syndrome":             "Hội chứng Stevens-Johnson",
    "substance withdrawal syndrome":        "Hội chứng cai nghiện",
    "hypophosphatemia":                     "Giảm phospho máu",
    "neurobehavioral manifestations":       "Biểu hiện thần kinh hành vi",
    "vasospasm, intracranial":              "Co thắt mạch máu não",
    "adenocarcinoma of lung":               "Ung thư biểu mô tuyến phổi",
    "adenocarcinoma":                       "Ung thư biểu mô tuyến",
    "agranulocytosis":                      "Mất bạch cầu hạt",
    "alopecia":                             "Rụng tóc (hói đầu)",
    "anemia":                               "Thiếu máu",
    "angina pectoris":                      "Đau thắt ngực",
    "anorexia":                             "Chán ăn",
    "asthenia":                             "Suy nhược cơ thể",
    "atrial fibrillation":                  "Rung tâm nhĩ",
    "atrial flutter":                       "Cuồng tâm nhĩ",
    "autonomic nervous system diseases":    "Bệnh hệ thần kinh tự chủ",
    "breast neoplasms":                     "Ung thư vú",
    "carcinoma":                            "Ung thư biểu mô",
    "carcinoma, non-small-cell lung":       "Ung thư phổi không tế bào nhỏ",
    "cardiac arrhythmias":                  "Rối loạn nhịp tim",
    "cardiovascular diseases":              "Bệnh tim mạch",
    "colorectal neoplasms":                 "Ung thư đại trực tràng",
    "coronary artery disease":              "Bệnh động mạch vành",
    "coronary disease":                     "Bệnh động mạch vành",
    "crohn disease":                        "Bệnh Crohn",
    "depression":                           "Trầm cảm",
    "diabetes mellitus":                    "Đái tháo đường",
    "diabetes mellitus, type 2":            "Đái tháo đường typ 2",
    "diarrhea":                             "Tiêu chảy",
    "digestive system diseases":            "Bệnh tiêu hóa",
    "drug-induced liver injury":            "Tổn thương gan do thuốc",
    "edema":                                "Phù",
    "exanthema":                            "Phát ban",
    "fatigue":                              "Mệt mỏi",
    "gastrointestinal diseases":            "Bệnh đường tiêu hóa",
    "gout":                                 "Gout (thống phong)",
    "headache":                             "Đau đầu",
    "heart failure":                        "Suy tim",
    "hepatitis":                            "Viêm gan",
    "hepatotoxicity":                       "Độc tính với gan",
    "hyperglycemia":                        "Tăng đường huyết",
    "hyperlipidemia":                       "Tăng lipid máu",
    "hypertension":                         "Tăng huyết áp",
    "hypertriglyceridemia":                 "Tăng triglyceride máu",
    "hypotension":                          "Hạ huyết áp",
    "hypothyroidism":                       "Suy giáp",
    "immunosuppression":                    "Ức chế miễn dịch",
    "infection":                            "Nhiễm trùng",
    "inflammation":                         "Viêm",
    "kidney diseases":                      "Bệnh thận",
    "leukemia":                             "Bệnh bạch cầu",
    "leukemia, myeloid":                    "Bạch cầu tủy",
    "liver diseases":                       "Bệnh gan",
    "lung neoplasms":                       "Ung thư phổi",
    "lymphoma":                             "U lympho",
    "mental disorders":                     "Rối loạn tâm thần",
    "metabolic diseases":                   "Bệnh chuyển hóa",
    "multiple myeloma":                     "Đa u tủy xương",
    "myocardial infarction":                "Nhồi máu cơ tim",
    "myopathy":                             "Bệnh cơ",
    "nausea":                               "Buồn nôn",
    "neoplasms":                            "Ung thư (khối u)",
    "nephrotoxicity":                       "Độc tính với thận",
    "nervous system diseases":              "Bệnh hệ thần kinh",
    "neutropenia":                          "Giảm bạch cầu trung tính",
    "obesity":                              "Béo phì",
    "osteoporosis":                         "Loãng xương",
    "pain":                                 "Đau",
    "parkinson disease":                    "Bệnh Parkinson",
    "peripheral nervous system diseases":   "Bệnh hệ thần kinh ngoại biên",
    "pneumonia":                            "Viêm phổi",
    "prostatic neoplasms":                  "Ung thư tuyến tiền liệt",
    "psoriasis":                            "Vảy nến",
    "pulmonary hypertension":               "Tăng áp phổi",
    "renal insufficiency":                  "Suy thận",
    "respiratory tract diseases":           "Bệnh đường hô hấp",
    "rheumatoid arthritis":                 "Viêm khớp dạng thấp",
    "schizophrenia":                        "Tâm thần phân liệt",
    "skin diseases":                        "Bệnh da liễu",
    "sleep disorders":                      "Rối loạn giấc ngủ",
    "stomach neoplasms":                    "Ung thư dạ dày",
    "stroke":                               "Đột quỵ não",
    "systemic lupus erythematosus":         "Lupus ban đỏ hệ thống",
    "thrombocytopenia":                     "Giảm tiểu cầu",
    "thrombosis":                           "Huyết khối",
    "thyroid diseases":                     "Bệnh tuyến giáp",
    "ulcer":                                "Loét",
    "urinary tract infections":             "Nhiễm trùng tiết niệu",
    "vomiting":                             "Nôn mửa",
    "weight gain":                          "Tăng cân",
    "weight loss":                          "Giảm cân",
}

# ── Từ điển gen/tên protein cho UniProt IDs phổ biến ────────────────────────
PROTEIN_INFO: dict = {
    "P22303": {"gene": "ACHE",   "name_en": "Acetylcholinesterase",        "name_vn": "Acetylcholinesterase"},
    "P06276": {"gene": "BCHE",   "name_en": "Butyrylcholinesterase",       "name_vn": "Butyrylcholinesterase"},
    "P36544": {"gene": "CHRNA7", "name_en": "Neuronal acetylcholine receptor alpha-7", "name_vn": "Thụ thể nicotinic acetylcholine α-7"},
    "O15244": {"gene": "SLC22A6","name_en": "Solute carrier family 22 member 6","name_vn": "Vận chuyển hữu cơ anion OAT1"},
    "O15245": {"gene": "SLC22A7","name_en": "Solute carrier family 22 member 7","name_vn": "Vận chuyển hữu cơ anion OAT2"},
    "O75751": {"gene": "SLC22A8","name_en": "Solute carrier family 22 member 8","name_vn": "Vận chuyển hữu cơ anion OAT3"},
    "O76082": {"gene": "SLC22A9","name_en": "Solute carrier family 22 member 9","name_vn": "Vận chuyển hữu cơ anion OAT4"},
    "Q9H015": {"gene": "SLC22A11","name_en": "Solute carrier family 22 member 11","name_vn": "Vận chuyển hữu cơ anion OAT4b"},
    "Q99707": {"gene": "MCM2",   "name_en": "DNA replication licensing factor MCM2","name_vn": "Yếu tố tái bản DNA MCM2"},
    "P50579": {"gene": "EIF2A",  "name_en": "Eukaryotic translation initiation factor 2A","name_vn": "Nhân tố khởi đầu dịch mã 2A"},
    "P42898": {"gene": "MTFMT",  "name_en": "Mitochondrial methionyl-tRNA formyltransferase","name_vn": "Enzyme formyl hóa methionyl-tRNA ty thể"},
    "P15104": {"gene": "GLUL",   "name_en": "Glutamine synthetase",        "name_vn": "Glutamine synthetase"},
    "Q8TF71": {"gene": "SLC22A20","name_en": "Solute carrier family 22 member 20","name_vn": "Vận chuyển hữu cơ anion 20"},
    "P04035": {"gene": "HMGCR",  "name_en": "3-Hydroxy-3-methylglutaryl-CoA reductase","name_vn": "HMG-CoA reductase (đích statin)"},
    "P10632": {"gene": "CYP2C8", "name_en": "Cytochrome P450 2C8",         "name_vn": "Cytochrome P450 2C8"},
    "Q9Y6L6": {"gene": "SLC10A2","name_en": "Ileal sodium/bile acid cotransporter","name_vn": "Vận chuyển acid mật hồi tràng"},
    "O94956": {"gene": "SLC22A5","name_en": "Sodium-dependent carnitine transporter OCTN2","name_vn": "Vận chuyển carnitine OCTN2"},
    "P08183": {"gene": "ABCB1",  "name_en": "P-glycoprotein 1 (MDR1)",     "name_vn": "P-glycoprotein 1 (kháng đa thuốc)"},
    "P46721": {"gene": "SLC10A1","name_en": "Sodium/bile acid cotransporter","name_vn": "Vận chuyển acid mật gan NTCP"},
}


def get_drug_vn(name_en: str) -> str:
    key = name_en.lower().strip()
    return DRUG_VN.get(key, name_en)


def get_disease_vn(name_en: str) -> str:
    key = name_en.lower().strip().strip('"')
    return DISEASE_VN.get(key, name_en.replace('"', '').title())


def generate_metadata_for_dataset(dataset: str):
    base = os.path.join(DATA_DIR, dataset)
    if not os.path.isdir(base):
        print(f"Dataset {dataset} không tìm thấy.")
        return

    meta = {"drugs": [], "diseases": [], "proteins": []}

    # ── Drugs ─────────────────────────────────────────────────────────────
    # B-dataset uses (name, id, smiles), F-dataset has index col
    drug_csv = os.path.join(base, 'DrugInformation.csv')
    if os.path.exists(drug_csv):
        df_drug = pd.read_csv(drug_csv)
        # Normalise columns
        df_drug.columns = [c.lower().strip() for c in df_drug.columns]
        # Drop unnamed index column
        df_drug = df_drug[[c for c in df_drug.columns if not c.startswith('unnamed')]]
        for idx, row in df_drug.iterrows():
            name_en = str(row.get('name', row.get('id', str(idx)))).strip()
            drug_id = str(row.get('id', '')).strip()
            smiles  = str(row.get('smiles', '')).strip()
            meta['drugs'].append({
                "idx":     int(idx),
                "id":      drug_id,
                "name_en": name_en,
                "name_vn": get_drug_vn(name_en),
                "smiles":  smiles,
            })

    # ── Diseases ──────────────────────────────────────────────────────────
    dis_csv = os.path.join(base, 'DiseaseFeature.csv')
    if os.path.exists(dis_csv):
        df_dis = pd.read_csv(dis_csv, header=None)
        for idx, row in df_dis.iterrows():
            dis_id = str(row.iloc[0]).strip().strip('"')
            meta['diseases'].append({
                "idx":     int(idx),
                "id":      dis_id,
                "name_en": dis_id.replace('"', '').title(),
                "name_vn": get_disease_vn(dis_id),
            })

    # ── Proteins ──────────────────────────────────────────────────────────
    prot_csv = os.path.join(base, 'ProteinInformation.csv')
    if os.path.exists(prot_csv):
        df_prot = pd.read_csv(prot_csv)
        df_prot.columns = [c.lower().strip() for c in df_prot.columns]
        for idx, row in df_prot.iterrows():
            prot_id  = str(row.iloc[0]).strip()
            sequence = str(row.get('sequence', '')).strip() if 'sequence' in df_prot.columns else ''
            info     = PROTEIN_INFO.get(prot_id, {})
            meta['proteins'].append({
                "idx":      int(idx),
                "id":       prot_id,
                "gene":     info.get("gene", prot_id),
                "name_en":  info.get("name_en", prot_id),
                "name_vn":  info.get("name_vn", info.get("name_en", prot_id)),
                "seq_len":  len(sequence),
            })

    # ── Lưu ──────────────────────────────────────────────────────────────
    out_path = os.path.join(base, 'metadata.json')
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print(f"[{dataset}] metadata → {out_path}")
    print(f"  Drugs: {len(meta['drugs'])}, Diseases: {len(meta['diseases'])}, Proteins: {len(meta['proteins'])}")
    return meta


if __name__ == '__main__':
    for ds in ['B-dataset', 'C-dataset', 'F-dataset']:
        generate_metadata_for_dataset(ds)
    print("\nHoàn thành tạo metadata!")
