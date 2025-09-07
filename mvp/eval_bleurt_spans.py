#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

# -------- BLEURT (Hugging Face) --------
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BleurtScorer:
    """
    Minimal BLEURT scorer using Hugging Face models:
      - Elron/bleurt-base-128   (faster, shorter max length)
      - Elron/bleurt-large-512  (better quality, slower)
    Returns a list of floats (one per pair).
    """
    def __init__(self, model_name: str = "Elron/bleurt-base-128", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def score(self, references: List[str], candidates: List[str]) -> List[float]:
        assert len(references) == len(candidates), "refs and cands must have same length"
        with torch.no_grad():
            inputs = self.tokenizer(
                candidates, references,
                return_tensors="pt",
                padding=True, truncation=True
            ).to(self.device)
            logits = self.model(**inputs).logits.squeeze(-1)
            return logits.detach().cpu().numpy().astype(float).tolist()

# -------- evaluation logic --------
ENTITY_FIELDS = ["PerpInd", "PerpOrg", "Target", "Victim", "Weapon"]

def normalize(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = s.strip("\"'“”‘’`")
    s = re.sub(r"\s+", " ", s)
    return s

def flatten_gold(doc: dict) -> Dict[str, List[str]]:
    """From one gold doc, return {field: [span, ...]}."""
    out = {f: [] for f in ENTITY_FIELDS}
    for t in doc.get("templates", []):
        for field in ENTITY_FIELDS:
            groups = t.get(field, [])
            for group in groups:
                for item in group:
                    if isinstance(item, list) and item:
                        out[field].append(item[0])
    return {k: [normalize(x) for x in v if x] for k, v in out.items()}

def load_gold(path: Path) -> Dict[str, Dict[str, List[str]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {d["docid"]: flatten_gold(d) for d in data}

def load_pred(path: Path) -> Dict[str, Dict[str, List[str]]]:
    """Pred format: [{id, answers: {PerpInd: [...], ...}}]"""
    data = json.loads(path.read_text(encoding="utf-8"))
    pred = {}
    for d in data:
        ans = d.get("answers", {}) or {}
        pred[d["id"]] = {k: [normalize(x) for x in (ans.get(k) or [])] for k in ENTITY_FIELDS}
    return pred

def bleurt_matrix(scorer: BleurtScorer, preds: List[str], golds: List[str]) -> np.ndarray:
    """Return |preds| x |golds| BLEURT matrix."""
    if not preds or not golds:
        return np.zeros((len(preds), len(golds)), dtype=float)
    mat = np.zeros((len(preds), len(golds)), dtype=float)
    for i, p in enumerate(preds):
        scores = scorer.score(references=golds, candidates=[p] * len(golds))
        mat[i, :] = np.array(scores, dtype=float)
    return mat

def match_with_hungarian(score_mat: np.ndarray, threshold: float) -> Tuple[int, int, int, List[float]]:
    """
    One-to-one matching with Hungarian algorithm.
    Count pairs with score >= threshold as TPs.
    """
    if score_mat.size == 0:
        return 0, score_mat.shape[0], score_mat.shape[1], []
    cost = 1.0 - score_mat  # maximize score -> minimize cost
    row_ind, col_ind = linear_sum_assignment(cost)
    tps, tp_scores = 0, []
    matched_pred, matched_gold = set(), set()
    for r, c in zip(row_ind, col_ind):
        s = score_mat[r, c]
        if s >= threshold:
            tps += 1
            tp_scores.append(float(s))
            matched_pred.add(r)
            matched_gold.add(c)
    fps = score_mat.shape[0] - len(matched_pred)
    fns = score_mat.shape[1] - len(matched_gold)
    return tps, fps, fns, tp_scores

def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1

def main():
    ap = argparse.ArgumentParser(description="BLEURT (HF) soft-matching eval for extracted spans.")
    ap.add_argument("--gold", required=True, help="Gold JSON (list of docs with templates).")
    ap.add_argument("--pred", required=True, help="Preds JSON (list of {id, answers}).")
    ap.add_argument("--model", default="Elron/bleurt-large-512",
                    help="HF model id, e.g. Elron/bleurt-base-128 or Elron/bleurt-large-512")
    ap.add_argument("--threshold", type=float, default=0.5, help="BLEURT threshold for a TP.")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    args = ap.parse_args()

    gold = load_gold(Path(args.gold))
    pred = load_pred(Path(args.pred))

    scorer = BleurtScorer(args.model)

    per_class = {f: {"tp": 0, "fp": 0, "fn": 0, "tp_scores": []} for f in ENTITY_FIELDS}
    per_doc = {}
    all_doc_ids = sorted(set(gold) | set(pred))

    for did in all_doc_ids:
        g = gold.get(did, {f: [] for f in ENTITY_FIELDS})
        p = pred.get(did, {f: [] for f in ENTITY_FIELDS})
        doc_stats = {"classes": {}}

        for field in ENTITY_FIELDS:
            g_spans, p_spans = g.get(field, []), p.get(field, [])
            mat = bleurt_matrix(scorer, p_spans, g_spans)
            tp, fp, fn, tp_scores = match_with_hungarian(mat, threshold=args.threshold)

            per_class[field]["tp"] += tp
            per_class[field]["fp"] += fp
            per_class[field]["fn"] += fn
            per_class[field]["tp_scores"].extend(tp_scores)

            p_field, r_field, f1_field = prf(tp, fp, fn)
            doc_stats["classes"][field] = {
                "gold": len(g_spans), "pred": len(p_spans),
                "tp": tp, "fp": fp, "fn": fn,
                "precision": p_field, "recall": r_field, "f1": f1_field,
                "avg_bleurt_tp": (sum(tp_scores)/len(tp_scores)) if tp_scores else 0.0
            }

        per_doc[did] = doc_stats

    # micro/macro
    micro_tp = sum(s["tp"] for s in per_class.values())
    micro_fp = sum(s["fp"] for s in per_class.values())
    micro_fn = sum(s["fn"] for s in per_class.values())
    micro_p, micro_r, micro_f1 = prf(micro_tp, micro_fp, micro_fn)

    macro = {}
    macro_f1s = []
    for field, s in per_class.items():
        p_, r_, f1_ = prf(s["tp"], s["fp"], s["fn"])
        macro[field] = {
            "tp": s["tp"], "fp": s["fp"], "fn": s["fn"],
            "precision": p_, "recall": r_, "f1": f1_,
            "avg_bleurt_tp": (sum(s["tp_scores"])/len(s["tp_scores"])) if s["tp_scores"] else 0.0
        }
        macro_f1s.append(f1_)
    macro_avg_f1 = (sum(macro_f1s)/len(macro_f1s)) if macro_f1s else 0.0

    report = {
        "config": {"model": args.model, "threshold": args.threshold, "fields": ENTITY_FIELDS},
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
        "macro_avg_f1": macro_avg_f1,
        "per_class": macro,
        "per_doc": per_doc
    }

    print(json.dumps(report, indent=2 if args.pretty else None))

if __name__ == "__main__":
    main()
