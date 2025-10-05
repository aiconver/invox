#!/usr/bin/env python3
"""
MUC-4 evaluator (camelCase schema)

Usage:
  python muc4_eval.py --gold path/to/gold.json --pred path/to/pred.json \
    [--embed_model all-MiniLM-L6-v2] [--pretty] [--show-matches]

Gold (array):
[
  {
    "docid": "DOC_ID",
    "doctext": "...",
    "templates": [{
      "incidentType": "BOMBING",
      "incidentDate": "1989-12-20" | null,
      "incidentLocation": "BOLIVIA: LA PAZ (CITY)",
      "incidentStage": "ACCOMPLISHED",
      "perpetratorIndividual": [...],
      "perpetratorOrganization": [...],
      "target": [...],
      "victim": [...],
      "weapon": [...]
    }]
  }
]

Pred (array):
[
  {
    "id": "DOC_ID",
    "answers": {
      "incidentType": "BOMBING",
      "incidentDate": "1989-12-20" | null,
      "incidentLocation": "LA PAZ",
      "incidentStage": "ACCOMPLISHED",
      "perpetratorIndividual": [...],
      "perpetratorOrganization": [...],
      "target": [...],
      "victim": [...],
      "weapon": [...]
    }
  }
]

pip install sentence-transformers
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers is required. Install with: pip install sentence-transformers")
    sys.exit(1)

# ---- Canonical field spec (camelCase only) ----
FIELD_SPECS = {
    "incidentType":            {"kind": "enum"},
    "incidentDate":            {"kind": "date"},
    "incidentLocation":        {"kind": "text"},
    "incidentStage":           {"kind": "enum"},
    "perpetratorIndividual":   {"kind": "list"},
    "perpetratorOrganization": {"kind": "list"},
    "target":                  {"kind": "list"},
    "victim":                  {"kind": "list"},
    "weapon":                  {"kind": "list"},
}
FIELD_IDS = list(FIELD_SPECS.keys())
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# ------------------------
# Helpers
# ------------------------
def _as_string_list(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        s = v.strip()
        return [s] if s else []
    if isinstance(v, list):
        out: List[str] = []
        for item in v:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    out.append(s)
            elif isinstance(item, list):
                out.extend([str(x).strip() for x in item if isinstance(x, str) and str(x).strip()])
        return out
    return []

def load_gold(path: Path) -> Dict[str, Dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    gold: Dict[str, Dict[str, object]] = {}
    for d in data:
        did = d["docid"]
        tpl = (d.get("templates") or [{}])[0]
        row: Dict[str, object] = {}
        for fid, spec in FIELD_SPECS.items():
            kind = spec["kind"]
            if kind == "list":
                row[fid] = [s for s in _as_string_list(tpl.get(fid)) if s and s != "-"]
            else:
                val = tpl.get(fid)
                if val is None:
                    row[fid] = None
                else:
                    s = str(val).strip()
                    row[fid] = s if s else None
        # validate date
        if isinstance(row.get("incidentDate"), str) and not DATE_RE.match(row["incidentDate"]):  # type: ignore
            row["incidentDate"] = None
        gold[did] = row
    return gold

# change signature:
def load_pred(path: Path) -> Tuple[Dict[str, Dict[str, object]], Dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pred: Dict[str, Dict[str, object]] = {}
    latency_ms: Dict[str, float] = {}

    for d in data:
        ans = d.get("answers", {}) or {}
        row: Dict[str, object] = {}
        for fid, spec in FIELD_SPECS.items():
            v = ans.get(fid)
            if spec["kind"] == "list":
                row[fid] = [s for s in _as_string_list(v) if s]
            else:
                if v is None:
                    row[fid] = None
                else:
                    s = str(v).strip()
                    row[fid] = s if s else None

        # validate date
        if isinstance(row.get("incidentDate"), str) and not DATE_RE.match(row["incidentDate"]):  # type: ignore
            row["incidentDate"] = None

        # ⬇️ capture latency
        lat = d.get("meta", {}).get("timing", {}).get("duration_ms")
        if isinstance(lat, (int, float)) and np.isfinite(lat):
            latency_ms[d["id"]] = float(lat)

        pred[d["id"]] = row

    return pred, latency_ms



def compute_latency_stats(latency_ms: Dict[str, float]) -> Dict[str, float]:
    if not latency_ms:
        return {"count": 0, "mean": 0.0, "median_p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(list(latency_ms.values()), dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median_p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


# ------------------------
# Embedding scorer
# ------------------------
class EmbeddingScorer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", quiet: bool = False):
        if not quiet:
            print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def cosine_similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 1.0 if (not a and not b) else 0.0
        emb = self.model.encode([a, b], normalize_embeddings=True)
        sim = float(np.dot(emb[0], emb[1]))
        return max(0.0, min(1.0, sim))

    def cosine_similarity_matrix(self, list_a: List[str], list_b: List[str]) -> np.ndarray:
        if not list_a or not list_b:
            return np.zeros((len(list_a), len(list_b)), dtype=float)
        emb_a = self.model.encode(list_a, normalize_embeddings=True)
        emb_b = self.model.encode(list_b, normalize_embeddings=True)
        sim = np.dot(emb_a, emb_b.T)
        return np.clip(sim, 0.0, 1.0)

    def best_matches(self, gold_values: List[str], pred_values: List[str]) -> Tuple[List[Tuple[str, str, float]], float]:
        if not gold_values:
            return [], (1.0 if not pred_values else 0.0)
        if not pred_values:
            return [(g, "", 0.0) for g in gold_values], 0.0
        sim = self.cosine_similarity_matrix(gold_values, pred_values)
        matches: List[Tuple[str, str, float]] = []
        for i, g in enumerate(gold_values):
            j = int(np.argmax(sim[i, :]))
            score = float(sim[i, j])
            matches.append((g, pred_values[j], score))
        avg = float(np.mean([m[2] for m in matches])) if matches else 0.0
        return matches, avg


# ------------------------
# Embedding metrics helpers
# ------------------------
def soft_coverage(scorer: "EmbeddingScorer", gold_vals: List[str], pred_vals: List[str]) -> float:
    # gold-anchored: avg of max sim(g -> any pred)
    if not gold_vals:
        return 1.0 if not pred_vals else 0.0
    if not pred_vals:
        return 0.0
    S = scorer.cosine_similarity_matrix(gold_vals, pred_vals)  # G x P
    return float(np.mean(np.max(S, axis=1)))

def soft_specificity(scorer: "EmbeddingScorer", gold_vals: List[str], pred_vals: List[str]) -> float:
    # pred-anchored: avg of max sim(p -> any gold)
    if not pred_vals:
        return 1.0 if not gold_vals else 0.0
    if not gold_vals:
        return 0.0
    S = scorer.cosine_similarity_matrix(pred_vals, gold_vals)  # P x G
    return float(np.mean(np.max(S, axis=1)))

def chamfer_symmetric(scorer: "EmbeddingScorer", gold_vals: List[str], pred_vals: List[str]) -> float:
    cov = soft_coverage(scorer, gold_vals, pred_vals)
    spec = soft_specificity(scorer, gold_vals, pred_vals)
    return (cov + spec) / 2.0


# ------------------------
# Per-field scoring
# ------------------------
def score_enum(gold: Optional[str], pred: Optional[str]) -> float:
    g = (gold or "").strip().upper()
    p = (pred or "").strip().upper()
    if not g and not p:
        return 1.0
    return 1.0 if (g and p and g == p) else 0.0

def score_date(gold: Optional[str], pred: Optional[str]) -> float:
    if not gold and not pred:
        return 1.0
    if gold and pred and gold == pred:
        return 1.0
    return 0.0

def score_text(gold: Optional[str], pred: Optional[str], scorer: EmbeddingScorer) -> float:
    return scorer.cosine_similarity(gold or "", pred or "")

def score_list(gold: List[str], pred: List[str], scorer: EmbeddingScorer, show_matches: bool, doc_id: str, field: str):
    matches, avg = scorer.best_matches(gold, pred)
    if show_matches and (gold or pred):
        print(f"\nDoc {doc_id} | Field {field}")
        if not gold and not pred:
            print("  (both empty)")
        else:
            for g, p, s in matches:
                arrow = "->" if p else "-> [no prediction]"
                print(f"  Gold '{g}' {arrow} '{p}'  (score: {s:.3f})")
            if not gold and pred:
                print("  NOTE: gold empty, predictions present -> field score = 0.0")
            if gold and not pred:
                print("  NOTE: predictions empty -> field score = 0.0")
        print(f"  Field doc-score: {avg:.3f}")
    return avg


def compute_list_metrics(
    gold_vals: List[str],
    pred_vals: List[str],
    scorer: "EmbeddingScorer",
    show_matches: bool,
    doc_id: str,
    field: str,
) -> Tuple[float, float, float, float]:
    """
    Returns:
      avg_best (gold-anchored best-match average)  -- this matches your current list scoring
      coverage (gold-anchored)
      specificity (pred-anchored)
      chamfer (symmetric avg of coverage & specificity)
    """
    # avg_best for backwards-compat is the same as coverage when both non-empty,
    # but we preserve your empty-handling by reusing best_matches().
    matches, avg_best = scorer.best_matches(gold_vals, pred_vals)
    coverage = soft_coverage(scorer, gold_vals, pred_vals)
    specificity = soft_specificity(scorer, gold_vals, pred_vals)
    chamfer = (coverage + specificity) / 2.0

    if show_matches and (gold_vals or pred_vals):
        print(f"\nDoc {doc_id} | Field {field}")
        if not gold_vals and not pred_vals:
            print("  (both empty)")
        else:
            for g, p, s in matches:
                arrow = "->" if p else "-> [no prediction]"
                print(f"  Gold '{g}' {arrow} '{p}'  (score: {s:.3f})")
            if not gold_vals and pred_vals:
                print("  NOTE: gold empty, predictions present -> field score = 0.0")
            if gold_vals and not pred_vals:
                print("  NOTE: predictions empty -> field score = 0.0")
        print(f"  Field doc-score (avg_best ≈ coverage): {avg_best:.3f}")

    return float(avg_best), float(coverage), float(specificity), float(chamfer)


# ------------------------
# Evaluation
# ------------------------
def evaluate(
    gold: Dict[str, Dict[str, object]],
    pred: Dict[str, Dict[str, object]],
    embed_model: str = "all-MiniLM-L6-v2",
    show_matches: bool = False
) -> Dict:
    scorer = EmbeddingScorer(embed_model)
    field_scores: Dict[str, List[float]] = {f: [] for f in FIELD_IDS}
    field_stats: Dict[str, Dict[str, float]] = {}
    field_cov: Dict[str, List[float]] = {f: [] for f in FIELD_IDS}
    field_spec: Dict[str, List[float]] = {f: [] for f in FIELD_IDS}
    field_chamfer: Dict[str, List[float]] = {f: [] for f in FIELD_IDS}

    all_doc_ids = sorted(set(gold.keys()) | set(pred.keys()))
    print(f"Evaluating {len(all_doc_ids)} documents...")

    for doc_id in all_doc_ids:
        gdoc = gold.get(doc_id, {})
        pdoc = pred.get(doc_id, {})

        for fid, spec in FIELD_SPECS.items():
            kind = spec["kind"]
            if kind == "enum":
                s = score_enum(gdoc.get(fid), pdoc.get(fid))  # type: ignore
                field_scores[fid].append(float(s))
                # enums: skip cov/spec/chamfer (or set to None)
            elif kind == "date":
                s = score_date(gdoc.get(fid), pdoc.get(fid))  # type: ignore
                field_scores[fid].append(float(s))
            elif kind == "text":
                # treat text as singleton lists to reuse the same embedding metrics
                gv = [gdoc.get(fid)] if gdoc.get(fid) else []  # type: ignore
                pv = [pdoc.get(fid)] if pdoc.get(fid) else []  # type: ignore
                avg_best, cov, specf, cham = compute_list_metrics(gv, pv, scorer, show_matches, doc_id, fid)
                # keep original scalar score for backwards compat (cosine on strings)
                s_scalar = score_text(gdoc.get(fid), pdoc.get(fid), scorer)  # type: ignore
                field_scores[fid].append(float(s_scalar))
                field_cov[fid].append(cov)
                field_spec[fid].append(specf)
                field_chamfer[fid].append(cham)
            else:
                # 'list'
                gvals = gdoc.get(fid) or []  # type: ignore
                pvals = pdoc.get(fid) or []  # type: ignore
                avg_best, cov, specf, cham = compute_list_metrics(gvals, pvals, scorer, show_matches, doc_id, fid)
                # keep original scalar score (avg_best) for backwards compat
                field_scores[fid].append(float(avg_best))
                field_cov[fid].append(cov)
                field_spec[fid].append(specf)
                field_chamfer[fid].append(cham)


        # existing per-field averages:
    for fid in FIELD_IDS:
        scores = field_scores[fid]
        avg_score = (sum(scores) / len(scores)) if scores else 0.0
        field_stats[fid] = {"average_score": avg_score, "num_documents": len(scores)}

    # NEW: per-field embedding metric averages
    emb_field_stats: Dict[str, Dict[str, float]] = {}
    for fid in FIELD_IDS:
        covs = field_cov[fid]
        specs = field_spec[fid]
        chams = field_chamfer[fid]
        emb_field_stats[fid] = {
            "soft_coverage": (sum(covs) / len(covs)) if covs else 0.0,
            "soft_specificity": (sum(specs) / len(specs)) if specs else 0.0,
            "chamfer_symmetric": (sum(chams) / len(chams)) if chams else 0.0,
            "num_documents": len(covs) if covs else 0  # aligns with lists/text only
        }


    # overall (old)
    all_scores = [s for arr in field_scores.values() for s in arr]
    overall_average = (sum(all_scores) / len(all_scores)) if all_scores else 0.0

    # NEW overall for embedding metrics (only across fields where we computed them)
    all_cov = [x for fid in FIELD_IDS for x in field_cov[fid]]
    all_spec = [x for fid in FIELD_IDS for x in field_spec[fid]]
    all_cham = [x for fid in FIELD_IDS for x in field_chamfer[fid]]
    overall_cov = (sum(all_cov) / len(all_cov)) if all_cov else 0.0
    overall_spec = (sum(all_spec) / len(all_spec)) if all_spec else 0.0
    overall_cham = (sum(all_cham) / len(all_cham)) if all_cham else 0.0


    return {
        "config": {
            "approach": "enum/date exact; location + lists via embeddings",
            "embed_model": embed_model,
            "fields": FIELD_IDS
        },
        "overall_average_score": overall_average,
        "field_scores": field_stats,
        "embedding_metrics": {
            "overall": {
                "soft_coverage": overall_cov,
                "soft_specificity": overall_spec,
                "chamfer_symmetric": overall_cham
            },
            "per_field": emb_field_stats
        },
        "total_comparisons": len(all_scores)
    }


# ------------------------
# CLI
# ------------------------
def main():
    ap = argparse.ArgumentParser(description="MUC-4 evaluator for camelCase schema.")
    ap.add_argument("--gold", required=True, help="Gold JSON")
    ap.add_argument("--pred", required=True, help="Pred JSON")
    ap.add_argument("--embed_model", default="all-MiniLM-L6-v2", help="Sentence-Transformers model")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    ap.add_argument("--show-matches", action="store_true", help="Print gold→best-pred matches for list fields")
    args = ap.parse_args()

    print("Loading gold standard...")
    gold = load_gold(Path(args.gold))

    print("Loading predictions...")
    pred, latency = load_pred(Path(args.pred))

    results = evaluate(gold, pred, embed_model=args.embed_model, show_matches=args.show_matches)

    # ⬇️ attach latency summary
    results["latency_ms"] = compute_latency_stats(latency)

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(json.dumps(results, indent=2 if args.pretty else None))

if __name__ == "__main__":
    main()
