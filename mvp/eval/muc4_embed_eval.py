#!/usr/bin/env python3
"""
MUC-4 embedding-only evaluator (single-file, no samples)

Usage:
  python muc4_embed_eval.py --gold path/to/gold.json --pred path/to/pred.json \
    [--embed_model all-MiniLM-L6-v2] [--pretty] [--show-matches]

Gold format (array of docs):
[
  {
    "docid": "DOC_ID",
    "doctext": "...",
    "templates": [
      {
        "incident_type": "...",
        "PerpInd": [...],
        "PerpOrg": [...],
        "Target": [...],
        "Victim": [...],
        "Weapon": [...]
      }
    ]
  }
]

Pred format (array of preds):
[
  {
    "id": "DOC_ID",
    "answers": {
      "PerpInd": [...],
      "PerpOrg": [...],
      "Target": [...],
      "Victim": [...],
      "Weapon": [...]
    }
  }
]

Install dependency:
  pip install sentence-transformers
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Embeddings dependency
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers is required. Install with: pip install sentence-transformers")
    sys.exit(1)

ENTITY_FIELDS = ["PerpInd", "PerpOrg", "Target", "Victim", "Weapon"]

# ------------------------
# Helpers for IO & parsing
# ------------------------
def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return s.strip()

def _as_string_list(v) -> List[str]:
    """Accepts strings, lists of strings, or nested lists; flattens into a list of strings."""
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        out = []
        for item in v:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, list) and item:
                if isinstance(item[0], str):
                    out.append(item[0])
                else:
                    out.extend([x for x in item if isinstance(x, str)])
        return out
    return []

def flatten_gold(doc: dict) -> Dict[str, List[str]]:
    """Extract normalized gold values from a gold document."""
    out = {f: [] for f in ENTITY_FIELDS}
    for t in doc.get("templates", []):
        for field in ENTITY_FIELDS:
            out[field].extend(_as_string_list(t.get(field)))

    # Normalize, drop empty & "-"
    return {
        k: [normalize_text(x) for x in v
            if normalize_text(x) and normalize_text(x) != "-"]
        for k, v in out.items()
    }

def load_gold(path: Path) -> Dict[str, Dict[str, List[str]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    gold = {}
    for d in data:
        did = d["docid"]
        gold[did] = flatten_gold(d)
    return gold

def load_pred(path: Path) -> Dict[str, Dict[str, List[str]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pred = {}
    for d in data:
        ans = d.get("answers", {}) or {}
        pred[d["id"]] = {
            k: [normalize_text(x) for x in (ans.get(k) or []) if normalize_text(x)]
            for k in ENTITY_FIELDS
        }
    return pred

# ------------------------
# Embedding scorer
# ------------------------
class EmbeddingScorer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", quiet: bool = False):
        if not quiet:
            print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def cosine_similarity_matrix(self, list_a: List[str], list_b: List[str]) -> np.ndarray:
        if not list_a or not list_b:
            return np.zeros((len(list_a), len(list_b)), dtype=float)
        emb_a = self.model.encode(list_a, normalize_embeddings=True)
        emb_b = self.model.encode(list_b, normalize_embeddings=True)
        sim = np.dot(emb_a, emb_b.T)
        return np.clip(sim, 0.0, 1.0)

    def best_matches(self, gold_values: List[str], pred_values: List[str]) -> Tuple[List[Tuple[str, str, float]], float]:
        """
        Returns list of (gold, best_pred, score) and the average of the best scores.
        Scoring logic:
          - If no gold: return 1.0 if also no pred else 0.0
          - If gold present and no pred: 0.0
          - Else: for each gold row, take max cosine to any pred.
        """
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
# Evaluation
# ------------------------
def evaluate_embedding_only(
    gold: Dict[str, Dict[str, List[str]]],
    pred: Dict[str, Dict[str, List[str]]],
    embed_model: str = "all-MiniLM-L6-v2",
    show_matches: bool = False
) -> Dict:
    scorer = EmbeddingScorer(embed_model)
    field_scores: Dict[str, List[float]] = {f: [] for f in ENTITY_FIELDS}
    field_stats: Dict[str, Dict[str, float]] = {}

    all_doc_ids = sorted(set(gold.keys()) | set(pred.keys()))
    print(f"Evaluating {len(all_doc_ids)} documents...")

    for doc_id in all_doc_ids:
        gold_doc = gold.get(doc_id, {f: [] for f in ENTITY_FIELDS})
        pred_doc = pred.get(doc_id, {f: [] for f in ENTITY_FIELDS})

        for field in ENTITY_FIELDS:
            gvals = gold_doc.get(field, [])
            pvals = pred_doc.get(field, [])
            matches, avg = scorer.best_matches(gvals, pvals)
            field_scores[field].append(avg)

            if show_matches and (gvals or pvals):
                print(f"\nDoc {doc_id} | Field {field}")
                if not gvals and not pvals:
                    print("  (both empty)")
                else:
                    for g, p, s in matches:
                        arrow = "->" if p else "-> [no prediction]"
                        print(f"  Gold '{g}' {arrow} '{p}'  (score: {s:.3f})")
                    if not gvals and pvals:
                        print("  NOTE: gold empty, predictions present -> field score = 0.0")
                    if gvals and not pvals:
                        print("  NOTE: predictions empty -> field score = 0.0")
                print(f"  Field doc-score: {avg:.3f}")

    # Aggregate per-field
    for field in ENTITY_FIELDS:
        scores = field_scores[field]
        avg_score = (sum(scores) / len(scores)) if scores else 0.0
        field_stats[field] = {"average_score": avg_score, "num_documents": len(scores)}

    # Overall
    all_scores = [s for arr in field_scores.values() for s in arr]
    overall_average = (sum(all_scores) / len(all_scores)) if all_scores else 0.0

    return {
        "config": {
            "approach": "embedding_only_best_match",
            "embed_model": embed_model,
            "description": "For each gold value, find best matching prediction using embedding similarity, then average all scores"
        },
        "overall_average_score": overall_average,
        "field_scores": field_stats,
        "total_comparisons": len(all_scores)
    }

# ------------------------
# CLI
# ------------------------
def main():
    ap = argparse.ArgumentParser(description="Simplified embedding-only MUC-4 evaluator (no samples)")
    ap.add_argument("--gold", required=True, help="Gold standard JSON file")
    ap.add_argument("--pred", required=True, help="Predictions JSON file")
    ap.add_argument("--embed_model", default="all-MiniLM-L6-v2", help="Sentence-Transformers model name")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    ap.add_argument("--show-matches", action="store_true", help="Print gold→best-pred matches for each doc/field")
    args = ap.parse_args()

    print("Loading gold standard...")
    gold = load_gold(Path(args.gold))
    print("Loading predictions...")
    pred = load_pred(Path(args.pred))

    results = evaluate_embedding_only(gold, pred, embed_model=args.embed_model, show_matches=args.show_matches)

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(json.dumps(results, indent=2 if args.pretty else None))

if __name__ == "__main__":
    main()
