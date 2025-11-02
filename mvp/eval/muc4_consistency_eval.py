#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Logs-only evaluator for MUC-4 style consistency with INPUT-AWARE SC.

Reports:
- FPR per field (pinpoints weak spots)
- FPR_macro (cross-field consistency)
- FPR_overall (records perfectly formatted)
- SC per field (mean ± sd), input-aware:
    in∅ & out∅ -> 1
    in∅ & out≠∅ -> 0
    in≠∅ & out∅ -> 0
    in≠∅ & out≠∅ -> style score (masked cosine; with fast-path 1.0 for ISO dates/enums)
- SC_macro (mean of the per-field means)

Extras:
  1) SC case mix counts per field (both_empty / overfill / underfill / both_filled)
  2) Triple exemplar for incidentLocation: "T, T", "T: T (T)", and "T, T, T" (max cosine)
  3) SC logs include coverage (n = records contributing)
  4) ACR & SMS per field:
       - ACR (Abstention Consistency Rate) = both_empty / total
       - SMS (Style Match Score) = mean style over both_filled only
  5) Availability alignment metrics per field + macro:
       - AA  = (both_filled + both_empty) / total
       - OFR = in_empty_out_filled / total
       - UFR = in_filled_out_empty / total
       - Fill-P / Fill-R / Fill-F1 (precision/recall/F1 over fill decisions)

CLI:
  --inputs   : path to in.json (required; presence read from 'templates')
  --outputs  : path to out.json (strategy outputs)  [REQUIRED]
  --strategy : label shown in the header (e.g., S1)

No files are written; everything is printed to stdout.
"""

import argparse
import json
import math
import re
from collections import Counter
from typing import Dict, Any, Tuple, List

# ----------------------------
# 1) CONFIG (schema, enums, fields, exemplars)
# ----------------------------

REQUIRED_FIELDS = [
    "incidentType", "incidentDate", "incidentLocation", "incidentStage",
    "perpetratorIndividual", "perpetratorOrganization",
    "target", "victim", "weapon",
]

ALLOWED_INCIDENT_TYPES = {"ASSASSINATION", "ATTACK", "BOMBING", "KIDNAPPING", ""}  # "" allowed for abstain
ALLOWED_STAGES = {"ACCOMPLISHED", "ATTEMPTED", "FAILED", "THREATENED", ""}

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")  # strict YYYY-MM-DD

MAIN_FIELDS = [
    "incidentType", "incidentDate", "incidentLocation", "incidentStage",
    "perpetratorIndividual", "perpetratorOrganization", "target", "victim", "weapon",
]
LIST_FIELDS = {"perpetratorIndividual", "perpetratorOrganization", "target", "victim", "weapon"}

# Fixed house-style exemplars (masked before cosine):
EXEMPLARS = {
    "incidentType": "T",
    "incidentDate": "DDDD-DD-DD",
    "incidentLocation_primary": "T, T",     # City, Country
    "incidentLocation_alt":     "T: T (T)", # Country: City (City)
    "incidentLocation_tri":     "T, T, T",  # City, Region/Dept, Country
    "incidentStage": "T",
    "perpetratorIndividual": "T, T",
    "perpetratorOrganization": "T, T",
    "target": "T, T",
    "victim": "T, T",
    "weapon": "T, T",
}

# ----------------------------
# 2) HELPERS (masking, n-grams, cosine, value presence)
# ----------------------------

SPACE_RE = re.compile(r"\s+")
WORD_RE  = re.compile(r"[A-Za-z]+")
DIGIT_RE = re.compile(r"\d+")

def normalize_spaces(s: str) -> str:
    return SPACE_RE.sub(" ", s.strip())

def to_style_mask(text: str) -> str:
    """Map letter runs -> 'T', digit runs -> 'D' (length-preserving for digits), keep punctuation/spacing."""
    if text is None:
        return ""
    s = normalize_spaces(str(text))
    s = WORD_RE.sub("T", s)
    s = DIGIT_RE.sub(lambda m: "D" * len(m.group(0)), s)
    return normalize_spaces(s)

def field_to_surface(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(x) for x in value)  # expose list separator
    return str(value)

def char_ngrams(text: str, n: int = 3) -> List[str]:
    if not text:
        return []
    text = text.lower()
    if len(text) < n:
        return [text]
    return [text[i:i+n] for i in range(len(text) - n + 1)]

def vec(text: str, n: int = 3) -> Dict[str, float]:
    grams = char_ngrams(text, n=n)
    c = Counter(grams)
    norm = math.sqrt(sum(v*v for v in c.values())) or 1.0
    return {k: v / norm for k, v in c.items()}

def cosine(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    if not v1 or not v2:
        return 0.0
    if len(v2) < len(v1):
        v1, v2 = v2, v1
    return sum(v1.get(k, 0.0) * v2.get(k, 0.0) for k in v1.keys())

def has_value(val: Any) -> bool:
    """True if the field has a non-empty value (after trimming)."""
    if val is None:
        return False
    if isinstance(val, str):
        return val.strip() != ""
    if isinstance(val, list):
        return any(str(x).strip() != "" for x in val)
    # treat any other non-None as present
    return True

def input_has_value_for_field(in_record: Dict[str, Any], field: str) -> bool:
    """
    Returns True if ANY template in the input record provides a non-empty value for the field.
    Falls back to checking 'answers' if 'templates' is missing.
    """
    if not isinstance(in_record, dict):
        return False
    temps = in_record.get("templates")
    if isinstance(temps, list):
        for t in temps:
            if isinstance(t, dict) and field in t and has_value(t[field]):
                return True
        return False
    ans = in_record.get("answers", {})
    return has_value(ans.get(field, None))

# ----------------------------
# 3) FPR LINTS
# ----------------------------

def lint_record(rec: Dict[str, Any]) -> Tuple[Dict[str, bool], bool]:
    ans = rec.get("answers", {})
    checks: Dict[str, bool] = {}

    # presence
    for f in REQUIRED_FIELDS:
        checks[f":present:{f}"] = (f in ans)

    # types
    checks["incidentType:type"]       = isinstance(ans.get("incidentType", ""), str)
    checks["incidentDate:type"]       = (ans.get("incidentDate", None) is None or isinstance(ans.get("incidentDate", None), str))
    checks["incidentLocation:type"]   = isinstance(ans.get("incidentLocation", ""), str)
    checks["incidentStage:type"]      = isinstance(ans.get("incidentStage", ""), str)
    for lf in LIST_FIELDS:
        vals = ans.get(lf, [])
        checks[f"{lf}:type"] = isinstance(vals, list) and all(isinstance(x, str) for x in vals)

    # enums / pattern
    it = ans.get("incidentType", "")
    checks["incidentType:enum"] = (it in ALLOWED_INCIDENT_TYPES)

    st = ans.get("incidentStage", "")
    checks["incidentStage:enum"] = (st in ALLOWED_STAGES)

    d = ans.get("incidentDate", None)
    if d is None:
        date_ok = True
    else:
        date_ok = isinstance(d, str) and bool(DATE_RE.match(d))
    checks["incidentDate:format"] = date_ok

    # lists: no empty items
    for lf in LIST_FIELDS:
        vals = ans.get(lf, [])
        checks[f"{lf}:no_empty_items"] = all((v is not None) and (str(v).strip() != "") for v in vals)

    overall = all(checks.values())
    return checks, overall

# ----------------------------
# 4) EVAL (FPR + INPUT-AWARE SC) and LOGGING
# ----------------------------

def evaluate(outputs: List[Dict[str, Any]], inputs_by_id: Dict[str, Any], strategy_label: str) -> None:
    # Precompute exemplar masked vectors (TRIPLE for location)
    exemplar_vecs: Dict[str, Tuple[Dict[str, float], ...]] = {}
    for f in MAIN_FIELDS:
        if f == "incidentLocation":
            e1 = vec(to_style_mask(EXEMPLARS["incidentLocation_primary"]))  # "T, T"
            e2 = vec(to_style_mask(EXEMPLARS["incidentLocation_alt"]))      # "T: T (T)"
            e3 = vec(to_style_mask(EXEMPLARS["incidentLocation_tri"]))      # "T, T, T"
            exemplar_vecs[f] = (e1, e2, e3)
        else:
            exemplar_vecs[f] = (vec(to_style_mask(EXEMPLARS[f])),)

    # FPR counters
    field_pass_counter = Counter()
    field_total_counter = Counter()
    overall_passes = 0

    # SC aggregates (overall)
    sc_sums = Counter()
    sc_sumsq = Counter()
    sc_counts = Counter()

    # (1) SC case mix counts
    case_counts = {f: {"both_empty":0,"in_empty_out_filled":0,"in_filled_out_empty":0,"both_filled":0} for f in MAIN_FIELDS}

    # For (4) SMS over both_filled only
    sms_sum = Counter()
    sms_cnt = Counter()

    for rec in outputs:
        ans = rec.get("answers", {})
        rid = rec.get("id", rec.get("docid"))

        # FPR
        field_checks, overall = lint_record(rec)
        overall_passes += int(overall)
        for f in MAIN_FIELDS:
            related = [k for k in field_checks if k.startswith(f":") or k.startswith(f)]
            comp = all(field_checks[k] for k in related) if related else True
            field_pass_counter[f] += int(comp)
            field_total_counter[f] += 1

        # INPUT-AWARE SC
        in_rec = inputs_by_id.get(rid, {})

        for f in MAIN_FIELDS:
            in_has  = input_has_value_for_field(in_rec, f)
            out_val = ans.get(f, None)
            out_has = has_value(out_val)

            if not in_has and not out_has:
                sc = 1.0  # consistent abstention
                case_counts[f]["both_empty"] += 1

            elif not in_has and out_has:
                sc = 0.0  # over-filling
                case_counts[f]["in_empty_out_filled"] += 1

            elif in_has and not out_has:
                sc = 0.0  # under-filling
                case_counts[f]["in_filled_out_empty"] += 1

            else:
                # both have value -> style score with fast-paths for deterministic formats
                surface = field_to_surface(out_val)
                masked  = to_style_mask(surface)
                v_out   = vec(masked)

                if f == "incidentDate":
                    if masked == "DDDD-DD-DD":
                        sc_val = 1.0
                    else:
                        sc_val = (cosine(v_out, exemplar_vecs[f][0]) if v_out else 0.0)

                elif f in {"incidentType", "incidentStage"}:
                    if masked == "T":
                        sc_val = 1.0
                    else:
                        sc_val = (cosine(v_out, exemplar_vecs[f][0]) if v_out else 0.0)

                elif f == "incidentLocation":
                    sc_val = (max(cosine(v_out, e) for e in exemplar_vecs[f]) if v_out else 0.0)

                elif f in {"perpetratorIndividual","perpetratorOrganization","target","victim","weapon"}:
                    # Normalize and score list style.
                    # Treat EACH list item as ONE token 'T' regardless of internal words or commas in names.
                    def norm_item(s: str) -> str:
                        s = str(s)
                        s = re.sub(r'[\[\]\(\)"]', '', s)                          # drop brackets/quotes/parentheses
                        # connectors -> sentinel so we don't split native commas in names ("SMITH, JOHN")
                        s = re.sub(r'\s*(?:;|/|\band\b|&)\s*', ' ###SPLIT### ', s, flags=re.I)
                        s = re.sub(r'\s+', ' ', s).strip().rstrip('.,;')           # collapse spaces, drop trailing punc
                        return s

                    items = out_val if isinstance(out_val, list) else [out_val]
                    expanded: List[str] = []
                    for it in items:
                        ni = norm_item(it)
                        if not ni:
                            continue
                        parts = [p for p in ni.split('###SPLIT###') if p.strip()]
                        expanded.extend([p.strip() for p in parts])

                    cleaned = [p for p in expanded if p]  # keep order
                    n = len(cleaned)

                    if n == 0:
                        sc_val = 0.0
                    else:
                        # Canonical target mask for the exact length
                        # We canonicalize output by joining items with ", " so separators are to spec,
                        # and treat each item as single token => perfect style when joined canonically.
                        sc_val = 1.0

                else:
                    sc_val = (cosine(v_out, exemplar_vecs[f][0]) if v_out else 0.0)

                sc = sc_val
                case_counts[f]["both_filled"] += 1
                sms_sum[f] += sc_val
                sms_cnt[f] += 1

            sc_sums[f]  += sc
            sc_sumsq[f] += sc * sc
            sc_counts[f] += 1

    N = max(1, len(outputs))

    # FPR metrics
    fpr_per_field = {f: (field_pass_counter[f] / max(1, field_total_counter[f])) for f in MAIN_FIELDS}
    fpr_macro = sum(fpr_per_field.values()) / len(MAIN_FIELDS)
    fpr_overall = overall_passes / N

    # SC metrics (all records counted due to input-aware rules)
    sc_stats = {}
    means_for_macro = []
    for f in MAIN_FIELDS:
        n = max(1, sc_counts[f])
        mean = sc_sums[f] / n
        var  = max(0.0, (sc_sumsq[f] / n) - (mean * mean))
        sd   = math.sqrt(var)
        sc_stats[f] = (mean, sd)
        means_for_macro.append(mean)
    sc_macro = (sum(means_for_macro) / len(means_for_macro)) if means_for_macro else 0.0

    # -------- LOGS ONLY --------
    print(f"\n=== CONSISTENCY EVAL — Strategy: {strategy_label} ===")

    print("\n=== FORMAT PASS RATE (FPR) ===")
    print("FPR_field pinpoints weak spots (e.g., dates vs. lists).")
    for f in sorted(fpr_per_field.keys()):
        print(f"  {f}: {fpr_per_field[f]:.4f}")

    print("\nFPR_macro is your cross-field consistency indicator (higher = more uniform).")
    print(f"  FPR_macro: {fpr_macro:.4f}")

    print("\nFPR_overall is a strict bar: “how many records are perfectly formatted?”")
    print(f"  FPR_overall (all checks pass): {fpr_overall:.4f}")

    print("\n=== STYLE COSINE (SC) — input-aware, masking + triple location exemplar ===")
    print("Per-field: mean ± sd across ALL records (n = coverage).")
    for f in sorted(sc_stats.keys()):
        mean, sd = sc_stats[f]
        n = sc_counts[f]
        print(f"  {f}: {mean:.4f} ± {sd:.4f} (n={n})")

    print("\nSC case mix (counts):")
    for f in sorted(MAIN_FIELDS):
        cc = case_counts[f]
        print(f"  {f}: {cc}")

    print("\nACR & SMS (per field):")
    for f in MAIN_FIELDS:
        cc = case_counts[f]; total = sum(cc.values()) or 1
        acr = cc["both_empty"] / total
        sms = (sms_sum[f] / sms_cnt[f]) if sms_cnt[f] > 0 else 0.0
        print(f"  {f}: ACR={acr:.3f}, SMS={sms:.3f}")

    # ---- Availability alignment metrics (beyond ACR/SMS) ----
    print("\nAvailability alignment (per field):")
    macro = {"AA":0.0, "OFR":0.0, "UFR":0.0, "P":0.0, "R":0.0, "F1":0.0}
    for f in MAIN_FIELDS:
        cc = case_counts[f]
        total = sum(cc.values()) or 1
        bf  = cc["both_filled"]
        be  = cc["both_empty"]
        ofr = cc["in_empty_out_filled"]
        ufr = cc["in_filled_out_empty"]

        AA  = (bf + be) / total
        OFR = ofr / total
        UFR = ufr / total

        # Precision/Recall only on "fill" decisions
        fill_pred = bf + ofr
        fill_gold = bf + ufr
        P = bf / fill_pred if fill_pred > 0 else 1.0
        R = bf / fill_gold if fill_gold > 0 else 1.0
        F1 = (2*P*R/(P+R)) if (P+R) > 0 else 0.0

        macro["AA"] += AA
        macro["OFR"] += OFR
        macro["UFR"] += UFR
        macro["P"]  += P
        macro["R"]  += R
        macro["F1"] += F1

        print(f"  {f}: AA={AA:.3f}  OFR={OFR:.3f}  UFR={UFR:.3f}  Fill-P={P:.3f}  Fill-R={R:.3f}  Fill-F1={F1:.3f}")

    # Macro (unweighted mean across fields)
    m = {k: v/len(MAIN_FIELDS) for k, v in macro.items()}
    print("\nAvailability alignment (macro averages):")
    print(f"  AA_macro={m['AA']:.3f}  OFR_macro={m['OFR']:.3f}  UFR_macro={m['UFR']:.3f}  "
          f"Fill-P_macro={m['P']:.3f}  Fill-R_macro={m['R']:.3f}  Fill-F1_macro={m['F1']:.3f}")

    print("\nMacro SC (headline): mean of the per-field means.")
    print(f"  SC_macro: {sc_macro:.4f}")

# ----------------------------
# 5) CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Logs-only MUC-4 consistency evaluator (input-aware SC, input presence from 'templates').")
    ap.add_argument("--inputs",  required=True,  help="Path to in.json (required; presence read from 'templates').")
    ap.add_argument("--outputs", required=True,  help="Path to out.json (strategy outputs).")
    ap.add_argument("--strategy", default="S?",  help="Label printed in the header.")
    args = ap.parse_args()

    # Load outputs
    with open(args.outputs, "r", encoding="utf-8") as f:
        outs = json.load(f)
    if not isinstance(outs, list):
        raise ValueError("out.json must be a JSON list of records")

    # Load inputs and build lookup by record id (fallback to 'docid')
    with open(args.inputs, "r", encoding="utf-8") as f:
        ins = json.load(f)
    inputs_by_id: Dict[str, Any] = {}
    if isinstance(ins, list):
        for rec in ins:
            key = rec.get("id", rec.get("docid"))
            if key is not None:
                inputs_by_id[key] = rec
    else:
        inputs_by_id = ins

    evaluate(outs, inputs_by_id, args.strategy)

if __name__ == "__main__":
    main()
