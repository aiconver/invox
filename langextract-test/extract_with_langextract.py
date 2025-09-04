#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import textwrap
from collections import defaultdict
import langextract as lx  # pip install langextract

# ---------------------------- Logging ----------------------------
def setup_logging(debug: bool):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)]
    )

def trunc(obj: Any, n: int = 800) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False)
    except Exception:
        s = repr(obj)
    return s if len(s) <= n else s[:n] + "...[truncated]"

# ---------------------------- Helpers ----------------------------
PROMPT_TEMPLATE = textwrap.dedent("""\
    Extract entities from the given news-like text.

    • Use exact text spans from the document for extraction_text (no paraphrasing).
    • Order extractions by first appearance; avoid overlapping spans.
    • Classes to extract: {classes}

    Return only entities that can be clearly identified in the text.
""")


import time

# At the top-level, keep track of last call timestamps
_last_call_times = []

def throttled_extract(**kwargs):
    global _last_call_times
    now = time.time()
    # prune old timestamps (older than 60s)
    _last_call_times = [t for t in _last_call_times if now - t < 60]

    # if we already did 1 calls in the past 60s, wait
    if len(_last_call_times) >= 1:
        sleep_for = 60 - (now - _last_call_times[0])
        if sleep_for > 0:
            print(f"⏳ Throttling: sleeping {sleep_for:.1f} seconds to respect 1 calls/minute")
            time.sleep(sleep_for)
            # after sleep, prune again
            now = time.time()
            _last_call_times = [t for t in _last_call_times if now - t < 60]

    # make the real API call
    result = lx.extract(**kwargs)
    _last_call_times.append(time.time())
    return result


def derive_classes(templates: List[Dict[str, Any]]) -> List[str]:
    classes = set()
    for t in templates:
        inc = t.get("incident_type", "unknown")
        for entity_type in t.keys():
            if entity_type not in ("incident_type", "incident_id"):
                classes.add(f"{inc}_{entity_type}")
    return sorted(classes)

def build_example_for_doc_text(text: str, templates: List[Dict[str, Any]]) -> Optional[lx.data.ExampleData]:
    """
    Turn your gold labels into a single ExampleData with many Extractions.
    We DO NOT add spans; we follow your style snippet and only set class + extraction_text.
    If a doc has no labeled strings, return None.
    """
    extractions: List[lx.data.Extraction] = []

    for t in templates:
        inc = t.get("incident_type", "unknown")
        for entity_type, groups in t.items():
            if entity_type in ("incident_type", "incident_id") or not groups:
                continue
            # groups is a list of lists: [ [ [text, start], ... ], [ ... ] ]
            for group in groups:
                for extraction_text, _start in group:
                    if not isinstance(extraction_text, str):
                        continue
                    extractions.append(
                        lx.data.Extraction(
                            extraction_class=f"{inc}_{entity_type}",
                            extraction_text=extraction_text
                            # attributes={}  # optional; omit like your sample
                        )
                    )

    if not extractions:
        return None

    return lx.data.ExampleData(text=text, extractions=extractions)

def build_global_examples(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build a pool: [{docid, example: ExampleData}]
    """
    pool = []
    for d in docs:
        docid = d.get("docid")
        text = d.get("doctext")
        templates = d.get("templates", [])
        if not text or not templates:
            continue
        ex = build_example_for_doc_text(text, templates)
        if ex:
            pool.append({"docid": docid, "example": ex})
    return pool

def pick_examples(pool: List[Dict[str, Any]], current_docid: str, max_examples: int) -> List[lx.data.ExampleData]:
    selected = []
    for item in pool:
        if item["docid"] == current_docid:
            continue
        selected.append(item["example"])
        if len(selected) >= max_examples:
            break
    return selected

def strip_incident_prefix(d: Dict[str, List[str]]) -> Dict[str, List[str]]:
    from collections import defaultdict
    out = defaultdict(list)
    for k, vals in d.items():
        _, _, tail = k.partition("_")
        out[(tail or k)].extend(vals)
    return dict(out)

def strip_prefix_evidence(ev: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    from collections import defaultdict
    out = {}
    tmp = defaultdict(lambda: {"spans": [], "offsets": []})
    for k, v in ev.items():
        _, _, tail = k.partition("_")
        t = tail or k
        tmp[t]["spans"].extend(v.get("spans", []))
        tmp[t]["offsets"].extend(v.get("offsets", []))
    return dict(tmp)

def parse_result(result) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, Any]]]:
    """
    Parse answers/evidence whether result is dict or object.
    We only rely on .extractions or ['extractions'] and extraction fields.
    """
    answers: Dict[str, List[str]] = defaultdict(list)
    evidence: Dict[str, Dict[str, Any]] = {}

    # get extractions list
    if isinstance(result, dict):
        extractions = result.get("extractions") or (result.get("data", {}) or {}).get("extractions")
    else:
        extractions = getattr(result, "extractions", None)

    if not extractions:
        return answers, evidence

    for ext in extractions:
        if isinstance(ext, dict):
            cls = ext.get("extraction_class")
            txt = ext.get("extraction_text")
        else:
            cls = getattr(ext, "extraction_class", None)
            txt = getattr(ext, "extraction_text", None)

        if not cls or not txt:
            continue

        answers[cls].append(txt)
        evidence.setdefault(cls, {"spans": [], "offsets": []})
        evidence[cls]["spans"].append(txt)
        # offsets unavailable because we didn’t provide spans; leave empty

    return answers, evidence

# ---------------------------- Main pipeline ----------------------------
def main():
    ap = argparse.ArgumentParser(description="LangExtract: dataset extractor (ExampleData/Extraction style).")
    ap.add_argument("--dataset", required=True, help="Path to input JSON (pretty_test.json).")
    ap.add_argument("--out", required=True, help="Path to output JSON.")
    ap.add_argument("--pretty", action="store_true", help="Pretty print output.")
    ap.add_argument("--model-id", default="gemini-2.5-flash")
    ap.add_argument("--api-key", default=None, help="If omitted, LANGEXTRACT_API_KEY env var is used.")
    ap.add_argument("--max-examples", type=int, default=3, help="Few-shot examples per doc.")
    ap.add_argument("--extraction-passes", type=int, default=2, help="Like your sample.")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--max-char-buffer", type=int, default=1500)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    setup_logging(args.debug)

    api_key = args.api_key or os.getenv("LANGEXTRACT_API_KEY")
    in_path = Path(args.dataset)
    out_path = Path(args.out)

    if not in_path.exists():
        logging.error(f"Dataset not found at '{in_path}'.")
        sys.exit(1)

    try:
        with open(in_path, "r", encoding="utf-8") as f:
            documents: List[Dict[str, Any]] = json.load(f)
        if not isinstance(documents, list):
            logging.error("Input must be a list of documents.")
            sys.exit(1)
    except Exception as e:
        logging.exception("Failed to read dataset:")
        sys.exit(1)

    # Build cross-doc few-shot pool first (as ExampleData objects)
    pool = build_global_examples(documents)
    logging.debug(f"Global pool size: {len(pool)}")

    outputs = []
    had_error = False
    processed = 0

    for doc in documents:
        docid = doc.get("docid")
        text = doc.get("doctext")
        templates = doc.get("templates", [])
        if not text or not templates:
            logging.warning(f"Skipping '{docid}' (missing text/templates).")
            continue

        classes = derive_classes(templates)
        prompt = PROMPT_TEMPLATE.format(classes=", ".join(classes))

        # Few-shot: prefer examples from *other* docs; if none, fall back to this doc’s own example
        examples = pick_examples(pool, current_docid=docid, max_examples=args.max_examples)
        if not examples:
            fallback = build_example_for_doc_text(text, templates)
            if fallback:
                examples = [fallback]
            else:
                # Last ditch: create a single neutral example from the first class using a small snippet
                snippet = (text[:200] + "...") if len(text) > 200 else text
                examples = [
                    lx.data.ExampleData(
                        text=snippet,
                        extractions=[
                            lx.data.Extraction(
                                extraction_class=classes[0] if classes else "entity_Label",
                                extraction_text=snippet.split(" ")[0] if snippet else "N/A"
                            )
                        ]
                    )
                ]

        extract_kwargs = dict(
            text_or_documents=text,
            prompt_description=prompt,
            examples=examples,
            model_id=args.model_id,
            extraction_passes=args.extraction_passes,
            max_workers=args.max_workers,
            max_char_buffer=args.max_char_buffer
        )
        if api_key:
            extract_kwargs["api_key"] = api_key

        logging.info(f"Doc {docid}: classes={classes} | examples_used={len(examples)}")

        t0 = time.time()
        try:
            result = throttled_extract(**extract_kwargs)
            logging.debug("Result preview: " + trunc(getattr(result, "__dict__", result)))
            ans, ev = parse_result(result)
            ans_plain = strip_incident_prefix(ans)
            ev_plain  = strip_prefix_evidence(ev)

            err = None
        except Exception as e:
            err = str(e)
            ans, ev = {}, {}
            logging.exception(f"Extraction failed for {docid}:")

        dt = round((time.time() - t0) * 1000.0, 1)
        outputs.append({
            "id": docid,
            "meta": {
                "model_id": "langextract",
                "timing": {"duration_ms": dt},
                "error": err
            },
            "answers": ans_plain,
            # "evidence": ev_plain
        })
        processed += 1
        if err:
            had_error = True

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=2 if args.pretty else None, ensure_ascii=False)
    except Exception as e:
        logging.exception("Failed to write output:")
        sys.exit(1)

    logging.info(f"✅ Processed {processed} docs. Wrote {out_path}.")
    sys.exit(1 if had_error else 0)

if __name__ == "__main__":
    main()
