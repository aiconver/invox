#!/usr/bin/env python3
import argparse, json, os, time, logging, sys, random
from pathlib import Path
from collections import defaultdict
import langextract as lx  # pip install langextract

FIELD_WHITELIST = {"PerpInd", "PerpOrg", "Target", "Victim", "Weapon"}
PROMPT = (
    "Extract entities from the given news-like text.\n\n"
    "• Use exact text spans from the document for extraction_text (no paraphrasing).\n"
    "• Order extractions by first appearance; avoid overlapping spans.\n"
    "• Classes to extract: {classes}\n\n"
    "Return only entities that can be clearly identified in the text.\n"
)

def main():
    ap = argparse.ArgumentParser(description="Minimal LangExtract (six-field, static few-shots, retry on 503).")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--examples-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--examples-per-type", type=int, default=1)
    ap.add_argument("--example-max-per-field", type=int, default=1)
    ap.add_argument("--model-id", default="gemini-2.5-flash")
    ap.add_argument("--api-key", default=os.getenv("LANGEXTRACT_API_KEY"))
    ap.add_argument("--extraction-passes", type=int, default=1)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--max-char-buffer", type=int, default=4000)
    ap.add_argument("--rpm", type=int, default=6, help="target req/min (client-side spacing)")
    ap.add_argument("--pretty", action="store_true")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)]
    )
    for name in ["langextract", "httpx", "httpcore", "google", "tenacity"]:
        logging.getLogger(name).setLevel(logging.WARNING)

    docs = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    examples_raw = json.loads(Path(args.examples_file).read_text(encoding="utf-8"))
    if not isinstance(docs, list) or not isinstance(examples_raw, list):
        logging.error("Both --dataset and --examples-file must be JSON arrays.")
        sys.exit(1)

    # Build static few-shots grouped by incident_type; keep only values that appear exactly in example text.
    ex_by_type = defaultdict(list)
    for ex in examples_raw:
        itype = (ex.get("incident_type") or "unknown").strip() or "unknown"
        text = ex.get("text", "")
        extractions = []
        for field in FIELD_WHITELIST:
            vals = ex.get("answers", {}).get(field, [])
            if isinstance(vals, (str, int, float)): vals = [str(vals)]
            seen, kept = set(), 0
            for v in vals or []:
                v = str(v).strip()
                if not v or v in seen: continue
                if v in text:  # exact substring to avoid fuzzy alignment slowness
                    extractions.append(lx.data.Extraction(extraction_class=f"{itype}_{field}", extraction_text=v))
                    seen.add(v); kept += 1
                    if kept >= args.example_max_per_field: break
        if extractions:
            ex_by_type[itype].append(lx.data.ExampleData(text=text, extractions=extractions))

    outputs, last_call = [], 0.0
    interval = max(60.0 / max(1, args.rpm), 0.01)

    for i, d in enumerate(docs, 1):
        docid, text, templates = d.get("docid"), d.get("doctext"), d.get("templates") or []
        if not text or not templates: continue

        classes = sorted({
            f"{(t.get('incident_type') or 'unknown').strip() or 'unknown'}_{k}"
            for t in templates for k in t.keys() if k in FIELD_WHITELIST and t.get(k)
        })
        if not classes: continue

        itypes = []
        for t in templates:
            it = (t.get("incident_type") or "unknown").strip() or "unknown"
            if it not in itypes: itypes.append(it)

        examples = []
        for it in itypes:
            examples.extend(ex_by_type.get(it, [])[:args.examples_per_type])

        if args.progress:
            logging.info(f"[{i}/{len(docs)}] {docid} itypes={itypes} classes={len(classes)} examples={len(examples)}")

        # Simple RPM limiter
        now = time.monotonic()
        if last_call and (now - last_call) < interval:
            time.sleep(interval - (now - last_call))

        # === Retry indefinitely while server returns 503 / overloaded / unavailable ===
        attempt = 0
        while True:
            try:
                result = lx.extract(
                    text_or_documents=text,
                    prompt_description=PROMPT.format(classes=", ".join(classes)),
                    examples=examples,
                    model_id=args.model_id,
                    extraction_passes=args.extraction_passes,
                    max_workers=args.max_workers,
                    max_char_buffer=args.max_char_buffer,
                    **({"api_key": args.api_key} if args.api_key else {})
                )
                break  # success
            except Exception as e:
                msg = str(e).lower()
                is_503 = (" 503" in msg) or ("unavailable" in msg) or ("overloaded" in msg)
                if not is_503:
                    raise  # non-503 errors bubble up
                # backoff: 1,2,4,8,16,32,60,60,... seconds + jitter
                backoff = [1,2,4,8,16,32][attempt] if attempt < 6 else 60
                attempt += 1
                logging.warning(f"503 from provider; retrying in ~{backoff}s (attempt {attempt})")
                time.sleep(backoff + random.uniform(0, 0.5))

        last_call = time.monotonic()

        # Parse results → strip '<incident>_' prefix; keep only whitelisted fields
        extrs = getattr(result, "extractions", None) or (result.get("extractions") if isinstance(result, dict) else None) or (result.get("data", {}).get("extractions") if isinstance(result, dict) else None) or []
        by_cls = defaultdict(list)
        for ext in extrs:
            cls = getattr(ext, "extraction_class", None) or (ext.get("extraction_class") if isinstance(ext, dict) else None)
            txt = getattr(ext, "extraction_text", None) or (ext.get("extraction_text") if isinstance(ext, dict) else None)
            if cls and txt:
                by_cls[cls].append(txt)
        answers = defaultdict(list)
        for k, vals in by_cls.items():
            tail = k.split("_", 1)[1] if "_" in k else k
            if tail in FIELD_WHITELIST:
                answers[tail].extend(vals)

        outputs.append({"id": docid, "meta": {"model_id": args.model_id}, "answers": dict(answers)})

    Path(args.out).write_text(json.dumps(outputs, indent=2 if args.pretty else None, ensure_ascii=False), encoding="utf-8")
    logging.info(f"✅ Wrote {args.out} with {len(outputs)} items.")

if __name__ == "__main__":
    main()
