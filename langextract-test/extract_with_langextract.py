#!/usr/bin/env python3
import argparse, json, os, time, logging, sys, random, re
from pathlib import Path
from collections import defaultdict
from datetime import date
import langextract as lx  # pip install langextract

# ---------- enums for label-classes ----------
TYPE_ENUMS  = ["ATTACK","BOMBING","KIDNAPPING","ASSASSINATION","ARSON","HIJACKING","OTHER"]
STAGE_ENUMS = ["ACCOMPLISHED","ATTEMPTED","PLANNED","FAILED","THREATENED"]

# ---------- span fields in your camelCase schema ----------
SPAN_FIELDS = {
  "perpetratorIndividual", "perpetratorOrganization",
  "target", "victim", "weapon",
  "incidentDate", "incidentLocation",
}
LIST_FIELDS = {"perpetratorIndividual","perpetratorOrganization","target","victim","weapon"}
SINGLETON_FIELDS = {"incidentDate","incidentLocation"}

PROMPT = (
    "Extract entities from the given news-like text.\n\n"
    "• Use exact text spans from the document for extraction_text (no paraphrasing).\n"
    "• Order extractions by first appearance; avoid overlapping spans.\n"
    "• For dates, extract the exact date phrase as it appears (e.g., '20 DEC 89'); normalization is downstream.\n"
    "• For locations, extract place-name spans only (e.g., 'LA PAZ', 'BOGOTA', 'PROVIDENCIA').\n"
    "• LABEL classes: If supported by the text, EMIT labels with a short evidence span:\n"
    "  - incidentType=ATTACK|BOMBING|KIDNAPPING|ASSASSINATION|ARSON|HIJACKING|OTHER\n"
    "  - incidentStage=ACCOMPLISHED|ATTEMPTED|PLANNED|FAILED|THREATENED\n"
    "  You MUST output exactly one incidentType=… label if a violent incident is described; otherwise none.\n"
    "  You MUST output at most one incidentStage=… label when the stage is stated or clearly implied.\n"
    "• Classes to extract: {classes}\n\n"
    "Return only entities that can be clearly identified in the text.\n"
)

# ---------- helpers for normalization ----------
_MON = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def normalize_date_yyyy_mm_dd(span: str):
    s = (span or "").strip().upper().replace(",", " ")
    s = re.sub(r"\s+", " ", s)
    if DATE_RE.match(s):
        return s
    m = re.match(r"^(\d{1,2})\s+([A-Z]{3,})\s+(\d{2,4})$", s)
    if m:
        d = int(m.group(1))
        mon = _MON.get(m.group(2)[:3])
        y = int(m.group(3))
        if mon:
            if y < 100:
                y += 1900 if y >= 50 else 2000
            try:
                return date(y, mon, d).strftime("%Y-%m-%d")
            except ValueError:
                return None
    return None

_BAD_FACILITY_WORDS = {"EMBASSY","CONSULATE","BUILDING","BANK","UNIVERSITY","MINISTRY","HEADQUARTERS"}
def choose_location(spans):
    spans = [x.strip(" .") for x in spans or [] if x and x.strip()]
    if not spans: return ""
    prefer = [x for x in spans if not any(w in x.upper() for w in _BAD_FACILITY_WORDS)]
    cand = prefer or spans
    cand.sort(key=len, reverse=True)
    return cand[0]

def _pick_label(votes: dict[str, list[str]]) -> str:
    # choose label with most evidence; tie-break by longest span; else ""
    best_label, best_count, best_len = "", 0, -1
    for label, spans in votes.items():
        cnt = len(spans)
        mxlen = max((len(s) for s in spans), default=0)
        if cnt > best_count or (cnt == best_count and mxlen > best_len):
            best_label, best_count, best_len = label, cnt, mxlen
    return best_label if best_count > 0 else ""

def _first_sentence(text: str, max_chars: int = 160) -> str:
    sent = re.split(r'(?<=[.!?])\s+', (text or "").strip(), maxsplit=1)[0]
    return (sent[:max_chars]).strip()

# ---------- robust fallbacks ----------
EMPTY_ANSWERS = {
    "incidentType": "",
    "incidentDate": None,          # -> null in JSON
    "incidentLocation": "",
    "incidentStage": "",
    "perpetratorIndividual": [],
    "perpetratorOrganization": [],
    "target": [],
    "victim": [],
    "weapon": [],
}

def make_empty_output(docid: str, model_id: str, latency_ms: int = 0, error: str | None = None):
    meta = {"model_id": model_id, "timing": {"duration_ms": latency_ms}}
    if error:
        meta["error"] = error
    return {"id": docid, "meta": meta, "answers": dict(EMPTY_ANSWERS)}

MAX_ATTEMPTS = int(os.getenv("LX_MAX_ATTEMPTS", "8"))

def main():
    ap = argparse.ArgumentParser(description="LangExtract (camelCase, span+label classes, static few-shots, retry on 503).")
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

    try:
        docs = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
        examples_raw = json.loads(Path(args.examples_file).read_text(encoding="utf-8"))
    except Exception as e:
        logging.error(f"Failed to read inputs: {e}")
        sys.exit(1)

    if not isinstance(docs, list) or not isinstance(examples_raw, list):
        logging.error("Both --dataset and --examples-file must be JSON arrays.")
        sys.exit(1)

    # ---- Build static few-shots grouped by incidentType; only keep substrings present in example text. ----
    ex_by_type = defaultdict(list)
    for ex in examples_raw:
        itype = (ex.get("incidentType") or "unknown").strip().upper() or "UNKNOWN"
        text = ex.get("text") or ex.get("transcript") or ""
        extractions = []

        # SPAN fields
        for field in SPAN_FIELDS:
            vals = ex.get("answers", {}).get(field, [])
            if isinstance(vals, (str, int, float)): vals = [str(vals)]
            seen, kept = set(), 0
            for v in vals or []:
                v = str(v).strip()
                if not v or v in seen: continue
                if v in text:
                    extractions.append(lx.data.Extraction(extraction_class=f"{itype}_{field}", extraction_text=v))
                    seen.add(v); kept += 1
                    if kept >= args.example_max_per_field: break

        # Seed label-class examples with a short evidence span
        it = (ex.get("incidentType") or "").strip().upper()
        if it in TYPE_ENUMS and text:
            extractions.append(lx.data.Extraction(
                extraction_class=f"incidentType={it}",
                extraction_text=_first_sentence(text)
            ))
        st = (ex.get("incidentStage") or "").strip().upper()
        if st in STAGE_ENUMS and text:
            extractions.append(lx.data.Extraction(
                extraction_class=f"incidentStage={st}",
                extraction_text=_first_sentence(text)
            ))

        if extractions:
            ex_by_type[itype].append(lx.data.ExampleData(text=text, extractions=extractions))
    
    # === GLOBAL FALLBACK POOL (for unknown/missing incident types) ===
    GLOBAL_KEY = "__GLOBAL__"
    global_pool = []
    for arr in ex_by_type.values():
        global_pool.extend(arr)
    ex_by_type[GLOBAL_KEY] = global_pool

    outputs, last_call = [], 0.0
    interval = max(60.0 / max(1, args.rpm), 0.01)

    for i, d in enumerate(docs, 1):
        docid = d.get("docid") or d.get("id") or f"doc_{i}"
        text = d.get("doctext") or d.get("transcript") or ""
        templates = d.get("templates") or []

        started = time.perf_counter()

        try:
            if not text or not templates:
                if args.progress:
                    logging.info(f"[{i}/{len(docs)}] {docid} missing text/templates → empty answers")
                outputs.append(make_empty_output(docid, args.model_id, int((time.perf_counter() - started)*1000), "missing text/templates"))
                continue

            # Incident types present in this doc (default to UNKNOWN)
            itypes = list({ (t.get("incidentType") or "unknown").strip().upper() or "UNKNOWN" for t in templates }) or ["UNKNOWN"]

            # Build span-classes prefixed by incident type
            span_classes = { f"{it}_{k}" for it in itypes for k in SPAN_FIELDS }

            # Add unscoped singletons to improve recall
            unscoped_singletons = set(SINGLETON_FIELDS)  # {"incidentDate","incidentLocation"}

            # Add label-classes (unprefixed)
            label_classes = { f"incidentType={x}" for x in TYPE_ENUMS } | { f"incidentStage={x}" for x in STAGE_ENUMS }

            classes = sorted(span_classes | unscoped_singletons | label_classes)

            if not classes:
                outputs.append(make_empty_output(docid, args.model_id, int((time.perf_counter() - started)*1000), "no classes"))
                continue

            # Collect few-shot examples per incident type
            examples = []
            for it in itypes:
                examples.extend(ex_by_type.get(it, [])[:args.examples_per_type])

            # --- Fallback 1: use global pool if type-specific examples are empty ---
            if not examples:
                examples = ex_by_type.get(GLOBAL_KEY, [])[:max(1, args.examples_per_type)]
                if args.progress:
                    logging.info(f"[{i}/{len(docs)}] {docid} using GLOBAL fallback examples")

            # --- Fallback 2: synthesize a minimal example to satisfy LangExtract ---
            if not examples:
                first = _first_sentence(text) or (text[:160] if text else "AN INCIDENT OCCURRED.")
                examples = [lx.data.ExampleData(
                    text=first,
                    extractions=[
                        lx.data.Extraction(extraction_class="incidentType=OTHER", extraction_text=first)
                    ],
                )]
                if args.progress:
                    logging.info(f"[{i}/{len(docs)}] {docid} using SYNTHESIZED fallback example")

            if args.progress:
                logging.info(f"[{i}/{len(docs)}] {docid} itypes={itypes} classes={len(classes)} examples={len(examples)}")

            # Simple RPM limiter
            now = time.monotonic()
            if last_call and (now - last_call) < interval:
                time.sleep(interval - (now - last_call))

            # === Retry loop for provider ===
            attempt = 0
            latency_ms = 0
            while True:
                try:
                    t0 = time.perf_counter()
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
                    latency_ms = int((time.perf_counter() - t0) * 1000)
                    break
                except Exception as e:
                    msg = str(e).lower()
                    is_503  = (" 503" in msg) or ("unavailable" in msg) or ("overloaded" in msg)
                    is_parse = ("failed to parse content" in msg) or ("resolverparsingerror" in msg) or ("must be a non-empty string" in msg)

                    if (not is_503 and not is_parse) or attempt >= MAX_ATTEMPTS:
                        raise  # bubble to outer try/except

                    backoff = [1,2,4,8,16,32][attempt] if attempt < 6 else 60
                    attempt += 1
                    kind = "503/overload" if is_503 else "parse"
                    logging.warning(f"{kind}: retrying in ~{backoff}s (attempt {attempt})")
                    time.sleep(backoff + random.uniform(0, 0.5))

            last_call = time.monotonic()

            # ---- Parse results ----
            extrs = getattr(result, "extractions", None) \
                or (result.get("extractions") if isinstance(result, dict) else None) \
                or (result.get("data", {}).get("extractions") if isinstance(result, dict) else None) \
                or []
            by_cls = defaultdict(list)
            for ext in extrs:
                cls = getattr(ext, "extraction_class", None) or (ext.get("extraction_class") if isinstance(ext, dict) else None)
                txt = getattr(ext, "extraction_text", None) or (ext.get("extraction_text") if isinstance(ext, dict) else None)
                if cls and txt:
                    by_cls[cls].append(txt)

            if args.debug:
                logging.debug(f"[{docid}] classes returned: {sorted(by_cls.keys())}")
                logging.debug(f"[{docid}] latency_ms={latency_ms}")

            # Gather span answers from "<TYPE>_<field>" and unscoped singletons
            answers = defaultdict(list)
            for k, vals in by_cls.items():
                tail = k.split("_", 1)[1] if "_" in k else k
                if tail in SPAN_FIELDS:
                    answers[tail].extend(vals)

            # Votes for label-classes
            type_votes  = {lab: by_cls.get(f"incidentType={lab}", []) for lab in TYPE_ENUMS}
            stage_votes = {lab: by_cls.get(f"incidentStage={lab}", []) for lab in STAGE_ENUMS}
            incident_type_chosen  = _pick_label(type_votes)
            incident_stage_chosen = _pick_label(stage_votes)

            # ---- Post-process to your schema ----
            final_answers = {}

            # list fields
            for f in LIST_FIELDS:
                final_answers[f] = answers.get(f, [])

            # singletons
            date_spans = answers.get("incidentDate", [])
            final_answers["incidentDate"] = normalize_date_yyyy_mm_dd(date_spans[0]) if date_spans else None

            loc_spans = answers.get("incidentLocation", [])
            final_answers["incidentLocation"] = choose_location(loc_spans) if loc_spans else ""

            # enums chosen via label-class votes
            final_answers["incidentType"]  = incident_type_chosen
            final_answers["incidentStage"] = incident_stage_chosen

            outputs.append({
                "id": docid,
                "meta": {"model_id": args.model_id, "timing": {"duration_ms": latency_ms}},
                "answers": dict(final_answers)
            })

        except Exception as e:
            latency_ms = int((time.perf_counter() - started) * 1000)
            logging.error(f"[{docid}] hard failure, emitting empty answers and continuing: {e}",
                          exc_info=args.debug)
            outputs.append(make_empty_output(docid, args.model_id, latency_ms, str(e)))
            continue

    Path(args.out).write_text(
        json.dumps(outputs, indent=2 if args.pretty else None, ensure_ascii=False),
        encoding="utf-8"
    )
    logging.info(f"✅ Wrote {args.out} with {len(outputs)} items.")

if __name__ == "__main__":
    main()
