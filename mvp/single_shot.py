#!/usr/bin/env python3
import argparse, json, os, sys, time, logging, subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI, APIConnectionError, RateLimitError, APIError

ENTITY_FIELDS = ["PerpInd", "PerpOrg", "Target", "Victim", "Weapon"]

# ---------- Schema via Pydantic ----------
class EntityExtraction(BaseModel):
    PerpInd: List[str] = Field(default_factory=list)
    PerpOrg: List[str] = Field(default_factory=list)
    Target:  List[str] = Field(default_factory=list)
    Victim:  List[str] = Field(default_factory=list)
    Weapon:  List[str] = Field(default_factory=list)

# ---------- Prompts ----------
SYSTEM_PROMPT = """You are an information extraction assistant.

Task: Extract entities from ONE news-like document.

Rules:
- Use EXACT text spans copied from the document; do not paraphrase or normalize.
- If an entity is not clearly present, return an EMPTY list.
- Order spans by first appearance in the document.
- Avoid overlapping spans and remove duplicates.
Always return your result by calling the provided tool with the five arrays.
""".strip()

USER_PROMPT_TEMPLATE = """Document ID: {docid}

Document:
\"\"\"{doctext}\"\"\"

Extract these fields as exact spans:
- PerpInd  (perpetrator individuals or descriptors like "some military")
- PerpOrg  (perpetrator organizations like "shining path")
- Target   (intended target, if any)
- Victim   (victims explicitly mentioned)
- Weapon   (weapons explicitly mentioned)
""".strip()

# ---------- Tool schema (Chat Completions) ----------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "save_extractions",
            "description": "Return extracted spans as arrays of strings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "PerpInd": {"type": "array", "items": {"type": "string"}},
                    "PerpOrg": {"type": "array", "items": {"type": "string"}},
                    "Target":  {"type": "array", "items": {"type": "string"}},
                    "Victim":  {"type": "array", "items": {"type": "string"}},
                    "Weapon":  {"type": "array", "items": {"type": "string"}}
                },
                "required": ["PerpInd", "PerpOrg", "Target", "Victim", "Weapon"],
                "additionalProperties": False
            }
        }
    }
]

# ---------- Helpers ----------
def setup_logging(debug: bool):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

def load_docs(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Input must be a list of documents.")
    return data

def dedupe_keep_order(items: List[str]) -> List[str]:
    seen, out = set(), []
    for x in items or []:
        if not isinstance(x, str): 
            continue
        v = x.strip()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out

def sanitize_extraction(e: EntityExtraction) -> EntityExtraction:
    return EntityExtraction(
        PerpInd=dedupe_keep_order(e.PerpInd),
        PerpOrg=dedupe_keep_order(e.PerpOrg),
        Target=dedupe_keep_order(e.Target),
        Victim=dedupe_keep_order(e.Victim),
        Weapon=dedupe_keep_order(e.Weapon),
    )

def call_openai_tool(
    client: OpenAI,
    model: str,
    docid: str,
    doctext: str,
    temperature: float = 0.0,
    max_retries: int = 5,
    retry_base: float = 1.6,
) -> EntityExtraction:
    user_prompt = USER_PROMPT_TEMPLATE.format(docid=docid, doctext=doctext)
    attempt = 0
    while True:
        try:
            chat = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                tools=TOOLS,
                tool_choice={"type": "function", "function": {"name": "save_extractions"}},
            )
            choice = chat.choices[0]
            tool_calls = choice.message.tool_calls or []
            if not tool_calls:
                logging.warning(f"[{docid}] Model did not call the tool; returning empty.")
                return EntityExtraction()

            args_str = tool_calls[0].function.arguments or "{}"
            try:
                data = json.loads(args_str)
                extraction = EntityExtraction(**data)
                return sanitize_extraction(extraction)
            except (json.JSONDecodeError, ValidationError) as ve:
                logging.warning(f"[{docid}] Invalid tool arguments: {ve}; returning empty.")
                return EntityExtraction()

        except (RateLimitError, APIConnectionError, APIError) as e:
            attempt += 1
            if attempt > max_retries:
                logging.error(f"[{docid}] Giving up after {attempt} attempts: {e}")
                return EntityExtraction()
            sleep_s = retry_base ** attempt
            logging.warning(f"[{docid}] OpenAI error {type(e).__name__}: {e} — retrying in {sleep_s:.1f}s...")
            time.sleep(sleep_s)
        except Exception as e:
            logging.exception(f"[{docid}] Unexpected error:")
            return EntityExtraction()

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Extract entities with ChatGPT (tool calling) and save results.")
    ap.add_argument("--dataset", required=True, help="Path to pretty_test.json")
    ap.add_argument("--out", required=True, help="Path to write predictions JSON")
    ap.add_argument("--model-id", default="gpt-4o-mini", help="OpenAI model (e.g., gpt-4o-mini, gpt-4o)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--sleep", type=float, default=0.4, help="Seconds to sleep between requests")
    ap.add_argument("--pretty", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--eval", action="store_true", help="Run BLEURT eval after extraction")
    ap.add_argument("--eval-model", default="Elron/bleurt-large-512")
    ap.add_argument("--eval-threshold", type=float, default=0.5)
    args = ap.parse_args()

    setup_logging(args.debug)

    if not os.getenv("OPENAI_API_KEY"):
        logging.error("OPENAI_API_KEY is not set.")
        sys.exit(1)

    client = OpenAI()

    in_path = Path(args.dataset)
    out_path = Path(args.out)

    try:
        docs = load_docs(in_path)
    except Exception:
        logging.exception("Failed to read dataset:")
        sys.exit(1)

    outputs: List[Dict[str, Any]] = []
    had_error = False

    logging.info(f"Starting extraction on {len(docs)} docs with model '{args.model_id}'")
    for d in docs:
        docid = d.get("docid")
        text = d.get("doctext") or ""
        if not docid or not text:
            logging.warning(f"Skipping invalid doc: {d.get('docid')}")
            continue

        logging.info(f"→ Extracting {docid} ...")
        t0 = time.time()
        err: Optional[str] = None
        answers = EntityExtraction().model_dump()

        try:
            extraction = call_openai_tool(
                client=client,
                model=args.model_id,
                docid=docid,
                doctext=text,
                temperature=args.temperature,
            )
            answers = extraction.model_dump()
        except Exception as e:
            err = str(e); had_error = True

        dt_ms = round((time.time() - t0) * 1000.0, 1)
        outputs.append({
            "id": docid,
            "meta": {"model_id": args.model_id, "timing": {"duration_ms": dt_ms}, "error": err},
            "answers": answers
        })
        logging.info(f"✓ {docid} in {dt_ms} ms")
        if args.sleep > 0:
            time.sleep(args.sleep)

    try:
        out_path.write_text(
            json.dumps(outputs, indent=2 if args.pretty else None, ensure_ascii=False),
            encoding="utf-8"
        )
        logging.info(f"✅ Wrote {len(outputs)} items to {out_path}")
    except Exception:
        logging.exception("Failed to write output:")
        sys.exit(1)

    if had_error:
        logging.warning("Some documents had errors; see logs.")

    if args.eval:
        evaluator = (Path(__file__).parent / "../langextract-test/eval_bleurt_spans.py").resolve()
        if not evaluator.exists():
            logging.error("eval_bleurt_spans.py not found next to this script. Skipping eval.")
            return
        cmd = [
            sys.executable, str(evaluator),
            "--gold", str(in_path),
            "--pred", str(out_path),
            "--model", args.eval_model,
            "--threshold", str(args.eval_threshold),
            "--pretty",
        ]
        logging.info("Running BLEURT eval...")
        try:
            subprocess.run(cmd, check=True)
        except Exception:
            logging.exception("BLEURT evaluation failed.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Fatal error:")
        sys.exit(1)
