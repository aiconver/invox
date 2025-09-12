#!/usr/bin/env python3
# single_shot.py
import argparse, json, os, sys, time, logging, subprocess, random
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI, APIConnectionError, RateLimitError, APIError, BadRequestError

ENTITY_FIELDS = ["PerpInd", "PerpOrg", "Target", "Victim", "Weapon"]

# ---------- Schema via Pydantic ----------
class EntityExtraction(BaseModel):
    PerpInd: List[str] = Field(default_factory=list)
    PerpOrg: List[str] = Field(default_factory=list)
    Target:  List[str] = Field(default_factory=list)
    Victim:  List[str] = Field(default_factory=list)
    Weapon:  List[str] = Field(default_factory=list)

# ---------- Prompts ----------
SYSTEM_PROMPT = """
You are an information extraction assistant specializing in violent incidents from news documents.

Input: one news-like document.
Output: call the provided tool with five arrays of exact text spans:

## Entity Definitions:

**PerpInd** — Individual human actors who carry out the harmful act
- Include: "soldiers", "gunmen", "troops", "guerrillas", person names, "attackers"
- Copy the EXACT span as it appears (e.g., "GUATEMALAN ARMY TROOPS")
- Do NOT break down compound phrases unless they clearly separate roles

**PerpOrg** — Organizations/groups responsible for the harmful act  
- Include: "FMLN", "MRTA", "GUATEMALAN ARMY", "ELN"
- Only extract if the organization is the perpetrator, not the victim
- Avoid extracting victim organizations as perpetrators

**Target** — Concrete inanimate objects/infrastructure that were attacked or intended to be attacked
- Include: specific buildings, vehicles that were attacked, facilities, installations
- Exclude: geographic areas ("downtown Lima", "farming community area")
- Exclude: people or human groups (they belong in Victim)
- Exclude: vehicles used only for transport (only if they were struck/attacked)

**Victim** — People/groups who were harmed (killed, injured, kidnapped, held hostage)
- Include: "peasants", "civilians", "students", person names, "patrol members"
- Copy compound phrases intact: "CIVIL SELF-DEFENSE PATROL (PAC) MEMBERS"
- Include age descriptors: "17-YEAR-OLD"

**Weapon** — Specific instruments or devices used to cause harm
- Include ONLY concrete nouns: "bomb", "rifle", "grenade", "explosives"
- Exclude verbs: NOT "shot", "fired", "shooting", "firing"
- Exclude effects: NOT "explosion" unless no specific weapon is named
- If only action verbs are mentioned (fired, shot), leave empty
- Use "-" only if explicitly stated as no weapon

## Critical Rules:

1. **Role Separation**: Clearly distinguish perpetrator from victim. If "ARMY TROOPS" attack "PAC MEMBERS", then ARMY TROOPS = PerpInd, PAC MEMBERS = Victim
2. **Geographic Areas**: Places like "downtown", "community area", "region" are NOT Targets unless they house a specific attacked facility
3. **Verbatim Extraction**: Copy spans exactly as they appear, maintain original capitalization and punctuation
4. **No Weapon Verbs**: If text only mentions "fired at them" or "shot", do not extract these as weapons
5. **Order by First Mention**: List entities in order of their first appearance in the text

Return only the five arrays via the tool call.
""".strip()



USER_PROMPT_TEMPLATE = (
    "Document ID: {docid}\n\n"
    "Document:\n"
    "\"\"\"{doctext}\"\"\"\n\n"
    "Extract these fields as exact spans:\n"
    "- PerpInd  (agent of the harmful act: “soldiers”, “gunmen”, names)\n"
    "- PerpOrg  (organization/group: “FMLN”, “MRTA”)\n"
    "- Target   (thing attacked: sites, vehicles, buildings, institutions)\n"
    "- Victim   (harmed parties only: killed/injured/kidnapped/hostage)\n"
    "- Weapon   (nouns only: instrument/device; no verbs/effects)\n"
)


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

# ---------- Tiny helpers ----------
def setup_logging(debug: bool):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

def load_json_list(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a list")
    return data

def dedupe_keep_order(items: List[str]) -> List[str]:
    seen, out = set(), []
    for x in items or []:
        if not isinstance(x, str):
            continue
        v = " ".join(x.split()).strip()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out

def sanitize_extraction(e: EntityExtraction) -> EntityExtraction:
    # Drop obvious verb-only Weapon mistakes like "shot"/"fired"
    bad_weapon = {"shot", "fired", "shooting", "firing"}
    weapons = [w for w in e.Weapon if w.strip().lower() not in bad_weapon]
    return EntityExtraction(
        PerpInd=dedupe_keep_order(e.PerpInd),
        PerpOrg=dedupe_keep_order(e.PerpOrg),
        Target=dedupe_keep_order(e.Target),
        Victim=dedupe_keep_order(e.Victim),
        Weapon=dedupe_keep_order(weapons),
    )

def examples_to_block(
    examples: List[Dict[str, Any]],
    examples_per_type: int,
    max_chars: int,
    seed: Optional[int] = None
) -> str:
    """
    Render few-shot examples as plain text Q/A pairs (no tool calls), grouped by incident_type.
    Each A: contains only the five fields we evaluate.
    """
    if seed is not None:
        random.seed(seed)
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for ex in examples:
        itype = (ex.get("incident_type") or "UNKNOWN").upper()
        by_type.setdefault(itype, []).append(ex)

    keep_lines: List[str] = []
    for itype, bucket in by_type.items():
        random.shuffle(bucket)
        for ex in bucket[:max(0, examples_per_type)]:
            txt = (ex.get("text") or "").strip()
            ans = ex.get("answers") or {}
            obj = {k: ans.get(k, []) for k in ENTITY_FIELDS}
            keep_lines.append("Q: " + txt + "\nA: " + json.dumps(obj, ensure_ascii=False))

    if not keep_lines:
        return ""
    block = "Examples\n" + "\n\n".join(keep_lines)
    if max_chars and len(block) > max_chars:
        block = block[:max_chars] + "\n…"
    return block

# ---------- OpenAI call with retries ----------
def call_openai_tool(
    client: OpenAI,
    model: str,
    docid: str,
    doctext: str,
    examples_block: str = "",
    temperature: float = 0.0,
    max_retries: int = 6,
    retry_base: float = 1.7,
) -> EntityExtraction:
    user_prompt = USER_PROMPT_TEMPLATE.format(docid=docid, doctext=doctext)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + ("\n\n" + examples_block if examples_block else "")},
        {"role": "user", "content": user_prompt},
    ]

    attempt = 0
    while True:
        try:
            chat = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages,
                tools=TOOLS,
                tool_choice={"type": "function", "function": {"name": "save_extractions"}},
            )
            choice = chat.choices[0]
            tool_calls = getattr(choice.message, "tool_calls", None) or []
            if tool_calls:
                args_str = tool_calls[0].function.arguments or "{}"
                try:
                    data = json.loads(args_str)
                    return sanitize_extraction(EntityExtraction(**data))
                except (json.JSONDecodeError, ValidationError) as ve:
                    logging.warning(f"[{docid}] Invalid tool arguments: {ve}; returning empty.")
                    return EntityExtraction()
            # Fallback: model returned JSON in content instead of tool-call
            content = (choice.message.content or "").strip()
            if content.startswith("{"):
                try:
                    data = json.loads(content)
                    return sanitize_extraction(EntityExtraction(**data))
                except Exception:
                    pass
            logging.warning(f"[{docid}] No tool call and no JSON content; returning empty.")
            return EntityExtraction()

        except (RateLimitError, APIConnectionError) as e:
            attempt += 1
            if attempt > max_retries:
                logging.error(f"[{docid}] Giving up after {attempt} attempts: {e}")
                return EntityExtraction()
            sleep_s = retry_base ** attempt
            logging.warning(f"[{docid}] Transient error {type(e).__name__}: {e} — retrying in {sleep_s:.1f}s...")
            time.sleep(sleep_s)
        except APIError as e:
            # Retry 5xx; don't retry 4xx (besides rate limit above)
            attempt += 1
            if getattr(e, "status_code", 0) and int(e.status_code) >= 500 and attempt <= max_retries:
                sleep_s = retry_base ** attempt
                logging.warning(f"[{docid}] Server error {e.status_code}: {e} — retrying in {sleep_s:.1f}s...")
                time.sleep(sleep_s)
                continue
            logging.error(f"[{docid}] APIError: {e}")
            return EntityExtraction()
        except BadRequestError as e:
            # Usually prompt/message shape issues; don't loop forever
            logging.error(f"[{docid}] BadRequestError: {e}")
            return EntityExtraction()
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
    ap.add_argument("--sleep", type=float, default=0.4, help="Seconds to sleep between documents")
    ap.add_argument("--pretty", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--progress", action="store_true", help="Print [i/total] counters")

    # Few-shot examples (Option A: plain Q/A inside system prompt)
    ap.add_argument("--examples-file", default=None, help="examples.json with [{incident_type, text, answers}]")
    ap.add_argument("--examples-per-type", type=int, default=3, help="max examples per incident type")
    ap.add_argument("--example-max-chars", type=int, default=1400, help="truncate examples block to this many chars")
    ap.add_argument("--fewshot-seed", type=int, default=None, help="seed for shuffling examples")

    # Optional inline eval via your BLEURT script
    ap.add_argument("--eval", action="store_true", help="Run BLEURT eval after extraction")
    ap.add_argument("--eval-model", default="Elron/bleurt-large-512")
    ap.add_argument("--eval-threshold", type=float, default=0.5)
    args = ap.parse_args()

    setup_logging(args.debug)

    if not os.getenv("OPENAI_API_KEY"):
        logging.error("OPENAI_API_KEY is not set.")
        sys.exit(1)

    client = OpenAI()

    in_path, out_path = Path(args.dataset), Path(args.out)
    try:
        docs = load_json_list(in_path)
    except Exception:
        logging.exception("Failed to read dataset:")
        sys.exit(1)

    # Build few-shot examples block (as plain text)
    examples_block = ""
    if args.examples_file:
        try:
            ex_data = load_json_list(Path(args.examples_file))
            examples_block = examples_to_block(
                ex_data, args.examples_per_type, args.example_max_chars, seed=args.fewshot_seed
            )
            logging.info(f"Loaded {len(ex_data)} examples from {args.examples_file}")
        except Exception:
            logging.exception("Failed to load examples; continuing without few-shot examples.")
            examples_block = ""

    outputs: List[Dict[str, Any]] = []
    had_error = False

    total = len(docs)
    logging.info(f"Starting extraction on {total} docs with model '{args.model_id}'")
    for idx, d in enumerate(docs, start=1):
        docid = d.get("docid")
        text = d.get("doctext") or ""
        if not docid or not text:
            logging.warning(f"Skipping invalid doc: {d.get('docid')}")
            continue

        prefix = f"[{idx}/{total}] " if args.progress else ""
        logging.info(f"{prefix}→ Extracting {docid} (few-shot={'on' if examples_block else 'off'}) ...")

        t0 = time.time()
        err: Optional[str] = None
        answers = EntityExtraction().model_dump()

        try:
            extraction = call_openai_tool(
                client=client,
                model=args.model_id,
                docid=docid,
                doctext=text,
                examples_block=examples_block,
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
        logging.info(f"{prefix}✓ {docid} in {dt_ms} ms")
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
        # Try to locate eval_bleurt_spans.py in CWD or alongside this file
        candidates = [
            Path.cwd() / "eval_bleurt_spans.py",
            Path(__file__).parent / "eval_bleurt_spans.py",
        ]
        evaluator = next((p for p in candidates if p.exists()), None)
        if not evaluator:
            logging.error("eval_bleurt_spans.py not found. Skipping eval.")
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
