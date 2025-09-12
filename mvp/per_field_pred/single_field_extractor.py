#!/usr/bin/env python3
# single_field_extractor.py
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

# ---------- Field-Specific System Prompts ----------
FIELD_PROMPTS = {
    "PerpInd": """
You are an information extraction assistant specializing in identifying INDIVIDUAL PERPETRATORS from violent incident reports.

Your task: Extract only PerpInd entities - individual human actors who carry out harmful acts.

## PerpInd Definition:
Individual human actors who carry out the harmful act:
- Include: "soldiers", "gunmen", "troops", "guerrillas", person names, "attackers", "assailants"
- Copy the EXACT span as it appears (e.g., "GUATEMALAN ARMY TROOPS", "NORTHEASTERN COBAN BASE TROOPS")
- Do NOT break down compound phrases - keep them intact
- Include specific military units: "COBAN BASE TROOPS", "NATIONAL GUARD PATROL"
- Include descriptive groups: "ARMED INDIVIDUALS", "MASKED MEN", "PAID ASSASSINS"

## What NOT to include:
- Organizations (those go in PerpOrg): "GUATEMALAN ARMY", "FMLN", "MRTA"
- Victims or people being harmed
- Vehicles, weapons, or locations

## Critical Rules:
1. Copy spans EXACTLY as they appear in the text
2. Include all variations and mentions of perpetrator individuals/groups
3. Maintain original capitalization and punctuation
4. Order by first appearance in text
5. Focus ONLY on who carried out the harmful act

Extract all PerpInd entities from the document.
""".strip(),

    "PerpOrg": """
You are an information extraction assistant specializing in identifying PERPETRATOR ORGANIZATIONS from violent incident reports.

Your task: Extract only PerpOrg entities - organizations/groups responsible for harmful acts.

## PerpOrg Definition:
Organizations/groups responsible for the harmful act:
- Include: "FMLN", "MRTA", "GUATEMALAN ARMY", "ELN", "TUPAC AMARU REVOLUTIONARY MOVEMENT"
- Include: "THE EXTRADITABLES", "FARABUNDO MARTI NATIONAL LIBERATION FRONT"
- Only extract if the organization is the PERPETRATOR, not the victim
- Avoid extracting victim organizations (like "PAC" if they were attacked)

## What NOT to include:
- Individual people (those go in PerpInd): "soldiers", "troops", "gunmen"
- Victim organizations that were attacked
- Government agencies that were victims
- Locations or weapons

## Critical Rules:
1. Copy organization names EXACTLY as they appear
2. Only include organizations that carried out the harmful act
3. If unsure whether an organization is perpetrator or victim, check the context carefully
4. Include both full names and abbreviations if both appear
5. Order by first appearance in text

Extract all PerpOrg entities from the document.
""".strip(),

    "Target": """
You are an information extraction assistant specializing in identifying TARGETS from violent incident reports.

Your task: Extract only Target entities - concrete inanimate objects/infrastructure that were attacked.

## Target Definition:
Concrete inanimate objects/infrastructure that were attacked or intended to be attacked:
- Include: specific buildings ("embassy", "newspaper facilities", "community center")
- Include: vehicles that were attacked ("jeep", "armored vehicle", "motorcycle", "bus")
- Include: infrastructure ("power lines", "telephone boxes", "installations")
- Include: specific facilities ("military garrison", "police station")

## What NOT to include:
- Geographic areas: "downtown Lima", "farming community area", "region", "border area"
- People or human groups (they belong in Victim)
- Vehicles used only for transport (only if they were struck/attacked)
- Abstract concepts or general locations

## Critical Rules:
1. Must be CONCRETE and SPECIFIC - not vague geographic areas
2. Must have been actually attacked or intended for attack
3. Copy spans EXACTLY as they appear
4. If a vehicle was just used for transport, don't include it
5. If a vehicle was attacked/destroyed, include it
6. Order by first appearance in text

Extract all Target entities from the document.
""".strip(),

    "Victim": """
You are an information extraction assistant specializing in identifying VICTIMS from violent incident reports.

Your task: Extract only Victim entities - people/groups who were harmed.

## Victim Definition:
People/groups who were harmed (killed, injured, kidnapped, held hostage):
- Include: "peasants", "civilians", "students", person names, "patrol members"
- Include: compound phrases intact: "CIVIL SELF-DEFENSE PATROL (PAC) MEMBERS"
- Include: age descriptors: "17-YEAR-OLD", "child", "teenager"
- Include: job titles: "driver", "bodyguard", "ambassador", "journalist"
- Include: any person who was killed, wounded, kidnapped, or held hostage

## What NOT to include:
- Perpetrators who carried out the attack
- Organizations (unless referring to people within them being harmed)
- Vehicles, buildings, or inanimate objects

## Critical Rules:
1. Copy compound phrases EXACTLY: "CIVIL SELF-DEFENSE PATROL (PAC) MEMBERS"
2. Include all variations of how victims are referred to
3. Include both general terms ("peasants") and specific individuals ("Carlos Julio Bonilla")
4. Include people who were harmed in any way (killed, wounded, kidnapped, threatened)
5. Maintain original capitalization and punctuation
6. Order by first appearance in text

Extract all Victim entities from the document.
""".strip(),

    "Weapon": """
You are an information extraction assistant specializing in identifying WEAPONS from violent incident reports.

Your task: Extract only Weapon entities - specific instruments/devices used to cause harm.

## Weapon Definition:
Specific instruments or devices used to cause harm:
- Include ONLY concrete nouns: "bomb", "rifle", "grenade", "explosives", "pistol"
- Include specific weapon types: "automatic weapons", "mortars", ".45-caliber pistol"
- Include weapon descriptors: "very powerful explosives", "fragmentation grenades"

## What NOT to include:
- Action verbs: NOT "shot", "fired", "shooting", "firing", "exploded"
- Effects or results: NOT "explosion", "blast", "gunfire" (unless no specific weapon is named)
- General violence terms: NOT "attack", "assault", "violence"
- If only action verbs are mentioned ("fired at them", "shot"), leave empty

## Special Cases:
- If text says "explosion" but no specific explosive device is mentioned, you may include "explosion"
- If text mentions both the weapon and the action ("fired rifles"), include only "rifles"
- Use "-" only if explicitly stated as no weapon used

## Critical Rules:
1. ONLY concrete weapon nouns - never verbs
2. Copy weapon names EXACTLY as they appear
3. Include quantity/descriptive phrases: "two fragmentation grenades"
4. If only shooting verbs are mentioned without weapon names, extract nothing
5. Order by first appearance in text

Extract all Weapon entities from the document.
""".strip()
}

# ---------- User Prompt Templates ----------
USER_PROMPT_TEMPLATES = {
    "PerpInd": "Document ID: {docid}\n\nDocument:\n\"\"\"{doctext}\"\"\"\n\nExtract all PerpInd entities (individual human actors who carried out the harmful act):",
    "PerpOrg": "Document ID: {docid}\n\nDocument:\n\"\"\"{doctext}\"\"\"\n\nExtract all PerpOrg entities (organizations/groups responsible for the harmful act):",
    "Target": "Document ID: {docid}\n\nDocument:\n\"\"\"{doctext}\"\"\"\n\nExtract all Target entities (concrete objects/infrastructure that were attacked):",
    "Victim": "Document ID: {docid}\n\nDocument:\n\"\"\"{doctext}\"\"\"\n\nExtract all Victim entities (people/groups who were harmed):",
    "Weapon": "Document ID: {docid}\n\nDocument:\n\"\"\"{doctext}\"\"\"\n\nExtract all Weapon entities (specific instruments/devices used to cause harm):"
}

# ---------- Field-Specific Tool Schemas ----------
def get_tool_schema(field_name: str):
    return [
        {
            "type": "function",
            "function": {
                "name": f"save_{field_name.lower()}",
                "description": f"Return extracted {field_name} entities as an array of strings.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        field_name: {"type": "array", "items": {"type": "string"}}
                    },
                    "required": [field_name],
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

def sanitize_weapons(weapons: List[str]) -> List[str]:
    """Remove obvious verb-only weapon mistakes"""
    bad_weapon = {"shot", "fired", "shooting", "firing", "exploded", "attacked"}
    return [w for w in weapons if w.strip().lower() not in bad_weapon]

def build_field_examples(field_name: str, examples: List[Dict[str, Any]], max_examples: int = 3) -> str:
    """Build field-specific examples"""
    if not examples:
        return ""
    
    lines = []
    count = 0
    for ex in examples:
        if count >= max_examples:
            break
        text = ex.get("text", "").strip()
        answers = ex.get("answers", {})
        field_values = answers.get(field_name, [])
        
        lines.append(f"Q: {text}")
        lines.append(f"A: {json.dumps({field_name: field_values}, ensure_ascii=False)}")
        count += 1
    
    if lines:
        return "Examples:\n" + "\n\n".join(lines)
    return ""

# ---------- Single Field OpenAI Call ----------
def extract_single_field(
    client: OpenAI,
    model: str,
    docid: str,
    doctext: str,
    field_name: str,
    previous_results: Dict[str, List[str]],
    examples_block: str = "",
    temperature: float = 0.0,
    max_retries: int = 6,
    retry_base: float = 1.7,
) -> List[str]:
    """Extract a single field type from the document"""
    
    # Build context from previous results
    context = ""
    if previous_results:
        context_parts = []
        for prev_field, prev_values in previous_results.items():
            if prev_values:
                context_parts.append(f"{prev_field}: {prev_values}")
        if context_parts:
            context = f"\n\nPrevious extractions from this document:\n" + "\n".join(context_parts) + "\n"
    
    system_prompt = FIELD_PROMPTS[field_name]
    if examples_block:
        system_prompt += "\n\n" + examples_block
    if context:
        system_prompt += context
    
    user_prompt = USER_PROMPT_TEMPLATES[field_name].format(docid=docid, doctext=doctext)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    tools = get_tool_schema(field_name)
    tool_choice = {"type": "function", "function": {"name": f"save_{field_name.lower()}"}}
    
    attempt = 0
    while True:
        try:
            chat = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )
            
            choice = chat.choices[0]
            tool_calls = getattr(choice.message, "tool_calls", None) or []
            
            if tool_calls:
                args_str = tool_calls[0].function.arguments or "{}"
                try:
                    data = json.loads(args_str)
                    field_values = data.get(field_name, [])
                    if field_name == "Weapon":
                        field_values = sanitize_weapons(field_values)
                    return dedupe_keep_order(field_values)
                except (json.JSONDecodeError, ValidationError) as ve:
                    logging.warning(f"[{docid}:{field_name}] Invalid tool arguments: {ve}")
                    return []
            
            # Fallback: check content
            content = (choice.message.content or "").strip()
            if content.startswith("{"):
                try:
                    data = json.loads(content)
                    field_values = data.get(field_name, [])
                    if field_name == "Weapon":
                        field_values = sanitize_weapons(field_values)
                    return dedupe_keep_order(field_values)
                except Exception:
                    pass
            
            logging.warning(f"[{docid}:{field_name}] No valid extraction found")
            return []

        except (RateLimitError, APIConnectionError) as e:
            attempt += 1
            if attempt > max_retries:
                logging.error(f"[{docid}:{field_name}] Giving up after {attempt} attempts: {e}")
                return []
            sleep_s = retry_base ** attempt
            logging.warning(f"[{docid}:{field_name}] Transient error - retrying in {sleep_s:.1f}s...")
            time.sleep(sleep_s)
            
        except APIError as e:
            attempt += 1
            if getattr(e, "status_code", 0) and int(e.status_code) >= 500 and attempt <= max_retries:
                sleep_s = retry_base ** attempt
                logging.warning(f"[{docid}:{field_name}] Server error - retrying in {sleep_s:.1f}s...")
                time.sleep(sleep_s)
                continue
            logging.error(f"[{docid}:{field_name}] APIError: {e}")
            return []
            
        except Exception as e:
            logging.error(f"[{docid}:{field_name}] Unexpected error: {e}")
            return []

# ---------- Sequential Field Extraction ----------
def extract_all_fields(
    client: OpenAI,
    model: str,
    docid: str,
    doctext: str,
    examples: List[Dict[str, Any]] = None,
    temperature: float = 0.0,
    sleep_between_fields: float = 0.1
) -> EntityExtraction:
    """Extract all fields sequentially, sharing context between extractions"""
    
    results = {}
    
    for field_name in ENTITY_FIELDS:
        logging.debug(f"[{docid}] Extracting {field_name}...")
        
        # Build field-specific examples
        examples_block = ""
        if examples:
            examples_block = build_field_examples(field_name, examples, max_examples=3)
        
        # Extract this field
        field_values = extract_single_field(
            client=client,
            model=model,
            docid=docid,
            doctext=doctext,
            field_name=field_name,
            previous_results=results,
            examples_block=examples_block,
            temperature=temperature
        )
        
        results[field_name] = field_values
        logging.debug(f"[{docid}] {field_name}: {len(field_values)} entities")
        
        # Sleep between field extractions
        if sleep_between_fields > 0:
            time.sleep(sleep_between_fields)
    
    return EntityExtraction(**results)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Extract entities one field at a time with ChatGPT")
    ap.add_argument("--dataset", required=True, help="Path to test dataset JSON")
    ap.add_argument("--out", required=True, help="Path to write predictions JSON")
    ap.add_argument("--model-id", default="gpt-4o-mini", help="OpenAI model")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--sleep", type=float, default=0.4, help="Seconds between documents")
    ap.add_argument("--field-sleep", type=float, default=0.1, help="Seconds between fields")
    ap.add_argument("--pretty", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--progress", action="store_true")
    
    # Examples
    ap.add_argument("--examples-file", help="examples.json file")
    
    # Eval
    ap.add_argument("--eval", action="store_true", help="Run BLEURT eval after extraction")
    ap.add_argument("--eval-model", default="Elron/bleurt-large-512")
    ap.add_argument("--eval-threshold", type=float, default=0.5)
    
    args = ap.parse_args()
    
    setup_logging(args.debug)
    
    if not os.getenv("OPENAI_API_KEY"):
        logging.error("OPENAI_API_KEY is not set.")
        sys.exit(1)
    
    client = OpenAI()
    
    # Load dataset
    in_path, out_path = Path(args.dataset), Path(args.out)
    try:
        docs = load_json_list(in_path)
    except Exception:
        logging.exception("Failed to read dataset:")
        sys.exit(1)
    
    # Load examples
    examples = []
    if args.examples_file:
        try:
            examples = load_json_list(Path(args.examples_file))
            logging.info(f"Loaded {len(examples)} examples")
        except Exception:
            logging.exception("Failed to load examples; continuing without them.")
    
    outputs = []
    had_error = False
    total = len(docs)
    
    logging.info(f"Starting single-field extraction on {total} docs (5x API calls per doc)")
    
    for idx, doc in enumerate(docs, start=1):
        docid = doc.get("docid")
        doctext = doc.get("doctext", "")
        
        if not docid or not doctext:
            logging.warning(f"Skipping invalid doc: {docid}")
            continue
        
        prefix = f"[{idx}/{total}] " if args.progress else ""
        logging.info(f"{prefix}→ Processing {docid}...")
        
        t0 = time.time()
        error = None
        
        try:
            extraction = extract_all_fields(
                client=client,
                model=args.model_id,
                docid=docid,
                doctext=doctext,
                examples=examples,
                temperature=args.temperature,
                sleep_between_fields=args.field_sleep
            )
            answers = extraction.model_dump()
            
        except Exception as e:
            logging.exception(f"[{docid}] Extraction failed:")
            error = str(e)
            had_error = True
            answers = EntityExtraction().model_dump()
        
        dt_ms = round((time.time() - t0) * 1000.0, 1)
        
        outputs.append({
            "id": docid,
            "meta": {
                "model_id": args.model_id,
                "extraction_type": "single_field_sequential",
                "timing": {"duration_ms": dt_ms},
                "error": error
            },
            "answers": answers
        })
        
        # Log summary
        total_entities = sum(len(v) for v in answers.values())
        logging.info(f"{prefix}✓ {docid} - {total_entities} entities in {dt_ms} ms")
        
        if args.sleep > 0:
            time.sleep(args.sleep)
    
    # Write results
    try:
        out_path.write_text(
            json.dumps(outputs, indent=2 if args.pretty else None, ensure_ascii=False),
            encoding="utf-8"
        )
        logging.info(f"✅ Wrote {len(outputs)} results to {out_path}")
    except Exception:
        logging.exception("Failed to write output:")
        sys.exit(1)
    
    if had_error:
        logging.warning("Some documents had errors; check logs.")
    
    # Optional evaluation
    if args.eval:
        candidates = [
            Path.cwd() / "eval_bleurt_spans.py",
            Path(__file__).parent / "eval_bleurt_spans.py",
        ]
        evaluator = next((p for p in candidates if p.exists()), None)
        if evaluator:
            cmd = [
                sys.executable, str(evaluator),
                "--gold", str(in_path),
                "--pred", str(out_path),
                "--model", args.eval_model,
                "--threshold", str(args.eval_threshold),
                "--pretty",
            ]
            logging.info("Running BLEURT evaluation...")
            try:
                subprocess.run(cmd, check=True)
            except Exception:
                logging.exception("BLEURT evaluation failed.")
        else:
            logging.error("eval_bleurt_spans.py not found. Skipping evaluation.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Fatal error:")
        sys.exit(1)