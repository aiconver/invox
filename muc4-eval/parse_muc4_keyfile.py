import re
import json

def clean(val):
    if not val or val.strip() in {"-", "*"}:
        return None
    return val.strip(' "').strip()

def parse_muc4_keyfile(path):
    with open(path, "r", encoding="latin1") as f:
        lines = f.readlines()

    gold = {}
    current = {}

    for line in lines:
        line = line.strip()

        if line.startswith("0.  MESSAGE: ID"):
            if current:
                gold[current["doc_id"]] = current
            doc_id = line.split()[-1]
            current = {
                "doc_id": doc_id,
                "incident_type": None,
                "perpetrator": None,
                "victim": None,
                "weapon": None,
                "location": None,
                "date": None
            }

        elif "INCIDENT: TYPE" in line:
            match = re.search(r'INCIDENT: TYPE\s+(.*)', line)
            val = clean(match.group(1).split(":")[0]) if match else None
            if val:
                current["incident_type"] = val.lower()

        elif "PERP: INDIVIDUAL ID" in line or "PERP: ORGANIZATION ID" in line:
            matches = re.findall(r'"([^"]+)"', line)
            if matches and not current["perpetrator"]:
                current["perpetrator"] = clean(matches[0])

        elif "HUM TGT: DESCRIPTION" in line:
            match = re.search(r'"([^"]+)"', line)
            if match:
                current["victim"] = clean(match.group(1))

        elif "INSTRUMENT TYPE" in line:
            match = re.search(r'"([^"]+)"', line)
            if match:
                current["weapon"] = clean(match.group(1))
            else:
                parts = line.split(":")
                if len(parts) > 1:
                    current["weapon"] = clean(parts[-1])

        elif "INCIDENT: LOCATION" in line:
            match = re.search(r'INCIDENT: LOCATION\s+(.*)', line)
            if match:
                loc = match.group(1).split(":")[0]
                current["location"] = clean(loc.title())

        elif "INCIDENT: DATE" in line:
            match = re.search(r'INCIDENT: DATE\s+(.*)', line)
            if match:
                full_date = match.group(1).replace("(", "").replace(")", "").replace("-", "").strip()
                current["date"] = clean(full_date)

    if current:
        gold[current["doc_id"]] = current

    print(f"✅ Parsed {len(gold)} unique gold entries")
    return list(gold.values())

if __name__ == "__main__":
    gold = parse_muc4_keyfile("data/muc4_gold.key")
    with open("muc4_gold.json", "w", encoding="utf-8") as f:
        json.dump(gold, f, indent=2)
    print("💾 Saved gold data to muc4_gold.json")
