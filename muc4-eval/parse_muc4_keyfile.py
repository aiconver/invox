# parse_muc4_keyfile.py

import re
import json

def parse_muc4_keyfile(path):
    with open(path, "r", encoding="latin1") as f:
        lines = f.readlines()

    docs = []
    current = {}
    
    for line in lines:
        line = line.strip()
        if line.startswith("0.  MESSAGE: ID"):
            # Save previous doc
            if current:
                docs.append(current)
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
            val = line.split("TYPE")[-1].strip().split()[0].lower()
            current["incident_type"] = val if val != "-" else None

        elif "PERP: INDIVIDUAL ID" in line or "PERP: ORGANIZATION ID" in line:
            match = re.search(r'"([^"]+)"', line)
            if match and not current["perpetrator"]:
                current["perpetrator"] = match.group(1)

        elif "HUM TGT: DESCRIPTION" in line:
            match = re.search(r'"([^"]+)"', line)
            if match:
                current["victim"] = match.group(1)

        elif "INSTRUMENT TYPE" in line:
            match = re.search(r'"([^"]+)"', line)
            if match:
                current["weapon"] = match.group(1)

        elif "INCIDENT: LOCATION" in line:
            location = line.split("LOCATION")[-1].strip()
            current["location"] = location.split(":")[0].title() if location and location != "-" else None

        elif "INCIDENT: DATE" in line:
            val = line.split("DATE")[-1].strip().split()[0]
            current["date"] = val if val != "-" else None

    if current:
        docs.append(current)

    print(f"✅ Parsed {len(docs)} gold entries")
    return docs

if __name__ == "__main__":
    gold = parse_muc4_keyfile("data/muc4_gold.key")
    with open("muc4_gold.json", "w", encoding="utf-8") as f:
        json.dump(gold, f, indent=2)
    print("💾 Saved gold data to muc4_gold.json")
