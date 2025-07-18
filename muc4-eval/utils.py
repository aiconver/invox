import re
import requests
import uuid

def load_local_muc4_documents(path, limit=None):
    texts = []
    current_doc = None

    with open(path, 'r', encoding='latin1') as f:
        for line in f:
            line = line.strip()

            # Start of a new doc
            if re.match(r"^TST\d-MUC4-\d{4}$", line):
                if current_doc:
                    texts.append(current_doc.strip())
                current_doc = line + "\n"
            elif current_doc is not None:
                current_doc += line + "\n"

        # Append the final doc
        if current_doc:
            texts.append(current_doc.strip())

    print(f"📂 Loaded {len(texts)} docs from {path}")
    return texts[:limit] if limit else texts

def call_fill_template_api(text, mock=True):
    if mock:
        return {
            "message": "Mocked: Template filled successfully!",
            "filledTemplate": {
                "incident_type": "bombing",
                "perpetrator": "The Revolutionary Front",
                "victim": "military personnel",
                "weapon": "improvised explosive device",
                "location": "Bogotá",
                "date": "March 5, 1987"
            },
            "confidence": 0.93,
            "missingFields": [],
            "warnings": []
        }

    payload = {
        "jsonrpc": "2.0",
        "method": "ai.fillTemplate",
        "params": {
            "transcript": text,
            "templateDefinition": {
                "templateName": "muc4-terrorist-incident",
                "fields": {
                    "incident_type": {"type": "string", "required": True},
                    "perpetrator": {"type": "string", "required": True},
                    "victim": {"type": "string"},
                    "weapon": {"type": "string"},
                    "location": {"type": "string"},
                    "date": {"type": "string"}
                }
            }
        },
        "id": str(uuid.uuid4())
    }

    try:
        res = requests.post("http://localhost:3000/rpc", json=payload)
        res.raise_for_status()
        result = res.json()
        return result["result"]
    except Exception as e:
        print("❌ RPC call failed:", e)
        if hasattr(e, 'response') and e.response is not None:
            print("❌ Status:", e.response.status_code)
            print("❌ Body:", e.response.text)
        return None
