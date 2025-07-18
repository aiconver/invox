# muc4_runner.py

import json
from utils import load_local_muc4_documents, call_fill_template_api

def main():
    print("📥 Loading MUC-4 test documents...")
    docs = load_local_muc4_documents("data/muc4/muc34/TASK/CORPORA/tst2-muc4", limit=100)
    print(f"📂 Loaded {len(docs)} SGML docs from tst2-muc4\n")

    results = []

    for i, text in enumerate(docs):
        doc_id_match = text.strip().split("\n", 1)[0]
        doc_id = doc_id_match.strip() if doc_id_match.startswith("TST") else f"TST2-MUC4-{1000+i}"
        print(f"\n--- DOC {i+1} ---\n{doc_id}\n")
        print(text[:600] + "...\n")  # Print first 600 chars

        result = call_fill_template_api(text, mock=False)
        if result:
            print("✅ Extraction result:")
            print(json.dumps(result, indent=2))
            results.append({
                "doc_id": doc_id,
                "text": text,
                "filledTemplate": result.get("filledTemplate", {}),
                "confidence": result.get("confidence"),
                "missingFields": result.get("missingFields", []),
                "warnings": result.get("warnings", [])
            })
        else:
            print("❌ API call failed.")
            results.append({
                "doc_id": doc_id,
                "text": text,
                "filledTemplate": {},
                "confidence": 0.0,
                "missingFields": ["incident_type", "perpetrator", "victim", "weapon", "location", "date"],
                "warnings": ["API call failed"]
            })

    # Save results
    with open("muc4_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        print("\n💾 Saved results to muc4_results.json")

if __name__ == "__main__":
    main()
