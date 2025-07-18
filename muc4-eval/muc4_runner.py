from utils import load_local_muc4_documents, call_fill_template_api
import json

def main():
    print("📥 Loading MUC-4 test documents...")
    docs = load_local_muc4_documents("data/muc4/muc34/TASK/CORPORA/tst2-muc4", limit=5)

    print(f"📂 Loaded {len(docs)} SGML docs from tst2-muc4\n")

    results = []

    for i, text in enumerate(docs):
        print(f"\n--- DOC {i + 1} ---")
        print(text[:500].strip() + "...\n")  # Print a snippet of the doc

        result = call_fill_template_api(text, mock=False)  # ✅ Real LLM call
        if result:
            print("✅ Extraction result:")
            print(json.dumps(result, indent=2))
            results.append({
                "doc_index": i,
                "input_text": text,
                "result": result
            })
        else:
            print("❌ API call failed.")
            results.append({
                "doc_index": i,
                "input_text": text,
                "result": None
            })

    # Save results
    with open("muc4_results.json", "w") as f:
        json.dump(results, f, indent=2)
        print("\n💾 Saved results to muc4_results.json")

if __name__ == "__main__":
    main()
