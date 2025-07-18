from utils import load_muc4, call_fill_template_api

def main():
    data = load_muc4(n=1)  # Load just one entry for mock test

    for example in data:
        print("\n📄 Transcript:\n", example["text"])
        
        result = call_fill_template_api(example["text"], mock=True)
        if result:
            print("✅ Mocked extraction result:\n", result)
        else:
            print("❌ API call failed or returned no data.")

if __name__ == "__main__":
    main()
