# preprocess_data.py

from datasets import load_dataset
import os
import textwrap

def clean_code(code):
    # Keep only normal-looking functions
    if not code.strip().startswith("def"):
        return None
    code = textwrap.dedent(code)
    if len(code) < 20 or len(code.split()) < 5:
        return None
    return code.strip()

def main():
    print("🔽 Loading CodeSearchNet (Python subset)...")
    dataset = load_dataset("code-search-net/code_search_net", "python", split="train")

    print("🧹 Cleaning Python functions...")
    cleaned_funcs = []
    for entry in dataset:
        raw_code = entry.get("func_code_string")
        if raw_code:
            cleaned = clean_code(raw_code)
            if cleaned:
                cleaned_funcs.append(cleaned)

    print(f"✅ Collected {len(cleaned_funcs):,} clean Python functions.")

    os.makedirs("data", exist_ok=True)
    output_path = "data/python_clean_code.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(cleaned_funcs))

    print(f"📁 Saved cleaned code to: {output_path}")

if __name__ == "__main__":
    main()
