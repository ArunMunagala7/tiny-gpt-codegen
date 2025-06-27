# train_tokenizer.py

from tokenizers import ByteLevelBPETokenizer
import os

def train_tokenizer():
    print("🧠 Starting tokenizer training...")
    tokenizer = ByteLevelBPETokenizer()

    # Train from cleaned Python code
    tokenizer.train(
        files="data/python_clean_code.txt",
        vocab_size=32000,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )

    # Save tokenizer files
    os.makedirs("tokenizer", exist_ok=True)
    tokenizer.save_model("tokenizer")

    print("✅ Tokenizer training complete.")
    print("📁 Files saved to: tokenizer/")
    print("  - vocab.json")
    print("  - merges.txt")

    # Sanity check
    enc = tokenizer.encode("def greet(name): print('Hello ' + name)")
    print("\n🔍 Sample encoding:")
    print(enc.tokens)
    print("Token IDs:", enc.ids)

if __name__ == "__main__":
    train_tokenizer()
