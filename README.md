# 🧠 Tiny GPT Code Generator

A minimal GPT-style Transformer that learns to autocomplete Python code — trained entirely from scratch using PyTorch and a Byte-Level BPE tokenizer. Inspired by models like GPT-2, this project demonstrates the end-to-end process of tokenizer training, Transformer training, and code generation with no external pretraining.

---

## 📦 Project Overview

This project contains:

- A custom-trained ByteLevel BPE tokenizer
- A GPT-style Transformer encoder model with learned position embeddings
- Training from scratch on tokenized Python code
- A code completion script that can continue Python snippets token by token

---

## 📁 Directory Structure

```
tiny-gpt-codegen/
│
├── data/
│   └── python_clean_code.txt        # Pre-cleaned training data
│
├── tokenizer/
│   ├── vocab.json                   # Trained BPE vocabulary
│   └── merges.txt                   # Merge rules learned by tokenizer
│
├── model/
│   └── gpt_epoch*.pt                # Saved model checkpoints after training
│
├── train_tokenizer.py              # Trains the ByteLevel BPE tokenizer
├── train_code.py                   # Trains the Transformer model
├── complete.py                     # Inference: generates Python completions
├── preprocess_data.py              # (Optional) data cleaner script
├── requirements.txt                # Dependencies
└── README.md                       # You're reading it!
```

---

## 🚀 How It Works

1. **Train the Tokenizer**  
   Tokenizes character-level Python using Byte-Pair Encoding to learn subword units.

2. **Train the GPT Model**  
   Learns to predict the next token given a sequence of previous tokens using a Transformer Encoder stack.

3. **Run Completions**  
   Takes a code prompt and predicts the next tokens greedily, generating autocompletions.

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/tiny-gpt-codegen.git
cd tiny-gpt-codegen
```

### 2. Create a Virtual Environment (Recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧱 Tokenizer Training (Phase 2)

Trains a ByteLevel BPE tokenizer on cleaned Python source code.

```bash
python train_tokenizer.py
```

This creates:
- `tokenizer/vocab.json`
- `tokenizer/merges.txt`

You'll also see a sample encoding printed to sanity-check the tokens.

---

## 🤖 Model Training (Phase 3)

Trains the GPT-style Transformer model from scratch.

```bash
python train_code.py
```

**Training Details:**
- 6 Transformer blocks
- 8 heads per block
- 256-dimensional embeddings
- 5 epochs over 1000 batches per epoch
- Sequences of 128 tokens
- Optimizer: AdamW

Checkpoints are saved under `/model/`.

---

## 🪄 Code Completion (Phase 6)

To generate code completions from a prompt:

1. Open `complete.py`
2. Modify the `prompt` string (e.g., `"def greet(name):\n    print(f\"Hello"`)
3. Run:

```bash
python complete.py
```

The output will show:

```text
🔷 INPUT PROMPT:
def greet(name):
    print(f"Hello

🔶 GENERATED COMPLETION:
def greet(name):
    print(f"Hello, {name}")
```

(Note: Output quality depends heavily on training size & steps.)

---

## 🧠 Model Architecture

| Component         | Value     |
|------------------|-----------|
| Embedding Dim     | 256       |
| Num Layers        | 6         |
| Num Heads         | 8         |
| Sequence Length   | 128       |
| Vocab Size        | 32,000    |
| Positional Embed  | Learned   |
| Optimizer         | AdamW     |
| Loss Function     | CrossEntropyLoss |

---

## 📚 Example Output

### Prompt:
```python
def greet(name):
    print(f"Hello
```

### Possible Output:
```python
def greet(name):
    print(f"Hello {name}!")
```

Or, based on training:
```python
def greet(name):
    print("Hello, " + name)
```

---

## 🔍 Troubleshooting

- **Stuck on "Loading and encoding dataset..."**  
  Large files or slow tokenization can cause delays. Wait or monitor CPU usage.

- **Output full of quotes/garbage**  
  This usually means not enough training or poor-quality dataset. Train longer or clean better.

---

## 📈 Future Improvements

- Add `top-k`, `top-p`, or temperature sampling for diversity
- Support streaming token generation
- Extend model depth/width for larger corpora
- Fine-tune on high-quality Python snippets or GitHub repos

---

## 📜 License

MIT License

---

## 🧾 Requirements

```txt
torch
tokenizers
tqdm
```

Install manually:

```bash
pip install torch tokenizers tqdm
```

---

## ✨ Credits

- Inspired by Karpathy's nanoGPT and OpenAI's GPT
- Tokenization powered by 🤗 HuggingFace `tokenizers`

---

Happy Hacking! 💻🐍
