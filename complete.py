import torch
from torch import nn
from tokenizers import ByteLevelBPETokenizer
from train_code import TinyGPT  # import model class

# === Load Tokenizer ===
tokenizer = ByteLevelBPETokenizer(
    "tokenizer/vocab.json",
    "tokenizer/merges.txt"
)

VOCAB_SIZE = tokenizer.get_vocab_size()
BLOCK_SIZE = 128
EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 6

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device}")

# === Load Trained Model ===
model = TinyGPT()
model.load_state_dict(torch.load("model/gpt_epoch5.pt", map_location=device))
model.to(device)
model.eval()

# === Completion Function ===
def generate_code(prompt, max_new_tokens=50):
    input_ids = tokenizer.encode(prompt).ids
    x = torch.tensor([input_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(x)
            next_id = torch.argmax(logits[:, -1, :], dim=-1)
            x = torch.cat([x, next_id.unsqueeze(0)], dim=1)

    output_text = tokenizer.decode(x[0].tolist())
    return output_text

# === Run Completion ===
prompt = "def greet(name):\n    print(f\"Hello"
output = generate_code(prompt)

print("\n🔹 INPUT PROMPT:")
print(prompt)
print("\n🔸 GENERATED COMPLETION:")
print(output)
