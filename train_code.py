# train_code.py

import torch
from torch import nn, optim
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm
import os
import math

# Load tokenizer
tokenizer = ByteLevelBPETokenizer(
    "tokenizer/vocab.json",
    "tokenizer/merges.txt"
)

# Model Config
VOCAB_SIZE = tokenizer.get_vocab_size()
BLOCK_SIZE = 128  # Max sequence length
EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 6
BATCH_SIZE = 16
EPOCHS = 5
LR = 3e-4

# Auto-select device
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Using device: {device}")

# Simple GPT block
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_embed = nn.Parameter(torch.zeros(1, BLOCK_SIZE, EMBED_DIM))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=NUM_HEADS),
            num_layers=NUM_LAYERS
        )
        self.ln = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def forward(self, x):
        token_embeddings = self.embed(x)  # (B, T, E)
        x = token_embeddings + self.pos_embed[:, :x.size(1), :]
        x = self.transformer(x)
        x = self.ln(x)
        logits = self.head(x)
        return logits

model = TinyGPT().to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# Load and tokenize dataset
print("📄 Loading and encoding dataset...")
with open("data/python_clean_code.txt", "r", encoding="utf-8") as f:
    data = f.read()

ids = tokenizer.encode(data).ids
print(f"✅ Total tokens: {len(ids):,}")

# Turn into training sequences
def get_batch():
    ix = torch.randint(0, len(ids) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([torch.tensor(ids[i:i+BLOCK_SIZE]) for i in ix])
    y = torch.stack([torch.tensor(ids[i+1:i+BLOCK_SIZE+1]) for i in ix])
    return x.to(device), y.to(device)

# Training loop
print("🏋️ Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for step in tqdm(range(1000), desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x, y = get_batch()
        logits = model(x)
        loss = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / 1000
    print(f"📉 Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    # Save model
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), f"model/gpt_epoch{epoch+1}.pt")

print("✅ Training complete. Checkpoints saved in /model/")
