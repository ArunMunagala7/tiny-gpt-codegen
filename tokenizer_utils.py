from tokenizers import ByteLevelBPETokenizer

def load_tokenizer():
    return ByteLevelBPETokenizer(
        "tokenizer/vocab.json",
        "tokenizer/merges.txt"
    )
