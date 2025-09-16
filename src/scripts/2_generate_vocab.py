import pickle
from pathlib import Path

def generate_vocab_from_corpus(corpus_path: Path, output_dir: Path):
    print(f"Loading corpus from {corpus_path}...")
    with open(corpus_path, "rb") as f:
        corpus = pickle.load(f)

    unique_tokens = set()
    for sentence in corpus:
        for token in sentence:
            unique_tokens.add(token)
    
    sorted_tokens = sorted(list(unique_tokens))

    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    all_vocab = special_tokens + sorted_tokens
    
    output_dir.mkdir(exist_ok=True)
    vocab_file_path = output_dir / "vocab.txt"

    print(f"Found {len(sorted_tokens)} unique tokens.")
    print(f"Total vocabulary size (including special tokens): {len(all_vocab)}")
    
    with open(vocab_file_path, "w", encoding="utf-8") as f:
        for token in all_vocab:
            f.write(token + "\n")
            
    print(f"Vocabulary file successfully created at: {vocab_file_path.resolve()}")

if __name__ == "__main__":
    CORPUS_PATH = Path("/home/tommy/Projects/PcodeBERT/outputs/preprocessed/pcode_corpus_x86_64.pkl")
    OUTPUT_DIR = Path("/home/tommy/Projects/PcodeBERT/outputs/tokenizer") 

    generate_vocab_from_corpus(CORPUS_PATH, OUTPUT_DIR)