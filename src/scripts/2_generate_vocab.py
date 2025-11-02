import pickle
from pathlib import Path

def generate_vocab_from_corpus(corpus_path: Path, output_dir: Path):
    print(f"Loading corpus from {corpus_path}...")
    
    unique_tokens = set()
    total_sequences_count = 0

    with open(corpus_path, "rb") as f:
        while True:
            try:
                sequences = pickle.load(f)
                for sequence in sequences:
                    total_sequences_count += 1
                    for token in sequence:
                        unique_tokens.add(token)
            except EOFError:
                break
    
    sorted_tokens = sorted(list(unique_tokens))

    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    all_tokens = special_tokens + sorted_tokens
    
    output_dir.mkdir(exist_ok=True)
    vocab_file_path = output_dir / "vocab.txt"

    print(f"Processed {total_sequences_count} total sequences")
    print(f"Found {len(all_tokens)} unique tokens")
    
    with open(vocab_file_path, "w", encoding="utf-8") as f:
        for token in all_tokens:
            f.write(token + "\n")
            
    print(f"Vocabulary file successfully created at: {vocab_file_path.resolve()}")

if __name__ == "__main__":
    CORPUS_PATH = Path("/home/tommy/Project/PcodeBERT/outputs/preprocessed/pcode_corpus_x86_64_new_data.pkl")
    OUTPUT_DIR = Path("/home/tommy/Project/PcodeBERT/outputs/tokenizer_new_data")

    generate_vocab_from_corpus(CORPUS_PATH, OUTPUT_DIR)
