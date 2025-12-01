import pickle
from gensim.models import Word2Vec
from pathlib import Path
from typing import List

from preprocessing import process_directory_by_architecture

def load_corpus(corpus_path: Path) -> List[List[str]]:
    """Load corpus from pickle file"""
    with open(corpus_path, 'rb') as f:
        return pickle.load(f)

def train_skipgram(corpus: List[List[str]], output_path: Path, arch_name: str):
    """Train Skip-gram model"""
    print(f"\nTraining Skip-gram for {arch_name}")
    print(f"Corpus size: {len(corpus)} sentences")
    
    model = Word2Vec(
        sentences=corpus,
        vector_size=200,
        window=5,
        min_count=1,
        workers=4,
        sg=1, 
        epochs=5
    )
    
    # Save model
    model_file = output_path / f'skipgram_{arch_name}.model'
    vectors_file = output_path / f'skipgram_{arch_name}_vectors.kv'
    
    model.save(str(model_file))
    model.wv.save(str(vectors_file))
    
    print(f"Model saved: {model_file}")
    print(f"Vocabulary size: {len(model.wv)}")
    print(f"Sample words: {list(model.wv.index_to_key[:10])}")

def main():
    # Paths
    raw_data_path = Path('/home/tommy/Project/PcodeBERT/outputs/preprocessed/results_BinKit_instruction/results')
    output_dir = Path('/home/tommy/Project/PcodeBERT/experiment/outputs')
    model_dir = Path('/home/tommy/Project/PcodeBERT/experiment/model/Unimap/MAIE')
    
    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("Processing data and creating corpus by architecture...")
    arch_corpus = process_directory_by_architecture(raw_data_path, output_dir)
    
    # Train Skip-gram for each architecture
    for arch in ['x86_64', 'arm_32']:
        corpus_file = output_dir / f'corpus_{arch}.pkl'
        
        if corpus_file.exists():
            print(f"\n{'='*60}")
            print(f"Processing {arch.upper()} architecture")
            print(f"{'='*60}")
            
            # Load corpus
            corpus = load_corpus(corpus_file)
            
            if corpus:
                # Train Skip-gram model
                train_skipgram(corpus, model_dir, arch)
            else:
                print(f"No data found for {arch}")
        else:
            print(f"Corpus file not found: {corpus_file}")
    
    print("\n" + "="*60)
    print("Training completed for all architectures!")

if __name__ == "__main__":
    main()
