import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm

from preprocessing import normalize_instruction, read_filenames_from_csv

# Paths
CSV_PATH = '/home/tommy/Project/PcodeBERT/dataset/csv/merged_adjusted_filtered.csv'
TARGET_DATA_DIR = Path('/home/tommy/Project/PcodeBERT/outputs/results')
OUTPUT_DATA_DIR = Path('/home/tommy/Project/PcodeBERT/experiment/outputs/data')
WORD2VEC_MODEL_DIR = Path('/home/tommy/Project/PcodeBERT/experiment/model/Unimap/MAIE')

# CPU to architecture mapping
CPU_ARCH_MAP = {
    'ARM': 'arm_32',
    'x86_64': 'x86_64'
}


def load_skipgram_model(arch: str) -> Word2Vec:
    """Load skipgram Word2Vec model for specified architecture"""
    model_path = WORD2VEC_MODEL_DIR / f'skipgram_{arch}.model'
    return Word2Vec.load(str(model_path))


def extract_instruction_sequences(json_data: Dict, arch: str) -> List[List[str]]:
    """Extract and normalize instruction sequences from JSON"""
    sequences = []
    for func_data in json_data.values():
        if 'basic_blocks' not in func_data:
            continue
        for bb_data in func_data['basic_blocks'].values():
            if 'instructions' not in bb_data:
                continue
            seq = []
            for inst in bb_data['instructions']:
                normalized = normalize_instruction(inst, arch=arch)
                if normalized:
                    seq.extend(normalized)
            if seq:
                sequences.append(seq)
    return sequences


def sequence_to_vector(sequence: List[str], model: Word2Vec) -> Optional[np.ndarray]:
    """Convert instruction sequence to averaged vector"""
    vectors = []
    for token in sequence:
        if token in model.wv:
            vectors.append(model.wv[token])
    
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


def process_json_file(json_path: Path, arch: str, model: Word2Vec) -> List[np.ndarray]:
    """Process single JSON file and return vectors"""
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    sequences = extract_instruction_sequences(json_data, arch)
    vectors = []
    for seq in sequences:
        vec = sequence_to_vector(seq, model)
        if vec is not None:
            vectors.append(vec)
    
    return vectors


def process_architecture(cpu: str, arch: str, file_names: List[str]) -> None:
    """Process all files for specific architecture"""
    print(f"\n=== Processing {cpu} ({arch}) ===")
    model = load_skipgram_model(arch)
    print(f"Loaded model: skipgram_{arch}.model")
    print(f"Vector dimension: {model.vector_size}")
    
    # Create output directory for this architecture
    arch_output_dir = OUTPUT_DATA_DIR / arch
    arch_output_dir.mkdir(parents=True, exist_ok=True)
    
    example_shown = False
    processed_count = 0
    total_vectors = 0
    
    for file_name in tqdm(file_names, desc=f"Processing {cpu}"):
        json_path = TARGET_DATA_DIR / file_name / f"{file_name}.json"
        
        if not json_path.exists():
            continue
        
        # Load JSON and extract sequences
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        sequences = extract_instruction_sequences(json_data, arch)
        
        # Show example instruction sequence once
        if not example_shown and sequences:
            example_shown = True
            print(f"\nExample instruction sequence from {file_name}:")
            print(f"  {sequences[0][:10]}...")  # Show first 10 instructions
        
        # Convert sequences to vectors
        vectors = []
        for seq in sequences:
            vec = sequence_to_vector(seq, model)
            if vec is not None:
                vectors.append(vec)
        
        if vectors:
            # Save each file separately
            output_path = arch_output_dir / f'{file_name}.pkl'
            with open(output_path, 'wb') as f:
                pickle.dump(vectors, f)
            
            processed_count += 1
            total_vectors += len(vectors)
    
    print(f"\nTotal processed: {processed_count} files, {total_vectors} vectors")
    print(f"Saved to: {arch_output_dir}/")


def main():
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each CPU architecture
    for cpu, arch in CPU_ARCH_MAP.items():
        print(f"Reading file names for {cpu}...")
        file_names = read_filenames_from_csv(CSV_PATH, cpu_filter=cpu)
        
        if not file_names:
            print(f"No files found for {cpu}, skipping...")
            continue
        
        process_architecture(cpu, arch, file_names)
        print()


if __name__ == '__main__':
    main()
