#!/usr/bin/env python3
"""Simple statistics for tokens before and after normalization"""

import pickle
from pathlib import Path
from collections import Counter
from preprocessing import load_json_file, extract_cpu_architecture

def analyze_tokens():
    """Analyze token statistics"""
    raw_data_path = Path('/home/tommy/Project/PcodeBERT/outputs/preprocessed/results_BinKit_instruction/results')
    output_dir = Path('/home/tommy/Project/PcodeBERT/experiment/outputs')
    
    for arch in ['x86_64', 'arm_32']:
        print(f"\n{'='*60}")
        print(f"{arch.upper()} Statistics")
        print(f"{'='*60}")
        
        # Count original instructions (before normalization)
        original_instructions = []
        for subdir in raw_data_path.iterdir():
            if not subdir.is_dir():
                continue
            for json_file in subdir.glob('*.json'):
                if extract_cpu_architecture(json_file.stem) == arch:
                    json_data = load_json_file(json_file)
                    if json_data:
                        for func_data in json_data.values():
                            if 'basic_blocks' in func_data:
                                for bb_data in func_data['basic_blocks'].values():
                                    if 'instructions' in bb_data:
                                        original_instructions.extend(bb_data['instructions'])
        
        # Load normalized corpus
        corpus_file = output_dir / f'corpus_{arch}.pkl'
        if corpus_file.exists():
            with open(corpus_file, 'rb') as f:
                corpus = pickle.load(f)
            
            # Count normalized tokens
            normalized_tokens = []
            for sentence in corpus:
                normalized_tokens.extend(sentence)
            
            # Calculate unique counts
            unique_original = len(set(original_instructions))
            unique_normalized = len(set(normalized_tokens))
            
            print(f"Sentences (basic blocks): {len(corpus):,}")
            print(f"\nOriginal instructions:")
            print(f"  Total: {len(original_instructions):,}")
            print(f"  Unique: {unique_original:,}")
            
            print(f"\nNormalized tokens:")
            print(f"  Total: {len(normalized_tokens):,}")
            print(f"  Unique: {unique_normalized:,}")
            
            print(f"\nReduction ratio: {unique_original:,} -> {unique_normalized:,} ({unique_normalized/unique_original*100:.1f}%)")

if __name__ == "__main__":
    analyze_tokens()
