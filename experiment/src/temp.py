import pickle
import os
from pathlib import Path
import pandas as pd

def find_intel_gpickle_files(csv_path, graph_dir):
    df = pd.read_csv(csv_path)
    intel_df = df[df['CPU'] == 'ARM']
    
    gpickle_files = []
    for file_name in intel_df['file_name']:
        prefix = file_name[:2]
        gpickle_path = Path(graph_dir) / prefix / f"{file_name}.gpickle"
        if gpickle_path.exists():
            gpickle_files.append(str(gpickle_path))
    
    return gpickle_files

def extract_tokens_from_gpickle(gpickle_path):
    with open(gpickle_path, 'rb') as f:
        graph = pickle.load(f)
    
    tokens = set()
    
    for node_id, node_data in graph.nodes(data=True):
        sentence = node_data.get('sentence', '')
        if sentence:
            words = sentence.split()
            tokens.update(words)
    
    return tokens

def main():
    csv_path = "/home/tommy/Project/PcodeBERT/dataset/csv/merged_adjusted_filtered.csv"
    graph_dir = "/home/tommy/Project/PcodeBERT/outputs/data/GNN/gpickle_merged_adjusted_filtered"
    
    print("Finding Intel gpickle files...")
    intel_files = find_intel_gpickle_files(csv_path, graph_dir)
    print(f"Found {len(intel_files)} Intel files")
    
    all_tokens = set()
    
    for file_path in intel_files:
        tokens = extract_tokens_from_gpickle(file_path)
        all_tokens.update(tokens)
    
    print(f"\nTotal unique tokens: {len(all_tokens)}")
    print(f"\nUnique tokens:")
    for token in sorted(all_tokens):
        print(token)

if __name__ == "__main__":
    main()
