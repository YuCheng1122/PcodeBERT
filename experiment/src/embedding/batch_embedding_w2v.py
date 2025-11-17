import os
import pickle
import numpy as np
from gensim.models import Word2Vec, FastText
from gensim.models import KeyedVectors
from tqdm import tqdm
import json

def load_word2vec_model(model_path, model_type):
    """Load word2vec style model"""
    if model_type == 'fasttext':
        model = FastText.load(os.path.join(model_path, 'fasttext_model.model'))
        return model.wv
    elif model_type == 'cbow':
        model = Word2Vec.load(os.path.join(model_path, 'cbow_model.model'))
        return model.wv
    elif model_type == 'skipgram':
        model = Word2Vec.load(os.path.join(model_path, 'skipgram_model.model'))
        return model.wv
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def get_sentence_embedding(sentence, model):
    """Get sentence embedding by averaging word embeddings"""
    words = sentence.split()
    embeddings = []
    
    for word in words:
        if word in model:
            embeddings.append(model[word])
    
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(model.vector_size)

def process_single_graph(graph_path, model, model_name):
    """Process single graph file"""
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
    
    node_embeddings = {}
    node_sentences = {}
    
    for node_id, node_data in graph.nodes(data=True):
        sentence = node_data.get('sentence', '')
        if sentence:
            embedding = get_sentence_embedding(sentence, model)
            node_embeddings[node_id] = embedding
            node_sentences[node_id] = sentence
    
    return {
        'file_path': graph_path,
        'node_embeddings': node_embeddings,
        'node_sentences': node_sentences,
        'num_nodes': len(node_embeddings),
        'embedding_dim': model.vector_size,
        'model_name': model_name
    }

def find_all_gpickle_files(base_path):
    """Find all gpickle files"""
    gpickle_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.gpickle'):
                gpickle_files.append(os.path.join(root, file))
    return gpickle_files

def batch_process_graphs(base_path, output_base_dir, model_path, model_name):
    """Batch process all graph files"""
    print(f"Loading {model_name} model...")
    model = load_word2vec_model(model_path, model_name)
    
    gpickle_files = find_all_gpickle_files(base_path)
    print(f"Found {len(gpickle_files)} gpickle files")
    
    processed_count = 0
    failed_count = 0
    total_nodes = 0
    
    for i, file_path in enumerate(tqdm(gpickle_files, desc=f"Processing with {model_name}")):
        try:
            result = process_single_graph(file_path, model, model_name)
            
            rel_path = os.path.relpath(file_path, base_path)
            output_path = os.path.join(output_base_dir, rel_path)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'wb') as f:
                pickle.dump(result, f)
            
            processed_count += 1
            total_nodes += result['num_nodes']
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            failed_count += 1
    
    stats = {
        'model_name': model_name,
        'total_files': len(gpickle_files),
        'processed_files': processed_count,
        'failed_files': failed_count,
        'total_nodes': total_nodes,
        'embedding_dim': model.vector_size
    }
    
    stats_path = os.path.join(output_base_dir, f"processing_stats_{model_name}.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"{model_name} results saved to: {output_base_dir}")
    print(f"{model_name} stats saved to: {stats_path}")

def main():
    base_path = "/home/tommy/Project/PcodeBERT/outputs/data/GNN/gpickle_merged_adjusted_filtered"
    output_base = "/home/tommy/Project/PcodeBERT/outputs/data/GNN"
    models_base = "/home/tommy/Project/PcodeBERT/outputs/models"
    
    models = [
        ('fasttext', os.path.join(models_base, 'fasttext')),
        ('cbow', os.path.join(models_base, 'cbow')),
        ('skipgram', os.path.join(models_base, 'skipgram'))
    ]
    
    print("Starting batch processing with Word2Vec models...")
    print(f"Input directory: {base_path}")
    print(f"Output base directory: {output_base}")
    
    for model_name, model_path in models:
        print(f"\n=== Processing with {model_name} model ===")
        output_dir = os.path.join(output_base, f"gpickle_merged_adjusted_filtered_{model_name}")
        batch_process_graphs(base_path, output_dir, model_path, model_name)

if __name__ == "__main__":
    main()
