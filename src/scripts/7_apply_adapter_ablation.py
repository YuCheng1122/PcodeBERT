import os
import sys
import pickle
import torch
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.adapter_models import AdapterEmbeddingModel
from adapters import AdapterConfig


def get_files_by_cpu(csv_path, target_cpus):
    df = pd.read_csv(csv_path)
    return (df[df['CPU'].isin(target_cpus)] if target_cpus else df)['file_name'].tolist()


def get_embeddings_batch(sentences, model, device, batch_size=256):
    all_embeddings = []
    
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i+batch_size]
        inputs = model.tokenizer(batch_sentences, return_tensors="pt", truncation=True, 
                                padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            embeddings = model(inputs['input_ids'], inputs['attention_mask'])
            all_embeddings.append(embeddings.cpu())
    
    return torch.cat(all_embeddings, dim=0) if all_embeddings else None


def process_single_graph(graph_path, model, device, base_path, output_base_path):
    try:
        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)
        
        node_ids, sentences = [], []
        for node_id, node_data in graph.nodes(data=True):
            sentence = node_data.get('sentence', '')
            if sentence:
                node_ids.append(node_id)
                sentences.append(sentence)

        if not sentences:
            return False, "No sentences in graph"
        
        embeddings = get_embeddings_batch(sentences, model, device)
        if embeddings is None:
            return False, "Embedding generation failed"
        
        embeddings_np = embeddings.numpy()
        node_embeddings = {node_id: embeddings_np[i] for i, node_id in enumerate(node_ids)}
        node_sentences = {node_id: sentences[i] for i, node_id in enumerate(node_ids)}
        
        result = {
            'file_path': graph_path,
            'node_embeddings': node_embeddings,
            'node_sentences': node_sentences,
            'num_nodes': len(node_embeddings),
            'embedding_dim': embeddings_np.shape[1]
        }
        
        rel_path = os.path.relpath(graph_path, base_path)
        output_path = os.path.join(output_base_path, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(result, f)
        
        return True, len(node_embeddings)
        
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description='Apply Adapter for Embedding Generation')
    parser.add_argument('--epoch', type=int, required=True, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       help='Adapter checkpoint epoch (1-10)')
    parser.add_argument('--loss', type=str, required=True, choices=['mse', 'cosine'],
                       help='Loss function type')
    args = parser.parse_args()
    
    # Paths configuration
    model_path = "/home/tommy/Project/PcodeBERT/outputs/models/RoBERTa/model_epoch_25"
    config_name = f"{args.loss}_6layers_epoch{args.epoch}"
    adapter_path = f"/home/tommy/Project/PcodeBERT/outputs/models/Adapters/ablation/{config_name}"
    input_path = "/home/tommy/Project/PcodeBERT/outputs/data/GNN/gpickle_merged_adjusted_filtered"
    output_path = f"/home/tommy/Project/PcodeBERT/outputs/data/GNN/gpickle_{args.loss}_epoch{args.epoch}"
    csv_path = "/home/tommy/Project/PcodeBERT/dataset/csv/merged_adjusted_filtered.csv"
    target_cpus = ["x86_64", "ARM"]
    
    print(f"\n{'='*60}")
    print(f"Applying Adapter for Embedding Generation")
    print(f"{'='*60}")
    print(f"Configuration: {config_name}")
    print(f"Pretrain Model: {model_path}")
    print(f"Adapter Path: {adapter_path}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Target CPUs: {target_cpus}")
    print(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Create adapter config
    adapter_config = AdapterConfig.load(
        "pfeiffer",
        reduction_factor=64,
        non_linearity="gelu"
    )
    
    # Load model with adapter
    model = AdapterEmbeddingModel(
        model_name=model_path,
        adapter_config=adapter_config,
        adapter_name="pcode_adapter"
    ).to(device)
    
    model.load_adapter(adapter_path)
    model.eval()
    
    print(f"Active adapters: {model.model.active_adapters}")
    print(f"Available adapters: {list(model.model.adapters_config.adapters.keys())}\n")
    
    # Get file list
    file_names = get_files_by_cpu(csv_path, target_cpus)
    print(f"Processing {len(file_names)} files\n")
    
    # Process files
    processed, failed, skipped, total_nodes = 0, 0, 0, 0
    failed_files = []
    
    for file_name in tqdm(file_names, desc="Processing"):
        prefix = file_name[:2]
        graph_path = Path(input_path) / prefix / f"{file_name}.gpickle"
        
        if not graph_path.exists():
            skipped += 1
            continue
        
        success, result = process_single_graph(
            str(graph_path), model, device, input_path, output_path
        )
        
        if success:
            processed += 1
            total_nodes += result
        else:
            failed += 1
            failed_files.append({'file': file_name, 'reason': result})
    
    # Save statistics
    stats = {
        'adapter_name': config_name,
        'pretrain_model': 'model_epoch_25',
        'adapter_epochs': args.epoch,
        'loss_function': args.loss,
        'target_cpus': target_cpus,
        'total_files_in_csv': len(file_names),
        'processed_files': processed,
        'failed_files': failed,
        'skipped_files': skipped,
        'total_nodes': total_nodes,
        'failed_details': failed_files
    }
    
    os.makedirs(output_path, exist_ok=True)
    stats_path = os.path.join(output_path, "processing_stats.json")
    
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Embedding Generation Summary")
    print(f"{'='*60}")
    print(f"Processed: {processed}/{len(file_names)}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Total nodes: {total_nodes}")
    print(f"Stats saved to: {stats_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
