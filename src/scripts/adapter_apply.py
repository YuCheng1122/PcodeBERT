import os
import sys
import pickle
import torch
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import RobertaForMaskedLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.adapter_temp import AdapterMapper
from configs.adapter import get_adapter_config


def get_files_by_cpu(csv_path, target_cpus):
    df = pd.read_csv(csv_path)
    if target_cpus:
        df_filtered = df[df['CPU'].isin(target_cpus)]
    else:
        df_filtered = df
    return df_filtered['file_name'].tolist()


def get_embeddings_batch(sentences, model, tokenizer, device, batch_size=1000):
    all_embeddings = []
    
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i : i + batch_size]
        
        inputs = tokenizer(
            batch_sentences, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.roberta(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(embeddings)
    
    if not all_embeddings:
        return None
        
    return torch.cat(all_embeddings, dim=0)


def process_single_graph(graph_path, roberta_model, adapter, tokenizer, device, base_path, output_base_path):
    try:
        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)
        
        node_ids_with_sentence = []
        sentences_to_process = []
        for node_id, node_data in graph.nodes(data=True):
            sentence = node_data.get('sentence', '')
            if sentence:
                node_ids_with_sentence.append(node_id)
                sentences_to_process.append(sentence)

        if not sentences_to_process:
            return False, "No sentences"
        
        roberta_embeddings = get_embeddings_batch(sentences_to_process, roberta_model, tokenizer, device)
        
        if roberta_embeddings is None:
            return False, "Embedding failed"
        
        with torch.no_grad():
            adapted_embeddings = adapter(roberta_embeddings).cpu().numpy()
        
        node_embeddings = {}
        node_sentences = {}
        for i, node_id in enumerate(node_ids_with_sentence):
            node_embeddings[node_id] = adapted_embeddings[i]
            node_sentences[node_id] = sentences_to_process[i]
        
        result = {
            'file_path': graph_path,
            'node_embeddings': node_embeddings,
            'node_sentences': node_sentences,
            'num_nodes': len(node_embeddings),
            'embedding_dim': adapted_embeddings.shape[1]
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
    config = get_adapter_config()
    
    input_path = config["inference_input_path"]
    output_path = config["inference_output_path"]
    adapter_path = config["save_path"]
    model_path = config["model_name"]
    csv_path = config["csv_path"]
    target_cpus = config["target_cpus"] if config["target_cpus"] else None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    file_names = get_files_by_cpu(csv_path, target_cpus)
    print(f"Target CPUs: {target_cpus if target_cpus else 'All'}")
    print(f"Found {len(file_names)} files in CSV")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    roberta_model = RobertaForMaskedLM.from_pretrained(model_path)
    roberta_model.to(device)
    roberta_model.eval()
    for param in roberta_model.parameters():
        param.requires_grad = False
    print("RoBERTa loaded")
    
    adapter = AdapterMapper(
        input_dim=config["input_dim"],
        output_dim=config["output_dim"],
        hidden_dim=config["hidden_dim"]
    ).to(device)
    adapter.load_state_dict(torch.load(adapter_path, map_location=device))
    adapter.eval()
    print("Adapter loaded")
    
    processed_count = 0
    failed_count = 0
    skipped_count = 0
    total_nodes = 0
    
    for file_name in tqdm(file_names, desc="Processing"):
        prefix = file_name[:2]
        graph_path = Path(input_path) / prefix / f"{file_name}.gpickle"
        
        if not graph_path.exists():
            skipped_count += 1
            continue
        
        success, result = process_single_graph(
            str(graph_path), roberta_model, adapter, tokenizer, device, input_path, output_path
        )
        
        if success:
            processed_count += 1
            total_nodes += result
        else:
            failed_count += 1
    
    stats = {
        'target_cpus': target_cpus if target_cpus else 'All',
        'total_files_in_csv': len(file_names),
        'processed_files': processed_count,
        'failed_files': failed_count,
        'skipped_files': skipped_count,
        'total_nodes': total_nodes
    }
    
    stats_path = os.path.join(output_path, "processing_stats.json")
    os.makedirs(output_path, exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Processed: {processed_count}/{len(file_names)}")
    print(f"Skipped: {skipped_count}")
    print(f"Total nodes: {total_nodes}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
