import os
import sys
import pickle
import torch
import numpy as np
from transformers import RobertaForMaskedLM, AutoTokenizer
import glob
from tqdm import tqdm
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_pretrained_model():
    """載入預訓練的模型和tokenizer"""
    model_path = "/home/tommy/Project/PcodeBERT/outputs/models/pretrain"
    
    print(f"Loading model from: {model_path}")
    
    # 載入tokenizer和model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = RobertaForMaskedLM.from_pretrained(model_path)
    
    # 設定device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on device: {device}")
    return model, tokenizer, device

def get_sentence_embedding(sentence, model, tokenizer, device):
    """對單個sentence生成embedding"""
    # Tokenize
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 生成embedding
    with torch.no_grad():
        outputs = model.roberta(**inputs)
        # 使用[CLS] token的embedding
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    return embedding[0]  # 返回一維array

def process_single_graph(graph_path, model, tokenizer, device):
    """處理單個graph檔案"""
    try:
        # 載入graph資料
        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)
        
        # 處理每個節點
        node_embeddings = {}
        node_sentences = {}
        
        for node_id, node_data in graph.nodes(data=True):
            sentence = node_data.get('sentence', '')
            if sentence:
                # 生成embedding
                embedding = get_sentence_embedding(sentence, model, tokenizer, device)
                node_embeddings[node_id] = embedding
                node_sentences[node_id] = sentence
        
        return {
            'file_path': graph_path,
            'node_embeddings': node_embeddings,
            'node_sentences': node_sentences,
            'num_nodes': len(node_embeddings),
            'embedding_dim': len(list(node_embeddings.values())[0]) if node_embeddings else 0
        }
        
    except Exception as e:
        print(f"Error processing {graph_path}: {e}")
        return None

def find_all_gpickle_files(base_path):
    """找到所有gpickle檔案"""
    gpickle_files = []
    
    # 遍歷所有子目錄
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.gpickle'):
                gpickle_files.append(os.path.join(root, file))
    
    return gpickle_files

def batch_process_graphs(base_path, output_base_dir):
    """批量處理所有graph檔案，保持相同目錄結構"""
    # 載入模型
    model, tokenizer, device = load_pretrained_model()
    
    # 找到所有gpickle檔案
    gpickle_files = find_all_gpickle_files(base_path)
    print(f"Found {len(gpickle_files)} gpickle files")
    
    # 統計資訊
    processed_count = 0
    failed_count = 0
    total_nodes = 0
    embedding_dim = 0
    
    for i, file_path in enumerate(tqdm(gpickle_files, desc="Processing graphs")):
        print(f"\nProcessing {i+1}/{len(gpickle_files)}: {os.path.basename(file_path)}")
        
        result = process_single_graph(file_path, model, tokenizer, device)
        
        if result:
            # 建立對應的輸出路徑結構
            # 從原始路徑取得相對路徑
            rel_path = os.path.relpath(file_path, base_path)
            output_path = os.path.join(output_base_dir, rel_path)
            
            # 創建輸出目錄
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 儲存單個檔案的embedding結果
            with open(output_path, 'wb') as f:
                pickle.dump(result, f)
            
            processed_count += 1
            total_nodes += result['num_nodes']
            if embedding_dim == 0:
                embedding_dim = result['embedding_dim']
            
            # print(f"  - Nodes: {result['num_nodes']}")
            # print(f"  - Embedding dim: {result['embedding_dim']}")
            # print(f"  - Saved to: {output_path}")
        else:
            failed_count += 1
    
    # 儲存統計資訊
    stats = {
        'total_files': len(gpickle_files),
        'processed_files': processed_count,
        'failed_files': failed_count,
        'total_nodes': total_nodes,
        'embedding_dim': embedding_dim
    }
    
    stats_path = os.path.join(output_base_dir, "processing_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # print(f"\n=== Processing Complete ===")
    # print(f"Total files: {stats['total_files']}")
    # print(f"Successfully processed: {stats['processed_files']}")
    # print(f"Failed: {stats['failed_files']}")
    # print(f"Total nodes embedded: {stats['total_nodes']}")
    # print(f"Embedding dimension: {stats['embedding_dim']}")
    print(f"Results saved to: {output_base_dir}")
    print(f"Stats saved to: {stats_path}")

def main():
    # 設定路徑
    base_path = "/home/tommy/Project/PcodeBERT/outputs/gpickle"
    output_dir = "/home/tommy/Project/PcodeBERT/outputs/embeddings"
    
    print(f"Starting batch processing...")
    print(f"Input directory: {base_path}")
    print(f"Output directory: {output_dir}")
    
    # 批量處理
    batch_process_graphs(base_path, output_dir)

if __name__ == "__main__":
    main()
