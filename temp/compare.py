"""
簡單比較 x86 和 arm 架構的腳本
"""
import os
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from transformers import RobertaForMaskedLM, AutoTokenizer
from scipy.spatial.distance import cosine


# ===== 圖結構分析 =====
def analyze_graph(dot_file):
    """分析 dot 檔案的圖結構"""
    with open(dot_file, 'r') as f:
        content = f.read()
    
    # 解析 nodes 和 edges
    nodes = re.findall(r'"([^"]+)"\s*\[label="([^"]+)"\]', content)
    edges = re.findall(r'"([^"]+)"\s*->\s*"([^"]+)"', content)
    
    # 計算 degree
    in_degree = {}
    out_degree = {}
    for src, dst in edges:
        out_degree[src] = out_degree.get(src, 0) + 1
        in_degree[dst] = in_degree.get(dst, 0) + 1
    
    return {
        'nodes': len(nodes),
        'edges': len(edges),
        'avg_in_degree': np.mean(list(in_degree.values())) if in_degree else 0,
        'avg_out_degree': np.mean(list(out_degree.values())) if out_degree else 0,
    }


# ===== 指令分析 =====
def load_model():
    """載入預訓練模型"""
    model_path = "/home/tommy/Project/PcodeBERT/outputs/models/pretrain"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = RobertaForMaskedLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def get_embedding(text, model, tokenizer, device):
    """取得文本 embedding"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.roberta(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
    
    return embedding


def analyze_instructions(json_file):
    """分析 JSON 中的指令"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    results = {}
    for func_addr, func_data in data.items():
        if 'instructions' not in func_data:
            continue
        
        instructions = func_data['instructions']
        opcodes = [inst.get('opcode', '') for inst in instructions]
        operations = [inst.get('operation', '') for inst in instructions]
        
        results[func_addr] = {
            'num_instructions': len(instructions),
            'opcodes': opcodes,
            'operations': operations,
            'unique_opcodes': list(set(opcodes))
        }
    
    return results


def get_function_embeddings(json_file, model, tokenizer, device):
    """取得每個 function 的 embedding"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    embeddings = {}
    for func_addr, func_data in data.items():
        if 'instructions' not in func_data:
            continue
        
        # 取前 5 個 operation 串接
        operations = [inst.get('operation', '') for inst in func_data['instructions'][:5]]
        text = ' '.join(operations)
        
        if text.strip():
            embeddings[func_addr] = get_embedding(text, model, tokenizer, device)
    
    return embeddings


# ===== 主要比較函數 =====
def compare_pair(x86_dir, arm_dir, model, tokenizer, device):
    """比較一對 x86 和 arm binary"""
    
    x86_json = x86_dir / f"{x86_dir.name}.json"
    x86_dot = x86_dir / f"{x86_dir.name}.dot"
    arm_json = arm_dir / f"{arm_dir.name}.json"
    arm_dot = arm_dir / f"{arm_dir.name}.dot"
    
    # 檢查檔案
    if not all([x86_json.exists(), x86_dot.exists(), arm_json.exists(), arm_dot.exists()]):
        return None
    
    print(f"Comparing: {x86_dir.name.split('_')[-1]}")
    
    # 1. 圖結構
    x86_graph = analyze_graph(x86_dot)
    arm_graph = analyze_graph(arm_dot)
    
    # 2. 指令分析
    x86_inst = analyze_instructions(x86_json)
    arm_inst = analyze_instructions(arm_json)
    
    # 找共同的 functions
    common_funcs = set(x86_inst.keys()) & set(arm_inst.keys())
    
    # 比較 opcodes
    x86_all_opcodes = []
    arm_all_opcodes = []
    for func in common_funcs:
        x86_all_opcodes.extend(x86_inst[func]['opcodes'])
        arm_all_opcodes.extend(arm_inst[func]['opcodes'])
    
    # 3. Embedding
    print("  Computing embeddings...")
    x86_emb = get_function_embeddings(x86_json, model, tokenizer, device)
    arm_emb = get_function_embeddings(arm_json, model, tokenizer, device)
    
    # 計算 embedding 距離
    common_emb_funcs = set(x86_emb.keys()) & set(arm_emb.keys())
    cosine_distances = []
    for func in common_emb_funcs:
        dist = cosine(x86_emb[func], arm_emb[func])
        cosine_distances.append(dist)
    
    return {
        'binary': x86_dir.name.split('_')[-1],
        
        # 圖結構
        'x86_nodes': x86_graph['nodes'],
        'arm_nodes': arm_graph['nodes'],
        'x86_edges': x86_graph['edges'],
        'arm_edges': arm_graph['edges'],
        'x86_avg_in_degree': x86_graph['avg_in_degree'],
        'arm_avg_in_degree': arm_graph['avg_in_degree'],
        'x86_avg_out_degree': x86_graph['avg_out_degree'],
        'arm_avg_out_degree': arm_graph['avg_out_degree'],
        
        # 指令
        'x86_functions': len(x86_inst),
        'arm_functions': len(arm_inst),
        'common_functions': len(common_funcs),
        'x86_unique_opcodes': len(set(x86_all_opcodes)),
        'arm_unique_opcodes': len(set(arm_all_opcodes)),
        
        # Embedding
        'avg_cosine_distance': np.mean(cosine_distances) if cosine_distances else 0,
        'std_cosine_distance': np.std(cosine_distances) if cosine_distances else 0,
    }


def find_pairs(data_dir):
    """找 x86 和 arm 的配對"""
    data_dir = Path(data_dir)
    x86_dirs = [d for d in data_dir.iterdir() if d.is_dir() and 'x86_64' in d.name]
    arm_dirs = [d for d in data_dir.iterdir() if d.is_dir() and 'arm_64' in d.name]
    
    pairs = []
    for x86_dir in x86_dirs:
        binary_name = x86_dir.name.split('_')[-1]
        for arm_dir in arm_dirs:
            if arm_dir.name.split('_')[-1] == binary_name:
                pairs.append((x86_dir, arm_dir))
                break
    
    return pairs


def main():
    # 設定
    data_dir = "/home/tommy/Project/PcodeBERT/outputs/align_sentences"
    output_file = "/home/tommy/Project/PcodeBERT/temp/comparison_results.csv"
    
    # 載入模型
    print("Loading model...")
    model, tokenizer, device = load_model()
    
    # 找配對
    print("\nFinding pairs...")
    pairs = find_pairs(data_dir)
    print(f"Found {len(pairs)} pairs")
    
    # 比較
    results = []
    for i, (x86_dir, arm_dir) in enumerate(pairs[:10], 1):  # 先只比較前 10 個
        print(f"\n[{i}/{min(10, len(pairs))}]")
        result = compare_pair(x86_dir, arm_dir, model, tokenizer, device)
        if result:
            results.append(result)
    
    # 儲存
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\n結果已儲存至: {output_file}")
    print("\n統計摘要:")
    print(df.describe())


if __name__ == "__main__":
    main()
