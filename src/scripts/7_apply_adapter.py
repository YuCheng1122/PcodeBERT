import os
import sys
import pickle
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.adapter import create_adapter


# 配置
data_path = "/home/tommy/Project/PcodeBERT/outputs/embeddings"
output_path = "/home/tommy/Project/PcodeBERT/outputs/embeddings_adapted"
adapter_path = "/home/tommy/Project/PcodeBERT/outputs/adapters/adapter.pth"
csv_path = "/home/tommy/Project/PcodeBERT/dataset/csv/base_dataset_filtered_v2.csv"

# 指定要處理的 CPU 類型
target_cpus = ["AMD X86-64", "ARM-32"]

# 設定 device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 讀取 CSV 並篩選符合條件的資料
print(f"Reading CSV from: {csv_path}")
df = pd.read_csv(csv_path)
df_filtered = df[df['CPU'].isin(target_cpus)]
print(f"Target CPUs: {target_cpus}")
print(f"Found {len(df_filtered)} files matching target CPUs\n")

# 載入 adapter
adapter = create_adapter(adapter_type='lstm', input_dim=256, hidden_dim=256)
adapter.load_state_dict(torch.load(adapter_path, map_location=device))
adapter.to(device)
adapter.eval()
print(f"Adapter loaded from: {adapter_path}\n")

# 處理每個檔案
processed = 0
skipped = 0

for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Processing files"):
    file_name = row['file_name']
    prefix = file_name[:2]
    
    # 構建路徑
    graph_path = Path(data_path) / prefix / f"{file_name}.gpickle"
    
    if not graph_path.exists():
        skipped += 1
        continue
    
    # 載入 gpickle
    with open(graph_path, 'rb') as f:
        data = pickle.load(f)
    
    node_embeddings = data.get('node_embeddings', {})
    if not node_embeddings:
        skipped += 1
        continue
    
    # 提取 embeddings 並轉換
    embeddings = [list(emb) for emb in node_embeddings.values()]
    x = torch.tensor(embeddings, dtype=torch.float32).to(device)
    
    # 套用 adapter
    with torch.inference_mode():
        adapted_x = adapter(x).cpu().numpy()
    
    # 更新 node_embeddings
    new_node_embeddings = {}
    for i, node_id in enumerate(node_embeddings.keys()):
        new_node_embeddings[node_id] = adapted_x[i]
    
    data['node_embeddings'] = new_node_embeddings
    
    # 保存到輸出目錄
    output_dir = Path(output_path) / prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{file_name}.gpickle"
    
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    processed += 1

print(f"\nDone!")
print(f"Processed: {processed} files")
print(f"Skipped: {skipped} files")
print(f"Output saved to: {output_path}")
