import os
import json
import hashlib
import pandas as pd
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

def load_filter_csv(csv_path):
    """
    載入 CSV 並過濾出 CPU 為 AMD X86-64 和 ARM-32 的資料
    
    Args:
        csv_path: CSV 檔案路徑
        
    Returns:
        dict: key 為 file_name，value 為 {'CPU': ..., 'label': ..., 'family': ...}
    """
    try:
        df = pd.read_csv(csv_path)
        
        # 過濾 CPU 欄位
        filtered_df = df[df['CPU'].isin(['AMD X86-64', 'ARM-32'])]
        
        # 轉換為 dict，以 file_name 為 key
        file_info = {}
        for _, row in filtered_df.iterrows():
            file_info[row['file_name']] = {
                'CPU': row['CPU'],
                'label': row['label'],
                'family': row['family']
            }
        
        print(f"CSV 載入完成：")
        print(f"  • 總共 {len(df)} 筆資料")
        print(f"  • 過濾後 {len(file_info)} 筆 (AMD X86-64 + ARM-32)")
        
        return file_info
    
    except Exception as e:
        print(f"載入 CSV 時發生錯誤: {e}")
        return {}

def extract_pcodes(json_path):
    """
    從 JSON 檔案中提取所有 opcode (pcode) 序列
    
    Args:
        json_path: JSON 檔案路徑
        
    Returns:
        list: opcode 字串的列表，如果失敗則返回 None
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        pcodes = []
        
        # 遍歷所有 function/address
        for func_addr, func_data in sorted(data.items()):
            if isinstance(func_data, dict) and 'instructions' in func_data:
                # 提取每個指令的 opcode
                for instr in func_data['instructions']:
                    if 'opcode' in instr:
                        pcodes.append(instr['opcode'])
        
        return pcodes
    
    except (FileNotFoundError, json.JSONDecodeError, IOError) as e:
        return None
    except Exception as e:
        print(f"處理 {json_path} 時發生未預期的錯誤: {e}")
        return None

def compute_pcode_hash(pcodes):
    """
    計算 pcode 序列的 hash 值
    
    Args:
        pcodes: opcode 字串的列表
        
    Returns:
        str: SHA256 hash 值
    """
    # 將所有 pcodes 連接成一個字串，用換行符分隔
    pcode_str = '\n'.join(pcodes)
    return hashlib.sha256(pcode_str.encode('utf-8')).hexdigest()

def process_folder(folder_path, folder_name, file_filter):
    """
    處理單一資料夾，提取 pcode 序列並計算 hash
    
    Args:
        folder_path: 資料夾完整路徑
        folder_name: 資料夾名稱
        file_filter: 從 CSV 讀取的檔案過濾字典
        
    Returns:
        tuple: (pcode_hash, folder_info) 或 None（如果處理失敗或不在過濾清單中）
    """
    # 檢查是否在過濾清單中
    if folder_name not in file_filter:
        return None
    
    json_file_path = os.path.join(folder_path, f"{folder_name}.json")
    
    if not os.path.exists(json_file_path):
        return None
    
    pcodes = extract_pcodes(json_file_path)
    
    if pcodes:
        pcode_hash = compute_pcode_hash(pcodes)
        return (pcode_hash, {
            'folder_name': folder_name,
            'pcode_count': len(pcodes),
            'CPU': file_filter[folder_name]['CPU'],
            'label': file_filter[folder_name]['label'],
            'family': file_filter[folder_name]['family']
        })
    
    return None

def find_duplicate_folders(directory, file_filter, max_workers=16):
    """
    在指定目錄中尋找具有相同 pcode 序列的子目錄（使用多執行緒）
    
    Args:
        directory: 要掃描的目標目錄路徑
        file_filter: 從 CSV 讀取的檔案過濾字典
        max_workers: 最大執行緒數量
        
    Returns:
        dict: key 為 hash，value 為具有相同 hash 的資料夾資訊列表
    """
    hash_to_folders = defaultdict(list)
    lock = Lock()
    
    print(f"\n階段二：正在使用 {max_workers} 個執行緒讀取並分析 JSON 檔案...")
    
    # 收集所有需要處理的資料夾（只處理在 CSV 中的檔案）
    folders_to_process = []
    for folder_name in sorted(os.listdir(directory)):
        if folder_name in file_filter:
            folder_path = os.path.join(directory, folder_name)
            if os.path.isdir(folder_path):
                folders_to_process.append((folder_path, folder_name))
    
    total_folders = len(folders_to_process)
    processed_count = 0
    success_count = 0
    
    print(f"  待處理資料夾數: {total_folders}")
    
    # 使用 ThreadPoolExecutor 平行處理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任務
        future_to_folder = {
            executor.submit(process_folder, folder_path, folder_name, file_filter): folder_name 
            for folder_path, folder_name in folders_to_process
        }
        
        # 處理完成的任務
        for future in as_completed(future_to_folder):
            result = future.result()
            processed_count += 1
            
            if result:
                pcode_hash, folder_info = result
                with lock:
                    hash_to_folders[pcode_hash].append(folder_info)
                success_count += 1
            
            # 每處理 100 個顯示進度
            if processed_count % 100 == 0:
                print(f"  進度: {processed_count}/{total_folders} 已處理")
    
    print(f"計算完成！共成功處理 {success_count}/{total_folders} 個資料夾。\n")
    
    return hash_to_folders

def main():
    # ================== 設定區 ==================
    target_dir = '/home/tommy/Projects/PcodeBERT/reverse/results'
    csv_path = '/home/tommy/Projects/PcodeBERT/dataset/csv/base_dataset_filtered.csv'
    output_file = 'duplicate_pcodes.txt'
    # 執行緒數量，可以根據你的 CPU 核心數調整
    MAX_WORKERS = 72
    # ==========================================

    if not os.path.isdir(target_dir):
        print(f"錯誤：目錄 '{target_dir}' 不存在。")
        return
    
    if not os.path.isfile(csv_path):
        print(f"錯誤：CSV 檔案 '{csv_path}' 不存在。")
        return

    print(f"開始處理...")
    print(f"掃描目錄: {target_dir}")
    print(f"CSV 檔案: {csv_path}")
    print(f"使用執行緒數: {MAX_WORKERS}\n")
    
    # 階段一：載入並過濾 CSV
    print("階段一：載入並過濾 CSV 資料...")
    file_filter = load_filter_csv(csv_path)
    
    if not file_filter:
        print("錯誤：沒有符合條件的資料（CPU = AMD X86-64 或 ARM-32）")
        return
    
    # 階段二：掃描資料夾並計算 pcode hash
    hash_to_folders = find_duplicate_folders(target_dir, file_filter, max_workers=MAX_WORKERS)
    
    # 找出有重複的群組
    duplicate_groups = {h: folders for h, folders in hash_to_folders.items() if len(folders) > 1}
    
    # 將結果寫入檔案
    print("階段三：生成報告...")
    with open(output_file, 'w', encoding='utf-8') as f:
        if not duplicate_groups:
            f.write("沒有找到任何具有相同 PCode 序列的檔案群組。\n")
            print(f"結果：沒有找到重複檔案。報告已寫入 {output_file}")
            return
        
        total_duplicates = sum(len(folders) for folders in duplicate_groups.values())
        total_groups = len(duplicate_groups)
        
        f.write(f"掃描完畢！共找到 {total_groups} 個重複群組，涉及 {total_duplicates} 個檔案。\n")
        f.write("="*80 + "\n\n")
        
        for i, (pcode_hash, folders) in enumerate(sorted(duplicate_groups.items(), 
                                                          key=lambda x: len(x[1]), 
                                                          reverse=True)):
            f.write(f"--- 重複群組 {i+1} (共 {len(folders)} 個成員) ---\n")
            f.write(f"PCode Hash: {pcode_hash[:16]}...\n")
            f.write(f"PCode 數量: {folders[0]['pcode_count']}\n\n")
            
            f.write("成員列表：\n")
            for folder_info in sorted(folders, key=lambda x: x['folder_name']):
                f.write(f"  - {folder_info['folder_name']}\n")
                f.write(f"      CPU: {folder_info['CPU']}, Label: {folder_info['label']}, Family: {folder_info['family']}\n")
            
            f.write("\n" + "-"*80 + "\n\n")
        
        # 額外輸出統計資訊
        f.write("="*80 + "\n")
        f.write("統計摘要：\n")
        f.write(f"  • CSV 過濾後資料數: {len(file_filter)}\n")
        f.write(f"  • 實際掃描資料夾數: {len(hash_to_folders)}\n")
        f.write(f"  • 發現重複群組數: {total_groups}\n")
        f.write(f"  • 涉及重複檔案數: {total_duplicates}\n")
        f.write(f"  • 可刪除檔案數: {total_duplicates - total_groups} (每組保留一個)\n")
        
        # CPU 分佈統計
        cpu_stats = defaultdict(int)
        for folders in duplicate_groups.values():
            for folder in folders:
                cpu_stats[folder['CPU']] += 1
        
        f.write(f"\n重複檔案 CPU 分佈：\n")
        for cpu, count in sorted(cpu_stats.items()):
            f.write(f"  • {cpu}: {count} 個\n")
    
    print(f"\n✓ 成功！重複檔案清單已儲存至: {output_file}")
    print(f"  共找到 {total_groups} 個重複群組，涉及 {total_duplicates} 個檔案")

if __name__ == '__main__':
    main()