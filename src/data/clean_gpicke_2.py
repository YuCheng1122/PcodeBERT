import json
import pickle
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union, Optional
from functools import partial
import multiprocessing as mp

import networkx as nx
import pandas as pd
from tqdm import tqdm

_operand_pattern = re.compile(r"\(([^ ,]+)\s*,\s*[^,]*,\s*([0-9]+)\)")

def _map_operand_BERT(op_type: str) -> str:
    op_type_l = op_type.lower()
    if op_type_l == 'register':
        return "REG"
    if op_type_l == 'ram':
        return "MEM"
    if op_type_l in {'const', 'constant'}:
        return "CONST"
    if op_type_l == 'unique':
        return "UNIQUE"
    if op_type_l == 'stack':
        return "STACK"
    else:
        return op_type.upper()


def create_instruction_sentence_BERT(instruction: Dict) -> Optional[List[str]]:
    opcode = instruction.get("opcode")
    if not opcode:
        return None
        
    tokens = [opcode]
    operation_str = instruction.get("operation", "")
    
    for match in _operand_pattern.finditer(operation_str): 
        raw_operand_type = match.group(1)
        standardized_token = _map_operand_BERT(raw_operand_type)
        tokens.append(standardized_token)
    
    return tokens


def clean_data(json_data, G_raw: nx.DiGraph) -> nx.DiGraph:
    G = nx.DiGraph()
    for node in G_raw.nodes():
        addr = str(node)
        func = json_data.get(addr)
        if not func:
            continue
        instructions = func.get("instructions", [])
        all_tokens_for_node = []
        
        for instr in instructions:
            if isinstance(instr, dict):
                sentence_tokens = create_instruction_sentence_BERT(instr)
                if sentence_tokens:
                    all_tokens_for_node.extend(sentence_tokens)

        if all_tokens_for_node:
            sentence = ' '.join(all_tokens_for_node)
            G.add_node(addr, sentence=sentence)
            
    for src, dst in G_raw.edges():
        src, dst = str(src), str(dst)
        if G.has_node(src) and G.has_node(dst):
            G.add_edge(src, dst)
    return G

def process_single_file(file_info: Tuple[Path, Path, str], output_base_path: Path) -> Union[str, Tuple[str, str]]:
    """
    處理單一檔案。
    成功時回傳 file_name (str)，失敗時回傳 (file_name, error_message) (tuple)。
    """
    json_path, dot_path, file_name = file_info
    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        G_raw = nx.drawing.nx_pydot.read_dot(dot_path)
        G = clean_data(json_data, G_raw)

        prefix = file_name[:2]
        output_dir = output_base_path / prefix
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{file_name}.gpickle"
        with open(out_path, 'wb') as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
        return file_name  # 成功，回傳 file_name
    except Exception as e:
        return (file_name, str(e))  # 失敗，回傳 file_name 和錯誤


def process_all(csv_file_path: Path, root_dir: Path, out_dir: Path, num_processes=None):
    df = pd.read_csv(csv_file_path)
    file_info_list = []
    file_metadata = {}  # 暫存存在的檔案的 CPU 和 label

    print("Collecting file paths...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Checking files"):
        file_name = row['file_name']
        json_path = root_dir / file_name / f"{file_name}.json"
        dot_path = root_dir / file_name / f"{file_name}.dot"
        
        if json_path.exists() and dot_path.exists():
            file_info_list.append((json_path, dot_path, file_name))
            # 儲存額外資訊以供稍後寫入 CSV
            file_metadata[file_name] = {'CPU': row['CPU'], 'label': row['label']}
        else:
            pass
            
    print(f"Found {len(file_info_list)} existing files to process.")

    if num_processes is None:
        num_processes = mp.cpu_count()

    process_func = partial(process_single_file, output_base_path=out_dir)

    print(f"Processing {len(file_info_list)} files using {num_processes} processes...")
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_func, file_info_list), total=len(file_info_list), desc="Processing"))

    # 總結並準備新的 CSV
    processed_data_list = []
    fail_logs = []

    for res in results:
        if isinstance(res, str):  # 成功 (回傳 file_name)
            file_name = res
            metadata = file_metadata[file_name]
            processed_data_list.append({
                'file_name': file_name,
                'CPU': metadata['CPU'],
                'label': metadata['label']
            })
        else:  # 失敗 (回傳 (file_name, error_msg))
            file_name, error_msg = res
            fail_logs.append(f"Error {file_name}: {error_msg}")

    # 儲存成功處理的檔案列表到新的 CSV
    if processed_data_list:
        processed_df = pd.DataFrame(processed_data_list)
        # 確保欄位順序
        processed_df = processed_df[['file_name', 'CPU', 'label']]
        
        output_csv_path = csv_file_path.with_name(f"{csv_file_path.stem}_processed.csv")
        processed_df.to_csv(output_csv_path, index=False)
        print(f"\nSaved list of {len(processed_df)} processed files to: {output_csv_path}")
    else:
        print("\nNo files were processed successfully.")

    # 輸出總結
    print(f"\nSuccess: {len(processed_data_list)}, Failed: {len(fail_logs)}")
    if fail_logs:
        print("\nFailed files (top 10):")
        for f in fail_logs[:10]:
            print(f"  {f}")


if __name__ == "__main__":
    CSV_FILE = Path("/home/tommy/Projects/PcodeBERT/dataset/csv/merged_adjusted_filtered.csv")
    DATA_DIR = Path("/home/tommy/Projects/PcodeBERT/reverse/new/results")
    OUTPUT_DIR = Path("/home/tommy/Projects/PcodeBERT/outputs/preprocessed/gpickle_merged_adjusted_filtered")

    process_all(CSV_FILE, DATA_DIR, OUTPUT_DIR)