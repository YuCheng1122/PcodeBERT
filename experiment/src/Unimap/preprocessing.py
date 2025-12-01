import re
import json
import pickle
import logging
import os
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any, Optional
from collections import defaultdict

ARM_REG_MAP = {
    'SP': 'R13', 'LR': 'R14', 'PC': 'R15', 'SB': 'R9', 'SL': 'R10', 'FP': 'R11', 'IP': 'R12',
    'CPSR': 'CPSR', 'SPSR': 'SPSR' # 狀態暫存器也要保留，否則會變成 TAG
}

# x86 指標修飾詞映射
X86_PTR_MAP = {
    'BYTE PTR': '<BYTE_PTR>', 'WORD PTR': '<WORD_PTR>', 'DWORD PTR': '<DWORD_PTR>', 
    'QWORD PTR': '<QWORD_PTR>', 'XMMWORD PTR': '<XMMWORD_PTR>', 'YMMWORD PTR': '<YMMWORD_PTR>',
    'TBYTE PTR': '<TBYTE_PTR>'
}

# ARM Opcode 映射
ARM_OPCODE_MAP = { 'CPY': 'MOV' }

def expand_arm_reg_list(reg_string):
    """解析 Ghidra 的暫存器列表字串並回傳展開後的暫存器列表"""
    content = reg_string.strip('{}')
    if not content: return []
    
    # 這裡處理逗號分隔，無論有無空白
    tokens = re.split(r'[\s,]+', content)
    expanded = []
    
    for token in tokens:
        token = token.upper().strip()
        if not token: continue
        
        if '-' in token:
            try:
                start, end = token.split('-')
                start = ARM_REG_MAP.get(start, start)
                end = ARM_REG_MAP.get(end, end)
                prefix = start[0]
                s_idx = int(start[1:])
                e_idx = int(end[1:])
                for i in range(s_idx, e_idx + 1):
                    expanded.append(f"{prefix}{i}")
            except:
                expanded.append(token)
        else:
            token = ARM_REG_MAP.get(token, token)
            expanded.append(token)
    return expanded

def normalize_mem_content(match, arch):
    """處理 [...] 內部，確保格式統一且不被切斷"""
    inner = match.group(1)
    
    if arch == 'x86_64':
        for ptr, tag in X86_PTR_MAP.items():
            inner = re.sub(ptr, tag, inner, flags=re.I)

    # 分割內部元素
    parts = re.split(r'([+\-*,\s]+)', inner)
    norm_parts = []
    
    for p in parts:
        p = p.strip()
        if not p: continue
        
        if p in ['+', '-', '*']:
            norm_parts.append(p)
        elif p == ',': 
            # ARM [R0, #4] -> [R0+4]
            if arch == 'arm_32':
                norm_parts.append('+') 
            else:
                continue 
        else:
            norm_parts.append(normalize_single_token(p, arch, is_mem_operand=True))
            
    return '[' + ''.join(norm_parts) + ']'

def normalize_single_token(token, arch, is_mem_operand=False):
    clean_token = token.strip() # 確保無空白
    
    if arch == 'arm_32' and clean_token.startswith('#'):
        clean_token = clean_token[1:]
    
    clean_token = clean_token.upper()

    # 1. 數值正規化
    if re.match(r'^-?0X[0-9A-F]+$', clean_token) or re.match(r'^-?\d+$', clean_token):
        return '0' if not clean_token.startswith('-') else '-0'

    # 2. 變數/參數/字串
    if clean_token.startswith(('LOCAL_', 'STACK_', 'VAR_')) or re.match(r'^[A-Z]VAR\d+$', clean_token):
        return '<VAR>'
    if clean_token.startswith(('PARAM_', 'ARG_')):
        return '<ARG>'
    if clean_token.startswith(('FUN_', 'SUB_')):
        return '<FOO>'
    if 'STR' in clean_token or clean_token.startswith(('S_', 'A_', 'U_')):
        return '<STRING>'
    
    # 3. 架構特定暫存器
    if arch == 'arm_32':
        if clean_token in ARM_REG_MAP:
            return ARM_REG_MAP[clean_token]
        if re.match(r'^R\d+$', clean_token):
            return clean_token

    # 4. x86 暫存器
    if arch == 'x86_64':
        # 簡單檢查常見暫存器
        if clean_token in {'RAX','RBX','RCX','RDX','RSI','RDI','RBP','RSP','RIP',
                           'EAX','EBX','ECX','EDX','ESI','EDI','EBP','ESP',
                           'AX','BX','CX','DX','SI','DI','BP','SP',
                           'AL','BL','CL','DL','AH','BH','CH','DH'}:
            return clean_token
        if re.match(r'^R\d+[D|W|B]?$', clean_token):
            return clean_token
            
    if is_mem_operand and clean_token.startswith('<') and clean_token.endswith('>'):
        return clean_token

    # 5. Catch-all
    return '<TAG>'

def normalize_instruction(instruction, arch='arm_32'):
    if not instruction: return []
    instruction = re.split(r'[;#/]', instruction)[0].strip()
    if not instruction: return []

    # =================================================
    # 關鍵修正：強制將逗號分開，避免 "r0,r1" 被視為一個 token
    # =================================================
    instruction = instruction.replace(',', ' , ')

    # 優先處理記憶體運算元 [...]，處理完後內部無空白，不會被 split 切斷
    instruction = re.sub(r'\[(.*?)\]', lambda m: normalize_mem_content(m, arch), instruction)

    parts = instruction.split(None, 1)
    opcode = parts[0].upper()
    operands_str = parts[1] if len(parts) > 1 else ""

    if arch == 'arm_32':
        if opcode.endswith('.W'): opcode = opcode[:-2]
        if opcode in ARM_OPCODE_MAP: opcode = ARM_OPCODE_MAP[opcode]

    normalized_insts = []

    # ARM 暫存器列表展開
    if arch == 'arm_32' and '{' in operands_str:
        match = re.search(r'^(.*?)?\{(.+)\}', operands_str)
        if match:
            pre_ops = match.group(1).strip() if match.group(1) else ""
            if pre_ops:
                # pre_ops 可能包含 "sp! ,"，我們移除逗號並正規化
                pre_tokens = [normalize_single_token(t, arch) for t in pre_ops.split() if t != ',']
                pre_ops_str = "~".join(pre_tokens)
            else:
                pre_ops_str = ""
            
            reg_list = expand_arm_reg_list(match.group(2))
            if opcode.startswith(('PUSH', 'STM', 'STR')):
                reg_list.reverse()

            for reg in reg_list:
                if pre_ops_str:
                    normalized_insts.append(f"{opcode}~{pre_ops_str}~{reg}")
                else:
                    normalized_insts.append(f"{opcode}~{reg}")
            return normalized_insts

    # 一般指令處理
    full_str = f"{opcode} {operands_str}"
    tokens = full_str.split() # 現在因為逗號有空白，可以正確切分了
    
    norm_tokens = []
    for i, token in enumerate(tokens):
        # 忽略單獨的逗號 (因為我們上面加了空白，逗號會變成獨立 token)
        if token == ',': continue
        
        if i == 0:
            norm_tokens.append(token)
        else:
            # 如果是已經處理過的 [...]，直接加入
            if token.startswith('[') and token.endswith(']'):
                norm_tokens.append(token)
            else:
                norm_tokens.append(normalize_single_token(token, arch))

    return ["~".join(norm_tokens)]

def extract_cpu_architecture(filename: str) -> Optional[str]:
    """Extract architecture from filename (x86_64 or arm_32)"""
    if 'x86_64' in filename:
        return 'x86_64'
    elif 'arm_32' in filename:
        return 'arm_32'
    return None

def load_json_file(filepath: Path) -> Optional[Dict]:
    """Load JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def extract_corpus_from_json(json_data: Dict, normalize: bool = True, arch: str = None) -> List[List[str]]:
    """Extract corpus: basic blocks as sentences, instructions as tokens
    
    Args:
        json_data: JSON data containing functions and basic blocks
        normalize: Whether to normalize instructions
        arch: Architecture type ('arm_32' or 'x86_64')
    """
    corpus = []
    for func_data in json_data.values():
        if 'basic_blocks' not in func_data:
            continue
        for bb_data in func_data['basic_blocks'].values():
            if 'instructions' not in bb_data:
                continue
            sentence = []
            for inst in bb_data['instructions']:
                if normalize:
                    normalized_insts = normalize_instruction(inst, arch=arch)
                    # normalize_instruction returns a list, so we need to extend
                    if normalized_insts:
                        sentence.extend(normalized_insts)
                else:
                    sentence.append(inst)
            if sentence:
                corpus.append(sentence)
    return corpus

def save_corpus(corpus: List[List[str]], filepath: Path):
    """Save corpus to pickle file"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(corpus, f)
    print(f"Saved {len(corpus)} sentences to {filepath}")

def process_directory_by_architecture(root_dir: Path, output_dir: Path) -> Dict[str, List]:
    """Process JSON files and create corpus by architecture"""
    arch_corpus = {'x86_64': [], 'arm_32': []}
    stats = defaultdict(lambda: {'files': 0, 'blocks': 0})
    
    for subdir in root_dir.iterdir():
        if not subdir.is_dir():
            continue
        
        for json_file in subdir.glob('*.json'):
            arch = extract_cpu_architecture(json_file.stem)
            if arch not in ['x86_64', 'arm_32']:
                continue
            
            print(f"Processing {arch}: {json_file.name}")
            json_data = load_json_file(json_file)
            if json_data:
                file_corpus = extract_corpus_from_json(json_data, normalize=True, arch=arch)
                arch_corpus[arch].extend(file_corpus)
                stats[arch]['files'] += 1
                stats[arch]['blocks'] += len(file_corpus)
    
    # Save corpus for each architecture
    for arch, corpus in arch_corpus.items():
        if corpus:
            output_path = output_dir / f'corpus_{arch}.pkl'
            save_corpus(corpus, output_path)
            print(f"\n{arch}: {stats[arch]['files']} files, {stats[arch]['blocks']} blocks")
    
    return arch_corpus

def configure_logging(output_dir: str) -> logging.Logger:
    """
    Configure logging settings.

    Args:
        output_dir (str): Path to the output directory.

    Returns:
        logging.Logger: The extraction_logger object.
    """
    extraction_log_file = os.path.join(output_dir, 'extraction.log')
    print(f"Logging to: {extraction_log_file}")
    extraction_logger = logging.getLogger('extraction_logger')
    extraction_logger.setLevel(logging.INFO)
    # Clear existing handlers to avoid duplication
    extraction_logger.handlers.clear()
    extraction_handler = logging.FileHandler(extraction_log_file)
    extraction_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    extraction_logger.addHandler(extraction_handler)

    return extraction_logger

def read_filenames_from_csv(csv_file_path: str | Path, cpu_filter: Optional[str] = None) -> List[str]:
    try:
        df = pd.read_csv(csv_file_path)
        if cpu_filter:
            print(f"Filtering files for CPU: {cpu_filter}")
            df_filtered = df[df['CPU'] == cpu_filter]
            print(f"Found {len(df_filtered)} files matching the filter.")
            return df_filtered['file_name'].tolist()
    
        return df['file_name'].tolist()
        
    except (FileNotFoundError, KeyError) as e:
        print(f"Error reading CSV: {e}")
        return []
    
