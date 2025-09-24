import json
from os import name
import pickle
import re
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Generator, Optional
from datasets import Dataset
import pandas as pd
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup


#Regex pattern preprocessing
#1)  opcode_pattern: Extract P-Code
#2)  opcode_pattern: Extract Calculation
OPCODE_PAT = re.compile(r"(?:\)\s+|---\s+)([A-Z_]+)")
OPERAND_PAT = re.compile(r"\(([^ ,]+)\s*,\s*[^,]*,\s*([0-9]+)\)")

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


def iterate_json_files(csv_file_path: Path, root_dir: Path, error_log_path: Path, cpu_filter: Optional[str] = None) -> Generator[Tuple[str, Dict], None, None]:
    file_names = read_filenames_from_csv(csv_file_path, cpu_filter=cpu_filter)
    for file_name in file_names:
        json_path = root_dir / file_name / f"{file_name}.json"
        if not json_path.exists():
            with open(error_log_path, "a", encoding="utf-8") as f_err:
                f_err.write(f"{file_name}\n")
            continue  
        try:
            with json_path.open(encoding="utf-8") as fp:
                yield file_name, json.load(fp)
        except json.JSONDecodeError:
            with open(error_log_path, "a", encoding="utf-8") as f_err:
                f_err.write(f"{file_name}\n")
            continue 



def _map_operand(op_type: str) -> str:
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
    return "UNK"


def _append_to_pickle(file_path: Path, new_data):
    """將新資料追加到現有的 pickle 檔案中"""
    if file_path.exists():
        with open(file_path, "rb") as f:
            existing_data = pickle.load(f)
        existing_data.extend(new_data)
    else:
        existing_data = new_data
    
    with open(file_path, "wb") as f:
        pickle.dump(existing_data, f)
        

def create_instruction_sentence(instruction_dict: Dict) -> Optional[List[str]]:
    operation_str = instruction_dict.get("operation", "")
    if not operation_str:
        return None
    
    command_match = OPCODE_PAT.search(operation_str)
    if not command_match:
        return None

    command = command_match.group(1)
    sentence = [command]
    
    operands = OPERAND_PAT.findall(operation_str)
    for op_type, _ in operands:
        sentence.append(_map_operand(op_type))
    
    return sentence

def extract_sentences_from_file(file_name_data: Tuple[str, Dict]) -> List[List[str]]:
    file_name, pcode_dict = file_name_data
    sentences = []
    try:
        for func_data in pcode_dict.values():
            if not isinstance(func_data, dict): continue
            for instruction in func_data.get("instructions", []):
                sentence = create_instruction_sentence(instruction)
                if sentence:
                    sentences.append(sentence)
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
    return sentences

def load_corpus_dataset(corpus_path):
    """Load and prepare the training dataset with processed data caching"""
    corpus_path = Path(corpus_path)
    processed_path = corpus_path.parent / f"{corpus_path.stem}_processed.pkl"
    
    # Check if processed dataset already exists
    if processed_path.exists():
        print(f"Loading processed dataset from: {processed_path}")
        with open(processed_path, 'rb') as f:
            dataset = pickle.load(f)
        print(f"Loaded processed dataset: {len(dataset)} samples")
        return dataset
    
    # Load and process original data
    print(f"Processing dataset from: {corpus_path}")
    with open(corpus_path, 'rb') as f:
        data = pickle.load(f)
        text_data = [" ".join(tokens) for tokens in data]
        dataset = Dataset.from_dict({"text": text_data})
    
    # Save processed dataset
    print(f"Saving processed dataset to: {processed_path}")
    with open(processed_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Processed dataset saved: {len(dataset)} samples")
    
    return dataset


def get_device():
    """Get the best available device (GPU if available, otherwise CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def create_scheduler(optimizer, num_training_steps, scheduler_type="linear", warmup_ratio=0.1):
    """
    Create a learning rate scheduler
    
    Args:
        optimizer: The optimizer to schedule
        num_training_steps: Total number of training steps
        scheduler_type: Type of scheduler ("linear", "cosine")
        warmup_ratio: Ratio of warmup steps to total steps
    
    Returns:
        Learning rate scheduler
    """
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    
    if scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        print(f"Created linear scheduler with {num_warmup_steps} warmup steps")
    elif scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        print(f"Created cosine scheduler with {num_warmup_steps} warmup steps")
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    return scheduler


def setup_training_environment():
    """Setup training environment with GPU support"""
    device = get_device()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        # Enable optimizations for better GPU performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    return device
