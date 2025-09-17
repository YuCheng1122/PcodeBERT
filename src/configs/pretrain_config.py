# src/configs/pretrain_config.py

from pathlib import Path

# --- 1. 流程控制開關 ---
# 設定 True 來執行該步驟，設定 False 則跳過
DO_BUILD_CORPUS = True
DO_TRAIN_TOKENIZER = True
DO_TRAIN_MODEL = True

# --- 2. 路徑設定 ---
# 專案根目錄
PROJECT_ROOT = Path("/home/tommy/Projects/PcodeBERT") # 請確認此路徑

# -- 輸入路徑 --
RAW_DATA_DIR = PROJECT_ROOT / "reverse/results"
METADATA_CSV_PATH = PROJECT_ROOT / "dataset/csv/base_dataset_filtered.csv"

# -- 輸出路徑 --
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TOKENIZER_DIR = OUTPUT_DIR / "tokenizer"
PREPROCESSED_DATA_DIR = OUTPUT_DIR / "preprocessed_data"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "models/pcodebert-v1"
TEMP_DIR = PREPROCESSED_DATA_DIR / "temp_files" # 暫存檔目錄
ERROR_LOG_PATH = OUTPUT_DIR / "error_log.txt"

# --- 3. 資料準備設定 ---
TARGET_CPU = "Advanced Micro Devices X86-64" # 請確保與 CSV 中完全一致
CORPUS_PKL_PATH = PREPROCESSED_DATA_DIR / f"pcode_corpus_{TARGET_CPU.replace(' ', '_')}.pkl"

# --- 4. Tokenizer 訓練設定 ---
# 由於您的詞彙表是固定的，這裡我們是「產生」詞彙表，而不是「訓練」
# VOCAB_SIZE 和 MIN_FREQUENCY 在此情境下不會被使用
VOCAB_FILE = TOKENIZER_DIR / "vocab.txt"

# --- 5. 模型架構設定 ---
MODEL_MAX_LEN = 512
MODEL_NUM_LAYERS = 6  # Transformer 層數 (6 是一個較小的尺寸，適合快速實驗)
MODEL_NUM_HEADS = 12 # 注意力頭的數量

# --- 6. 模型訓練設定 (TrainingArguments) ---
TRAIN_NUM_EPOCHS = 10
TRAIN_PER_DEVICE_BATCH_SIZE = 16
TRAIN_FP16 = True # 如果您的 GPU 支援，設為 True 可加速並節省記憶體
TRAIN_SAVE_STEPS = 10_000
TRAIN_SAVE_TOTAL_LIMIT = 2