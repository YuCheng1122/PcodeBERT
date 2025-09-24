from transformers import RobertaConfig

def get_pretrain_config():
    VOCAB_SIZE = 68
    MAX_LENGTH = 512
    BASE_PATH = "/home/tommy/Project/PcodeBERT"

    config = {
        # Model configuration
        "model_config": RobertaConfig(
            vocab_size=VOCAB_SIZE,
            max_position_embeddings= MAX_LENGTH+ 2,
            num_attention_heads=8,
            num_hidden_layers=6,
            type_vocab_size=1,
            hidden_size=256,
        ),
        
        # Data paths
        "vocab_path": f"{BASE_PATH}/outputs/tokenizer/vocab.txt",
        "corpus_path": f"{BASE_PATH}/outputs/preprocessed/pcode_corpus_x86_64.pkl",
        "output_dir": f"{BASE_PATH}/outputs/models/pretrain",
        "tokenizer_output_path": f"{BASE_PATH}/outputs/tokenizer/pcode_tokenizer.json",
        "checkpoint_dir": f"{BASE_PATH}/checkpoints",
        
        # Training parameters
        "epochs": 10,
        "batch_size": 16,
        "learning_rate": 5e-5,
        "max_length": MAX_LENGTH,
        "vocab_size": VOCAB_SIZE,
        
        # MLM parameters
        "mlm_probability": 0.15,
        
        # Tokenizer parameters
        "special_tokens": {
            "unk_token": "[UNK]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "mask_token": "[MASK]"
        },
        
        # Saving parameters
        "save_steps": 1000,
        "save_total_limit": 2,
        "logging_steps": 50
    }
    return config
