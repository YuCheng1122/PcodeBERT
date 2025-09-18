from transformers import RobertaConfig

def get_pretrain_config():
    VOCAB_SIZE = 69
    BASE_PATH = "/home/tommy/Projects/PcodeBERT"

    config = {
        # Model configuration
        "model_config": RobertaConfig(
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=514,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1,
        ),
        
        # Data paths
        "tokenizer_path": f"{BASE_PATH}/outputs/tokenizer/vocab.txt",
        "corpus_path": f"{BASE_PATH}/outputs/preprocessed/pcode_corpus_x86_64.pkl",
        "output_dir": f"{BASE_PATH}/outputs/models/pretrain",
        "checkpoint_dir": f"{BASE_PATH}/checkpoints",
        
        # Training parameters
        "epochs": 10,
        "batch_size": 16,
        "learning_rate": 5e-5,
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
