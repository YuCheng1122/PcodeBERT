from transformers import RobertaConfig

def get_pretrain_config():

    VOCAB_SIZE = 69

    config = {
        "model_config": RobertaConfig(
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=514,
            num_attention_heads=12,
            num_hidden_layers=6,
            type_vocab_size=1,
        ),
        "tokenizer_path": "/home/tommy/Projects/PcodeBERT/outputs/tokenizer/vocab.txt",
        "corpus_path": "/home/tommy/Projects/PcodeBERT/outputs/preprocessed/pcode_corpus_x86_64.pkl",
        "output_dir": "/home/tommy/Projects/PcodeBERT/outputs/models/pretrain",
        "epochs": 10,
        "batch_size": 16,
        "learning_rate": 5e-5,
        "vocab_size": VOCAB_SIZE, 
    }
    return config
