def get_gnn_config():
    BASE_PATH = "/home/tommy/Project/PcodeBERT"
    
    config = {
        "classification": False,  
        "source_cpus": ["AMD X86-64"],     
        "target_cpus": ["ARM-32"],        
        
        "csv_path": f"{BASE_PATH}/dataset/csv/base_dataset_filtered_v2.csv",
        "graph_dir": f"{BASE_PATH}/outputs/embeddings_200",
        "cache_file": f"{BASE_PATH}/outputs/cache/gnn_data_ppc_200.pkl",
        "model_output_dir": f"{BASE_PATH}/outputs/models/gnn",
        
        "batch_size": 32,
        "hidden_channels": 128,
        "learning_rate": 0.01,
        "epochs": 200,
        "patience": 20,
        
        "seeds": [42, 123, 2025, 31415, 8888],
        "device": "cuda"
    }
    
    return config
