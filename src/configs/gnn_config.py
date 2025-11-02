def get_gnn_config():
    BASE_PATH = "/home/tommy/Project/PcodeBERT"
    
    config = {
        "classification": False,  
        "source_cpus": ["x86_64"],     
        "target_cpus": ["ARM"],

        "csv_path": f"{BASE_PATH}/dataset/csv/merged_filtered_final.csv",
        "graph_dir": f"{BASE_PATH}/outputs/data/GNN/gpickle_merged_adjusted_filtered_adapter",
        "cache_file": f"{BASE_PATH}/outputs/cache/gnn_data_adapter_100_mse.pkl",
        "model_output_dir": f"{BASE_PATH}/outputs/models/GNN/gnn_adapter_100_mse",
        
        "batch_size": 32,
        "hidden_channels": 64,
        "learning_rate": 0.01,
        "epochs": 200,
        "patience": 20,
        
        "seeds": [42, 123, 2025, 31415, 8888],
        "device": "cuda"
    }
    
    return config
