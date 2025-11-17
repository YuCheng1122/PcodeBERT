def get_gnn_config():
    BASE_PATH = "/home/tommy/Project/PcodeBERT"
    
    config = {
        "classification": False,  
        "source_cpus": ["x86_64"],     
        "target_cpus": ["ARM"],

        "csv_path": f"{BASE_PATH}/dataset/csv/merged_adjusted_filtered.csv",
<<<<<<< HEAD
        "graph_dir": f"{BASE_PATH}/outputs/models/GNN/gpickle_merged_adjusted_filtered_epoch30_cosine",
        "cache_file": f"{BASE_PATH}/outputs/cache/gpickle_merged_adjusted_filtered_epoch30_cosine.pkl",
        "model_output_dir": f"{BASE_PATH}/outputs/models/GNN/model_cosine_epoch_30",
=======
        "graph_dir": f"{BASE_PATH}/outputs/data/GNN/gpickle_merged_adjusted_filtered_cosine_epoch5",
        "cache_file": f"{BASE_PATH}/outputs/cache/embeddings_gpickle_merged_adjusted_filtered_cosine_epoch5.pkl",
        "model_output_dir": f"{BASE_PATH}/outputs/models/GNN/model_epoch_5_cosine",
>>>>>>> cc056ac5b2b4f41a1eb71d253cb6ebf75155470b
        
        "batch_size": 32,
        "hidden_channels": 128,
        "learning_rate": 0.01,
        "epochs": 200,
        "patience": 20,
        
        "seeds": [42, 123, 2025, 31415, 8888],
        "device": "cuda"
    }
    
    return config
