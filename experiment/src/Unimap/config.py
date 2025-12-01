def get_lstm_config():
    BASE_PATH = "/home/tommy/Project/PcodeBERT/experiment"
    
    config = {
        "classification": False,  
        "source_cpus": ["x86_64"],     
        "target_cpus": ["ARM"],

        "csv_path": f"{BASE_PATH}/dataset/csv/merged_adjusted_filtered.csv",
        "source_dir": "/home/tommy/Project/PcodeBERT/experiment/outputs/data",
        "cache_file": f"/home/tommy/Project/PcodeBERT/experiment/cache",
        "model_output_dir": f"/home/tommy/Project/PcodeBERT/experiment/outputs/model",
        
        "input_dim": 200,
        "output_dim": 64,
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 200,
        "patience": 20,
        
        
        "seeds": [42, 123, 2025, 31415, 8888],
        "device": "cuda"
    }
    
    return config
