def get_adapter_config():
    return {
        "model_name": "/home/tommy/Project/PcodeBERT/outputs/models/RoBERTa/model_epoch_25",
        "adapter_name": "pcode_adapter",  
        
        "adapter_config": "pfeiffer",
        "reduction_factor": 32,  
        "non_linearity": "gelu",  
        "use_projection": False,
        
        "data_path": "/home/tommy/Project/PcodeBERT/outputs/data/Adapters/train_x86_64_arm_32_functions_deduped.pickle",
        "val_data_path": None,  
        "val_split": 0.2,  
        
        "batch_size": 64,
        "learning_rate": 5e-5,
        "warmup_ratio": 0.1,
        "epochs": 30,
        "weight_decay": 0.01,
        
        "loss_functions": ["mse"],
        "scheduler_type": "cosine",
        
        "early_stop_patience": 10,
        
        "device": "cuda",
        "save_base_dir": "/home/tommy/Project/PcodeBERT/outputs/models/Adapters/manual",
        "save_model_name": "adapter_roberta",  
        
        "max_length": 512, 
        "seed": 42,
        "log_interval": 10,  
    }


def get_inference_config():
    return {
        "model_path": "/home/tommy/Project/PcodeBERT/outputs/models/RoBERTa/model_epoch_25",
        "adapter_path": "/home/tommy/Project/PcodeBERT/outputs/models/Adapters/ablation/mse_6layers_epoch10",
        "adapter_name": "pcode_adapter",
        "input_path": "/home/tommy/Project/PcodeBERT/outputs/data/GNN/gpickle_merged_adjusted_filtered",
        "output_path": "/home/tommy/Project/PcodeBERT/outputs/models/GNN/embeddings_mse_6layers_epoch10",
        "csv_path": "/home/tommy/Project/PcodeBERT/dataset/csv/merged_adjusted_filtered.csv",
        "target_cpus": ["x86_64", "ARM"],
        "batch_size": 64,
        "device": "cuda",
        "max_length": 512,
    }
