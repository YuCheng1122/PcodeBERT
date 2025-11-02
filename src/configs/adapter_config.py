"""
Adapter 訓練配置檔案
使用 adapter-transformers 套件進行 Adapter 訓練
"""

def get_adapter_config():
    """
    返回 Adapter 訓練的配置參數
    
    Returns:
        dict: 包含所有訓練參數的配置字典
    """
    return {
        "model_name": "/home/tommy/Project/PcodeBERT/outputs/models/RoBERTa/model_epoch_100",
        "adapter_name": "pcode_adapter",  
        
        "adapter_config": "pfeiffer",
        "reduction_factor": 32,  
        "non_linearity": "gelu",  
        "leave_out": [0, 1, 2, 3, 4], 
        "input_dim": 256,  
        "output_dim": 256,
        "hidden_dim": 128, 
        "use_projection": False, 
        
        "data_path": "/home/tommy/Project/PcodeBERT/outputs/data/Adapters/train_x86_64_arm_32_functions_deduped.pickle",
        "val_data_path": None,  
        "val_split": 0.2,  
        
        "batch_size": 128,
        "learning_rate": 1e-4, 
        "epochs": 5,
        "weight_decay": 0.01,
        
        "loss_functions": ["mse"],
        "triplet_margin": 1.0,
        "triplet_p": 2,
        
        "scheduler_type": "cosine",  
        "scheduler_patience": 10,  
        "scheduler_factor": 0.5,   
        
        "early_stop_patience": 10,
        
        "device": "cuda",
        "save_dir": "/home/tommy/Project/PcodeBERT/outputs/adapter",
        "save_model_name": "adapter_roberta",  
        
        "max_length": 512, 
        "seed": 42,
        "log_interval": 10,  
    }


def get_inference_config():
    """
    返回推理/應用 Adapter 的配置參數
    
    Returns:
        dict: 推理配置字典
    """
    return {
        "model_path": "/home/tommy/Project/PcodeBERT/outputs/models/RoBERTa/model_epoch_100",
        "adapter_path": "/home/tommy/Project/PcodeBERT/outputs/adapter/adapter_roberta_mse",
        "adapter_name": "pcode_adapter",
        "input_path": "/home/tommy/Project/PcodeBERT/outputs/data/GNN/gpickle_merged_adjusted_filtered",
        "output_path": "/home/tommy/Project/PcodeBERT/outputs/data/GNN/gpickle_merged_adjusted_filtered_adapter",
        "csv_path": "/home/tommy/Project/PcodeBERT/dataset/csv/merged_adjusted_filtered.csv",
        "target_cpus": ["x86_64", "ARM"],
        "batch_size": 64,
        "device": "cuda",
        "max_length": 512,
    }
