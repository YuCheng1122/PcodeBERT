def get_contrastive_config():
    """
    對比式學習的配置檔案
    用於訓練 adapter 模型
    """
    BASE_PATH = "/home/tommy/Project/PcodeBERT"
    
    config = {
        "train_data_path": f"{BASE_PATH}/outputs/alignment_vector/train_arm_vector_contrastive_bert.pickle",
        "val_data_path": None, 
        
        "adapter_type": "lstm",  
        "input_dim": 256,       
        "hidden_dim": 256,      
        "num_blocks": 2,       
        "num_layers": 1,       
        

        "batch_size": 64,
        "learning_rate": 1e-3,
        "epochs": 50,
        "save_every": 5,        

        "output_dir": f"{BASE_PATH}/outputs/adapters",
        "save_name": "adapter.pth",  
    }
    
    return config


def get_custom_contrastive_config(
    train_data_path=None,
    val_data_path=None,
    adapter_type="mlp",
    input_dim=256,
    hidden_dim=256,
    batch_size=64,
    learning_rate=1e-3,
    epochs=10
):
    """
    獲取自訂的對比式學習配置
    
    Args:
        train_data_path: 訓練數據路徑
        val_data_path: 驗證數據路徑
        adapter_type: adapter 類型
        input_dim: 輸入維度
        hidden_dim: 隱藏層維度
        batch_size: 批次大小
        learning_rate: 學習率
        epochs: 訓練輪數
    
    Returns:
        配置字典
    """
    BASE_PATH = "/home/tommy/Project/PcodeBERT"
    
    config = get_contrastive_config()
    
    # 更新自訂參數
    if train_data_path:
        config["train_data_path"] = train_data_path
    if val_data_path:
        config["val_data_path"] = val_data_path
    
    config["adapter_type"] = adapter_type
    config["input_dim"] = input_dim
    config["hidden_dim"] = hidden_dim
    config["batch_size"] = batch_size
    config["learning_rate"] = learning_rate
    config["epochs"] = epochs
    
    # 根據 adapter 類型調整保存名稱
    config["save_name"] = f"{adapter_type}_adapter.pth"
    
    return config
