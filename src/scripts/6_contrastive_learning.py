import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.adapter import create_adapter
from configs.contrastive_config import get_contrastive_config
from scripts.utils import setup_training_environment


def similarity_score(x1, x2):
    """
    計算兩個向量之間的相似度分數
    使用歐氏距離轉換為相似度
    
    Args:
        x1: 第一組向量 [batch_size, dim]
        x2: 第二組向量 [batch_size, dim]
    
    Returns:
        相似度分數 [batch_size]
    """
    dist = torch.norm(x1 - x2, dim=1)
    return 1 / (1 + dist)


def contrastive_loss(x1, x2, labels):
    """
    對比式學習損失函數
    
    Args:
        x1: 第一組特徵向量
        x2: 第二組特徵向量
        labels: 標籤 (1 表示相似, 0 表示不相似)
    
    Returns:
        損失值
    """
    sims = similarity_score(x1, x2)
    loss = F.binary_cross_entropy(sims, labels.float())
    return loss


def train_adapter_step(batch, adapter, optimizer, device):
    """
    訓練 adapter 的單個步驟
    
    Args:
        batch: 包含 (v1, v2, labels) 的批次數據
        adapter: adapter 模型
        optimizer: 優化器
        device: 設備 (cpu/cuda)
    
    Returns:
        損失值
    """
    v1, v2, labels = [b.to(device) for b in batch]
    
    adapter.train()
    optimizer.zero_grad()

    # 通過 adapter 轉換特徵
    z1 = adapter(v1)
    z2 = adapter(v2)

    # 計算損失
    loss = contrastive_loss(z1, z2, labels)
    loss.backward()
    optimizer.step()
    
    return loss.item()


def evaluate_adapter(dataloader, adapter, device):
    """
    評估 adapter 性能
    
    Args:
        dataloader: 評估數據載入器
        adapter: adapter 模型
        device: 設備
    
    Returns:
        平均損失和準確率
    """
    adapter.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            v1, v2, labels = [b.to(device) for b in batch]
            
            z1 = adapter(v1)
            z2 = adapter(v2)
            
            # 計算損失
            loss = contrastive_loss(z1, z2, labels)
            total_loss += loss.item()
            
            # 計算準確率
            sims = similarity_score(z1, z2)
            predictions = (sims > 0.5).long()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def load_data(path, batch_size=64, shuffle=True):
    """
    載入訓練數據
    
    Args:
        path: pickle 檔案路徑
        batch_size: 批次大小
        shuffle: 是否打亂數據
    
    Returns:
        DataLoader
    """
    print(f"Loading data from: {path}")
    
    with open(path, "rb") as f:
        data = pickle.load(f)  # list of (vec1, vec2, label)

    v1 = torch.tensor(np.array([d[0] for d in data]), dtype=torch.float32)
    v2 = torch.tensor(np.array([d[1] for d in data]), dtype=torch.float32)
    labels = torch.tensor(np.array([d[2] for d in data]), dtype=torch.long)

    print(f"Loaded {len(data)} samples")
    print(f"Vector dimension: {v1.shape[1]}")
    
    dataset = TensorDataset(v1, v2, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def main():
    # 載入配置
    config = get_contrastive_config()
    
    # 設置環境
    device = setup_training_environment()
    print(f"Using device: {device}")
    
    # 載入訓練數據
    print("\n" + "="*50)
    print("Loading Training Data")
    print("="*50)
    train_loader = load_data(
        config["train_data_path"], 
        batch_size=config["batch_size"], 
        shuffle=True
    )
    
    # 載入驗證數據（如果有）
    val_loader = None
    if config["val_data_path"] and os.path.exists(config["val_data_path"]):
        print("\n" + "="*50)
        print("Loading Validation Data")
        print("="*50)
        val_loader = load_data(
            config["val_data_path"], 
            batch_size=config["batch_size"], 
            shuffle=False
        )
    
    # 創建 adapter
    print("\n" + "="*50)
    print("Creating Adapter Model")
    print("="*50)
    print(f"Adapter type: {config['adapter_type']}")
    print(f"Input dimension: {config['input_dim']}")
    print(f"Hidden dimension: {config['hidden_dim']}")
    
    adapter = create_adapter(
        adapter_type=config["adapter_type"],
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_blocks=config["num_blocks"],
        num_layers=config["num_layers"]
    ).to(device)
    
    num_params = sum(p.numel() for p in adapter.parameters())
    print(f"Total parameters: {num_params:,}")
    
    # 創建優化器
    optimizer = torch.optim.Adam(adapter.parameters(), lr=config["learning_rate"])
    
    # 創建輸出目錄
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # 訓練循環
    print("\n" + "="*50)
    print(f"Starting Training for {config['epochs']} Epochs")
    print("="*50)
    
    best_val_loss = float('inf')
    
    for epoch in range(config["epochs"]):
        # 訓練階段
        adapter.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        
        for batch in pbar:
            loss = train_adapter_step(batch, adapter, optimizer, device)
            total_loss += loss
            pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        avg_train_loss = total_loss / len(train_loader)
        
        # 評估階段
        log_msg = f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {avg_train_loss:.4f}"
        
        if val_loader:
            val_loss, val_acc = evaluate_adapter(val_loader, adapter, device)
            log_msg += f" - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(
                    config["output_dir"], 
                    config["save_name"].replace('.pth', '_best.pth')
                )
                torch.save(adapter.state_dict(), best_model_path)
                print(f"  -> Saved best model to {best_model_path}")
        
        print(log_msg)
        
        # 定期保存 checkpoint
        if (epoch + 1) % config["save_every"] == 0:
            checkpoint_path = os.path.join(
                config["output_dir"],
                config["save_name"].replace('.pth', f'_epoch{epoch+1}.pth')
            )
            torch.save(adapter.state_dict(), checkpoint_path)
            print(f"  -> Saved checkpoint to {checkpoint_path}")
    
    # 保存最終模型
    final_model_path = os.path.join(config["output_dir"], config["save_name"])
    torch.save(adapter.state_dict(), final_model_path)
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Final model saved to: {final_model_path}")
    
    if val_loader:
        print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
