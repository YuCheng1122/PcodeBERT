import sys
import os
sys.path.append('/home/tommy/Project/PcodeBERT/src')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import pickle
from adapters import AdapterConfig
from tqdm import tqdm

from models.adapter_models import AdapterEmbeddingModel


class TextPairDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch, tokenizer):
    texts1 = [item[0] for item in batch]
    texts2 = [item[1] for item in batch]
    inputs1 = tokenizer(texts1, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs2 = tokenizer(texts2, padding=True, truncation=True, max_length=512, return_tensors="pt")
    return inputs1, inputs2


def train_epoch(model, dataloader, optimizer, criterion, loss_type, device):
    model.train()
    total_loss = 0
    
    for inputs1, inputs2 in tqdm(dataloader, desc="Training", leave=False):
        inputs1 = {k: v.to(device) for k, v in inputs1.items()}
        inputs2 = {k: v.to(device) for k, v in inputs2.items()}
        
        emb1 = model(inputs1['input_ids'], inputs1['attention_mask'])
        emb2 = model(inputs2['input_ids'], inputs2['attention_mask'])
        
        if loss_type == "mse":
            loss = criterion(emb1, emb2)
        else:
            loss = criterion(emb1, emb2, torch.ones(emb1.size(0), device=device))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, loss_type, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for inputs1, inputs2 in tqdm(dataloader, desc="Validation", leave=False):
            inputs1 = {k: v.to(device) for k, v in inputs1.items()}
            inputs2 = {k: v.to(device) for k, v in inputs2.items()}
            
            emb1 = model(inputs1['input_ids'], inputs1['attention_mask'])
            emb2 = model(inputs2['input_ids'], inputs2['attention_mask'])
            
            if loss_type == "mse":
                loss = criterion(emb1, emb2)
            else:
                loss = criterion(emb1, emb2, torch.ones(emb1.size(0), device=device))
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train_config(loss_type, num_layers, save_dir, model_path, data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    leave_out = [] if num_layers == 6 else [0, 1, 2, 3]
    config_name = f"{loss_type}_{num_layers}layers"
    
    print(f"\n{'='*60}")
    print(f"Training: {config_name}")
    print(f"Loss: {loss_type.upper()}, Adapter Layers: {num_layers}")
    print(f"Leave out: {leave_out}")
    print(f"{'='*60}")
    
    adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=32, non_linearity="gelu", leave_out=leave_out)
    model = AdapterEmbeddingModel(model_path, adapter_config, "pcode_adapter").to(device)
    
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01)
    criterion = nn.MSELoss() if loss_type == "mse" else nn.CosineEmbeddingLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=30)
    
    dataset = TextPairDataset(data_path)
    val_size = int(len(dataset) * 0.2)
    train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                            collate_fn=lambda b: collate_fn(b, model.tokenizer))
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,
                          collate_fn=lambda b: collate_fn(b, model.tokenizer))
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5
    
    for epoch in range(1, 31):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, loss_type, device)
        val_loss = validate(model, val_loader, criterion, loss_type, device)
        scheduler.step()
        
        print(f"Epoch {epoch:2d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        if epoch in [10, 20, 30]:
            save_path = os.path.join(save_dir, f"{config_name}_epoch{epoch}")
            model.save_adapter(save_path)
            print(f"  → Checkpoint saved")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  → Early stopping at epoch {epoch}")
                break
    
    print(f"Best validation loss: {best_val_loss:.4f}\n")


def main():
    model_path = "/home/tommy/Project/PcodeBERT/outputs/models/RoBERTa/model_epoch_25"
    data_path = "/home/tommy/Project/PcodeBERT/outputs/data/Adapters/train_x86_64_arm_32_functions_deduped.pickle"
    save_dir = "/home/tommy/Project/PcodeBERT/outputs/models/Adapters/ablation"
    
    os.makedirs(save_dir, exist_ok=True)
    
    for loss_type in ['mse', 'cosine']:
        for num_layers in [6]:
            train_config(loss_type, num_layers, save_dir, model_path, data_path)
    
    print("="*60)
    print("All adapter training completed!")
    print("="*60)


if __name__ == '__main__':
    main()
