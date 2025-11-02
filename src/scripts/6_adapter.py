import sys
import os
sys.path.append('/home/tommy/Project/PcodeBERT/src')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import pickle
from adapters import AdapterConfig
from tqdm import tqdm

from configs.adapter_config import get_adapter_config
from models.adapter_models import AdapterEmbeddingModel


class TextPairDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch, tokenizer, max_length=512):
    texts1, texts2 = [item[0] for item in batch], [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)
    
    inputs1 = tokenizer(texts1, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    inputs2 = tokenizer(texts2, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    
    return inputs1, inputs2, labels


def get_loss_fn(loss_type, config):
    if loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "cosine":
        return nn.CosineEmbeddingLoss()
    elif loss_type == "triplet":
        return nn.TripletMarginLoss(margin=config["triplet_margin"], p=config["triplet_p"])
    raise ValueError(f"Unknown loss: {loss_type}")


def compute_loss(loss_type, criterion, emb1, emb2, device):
    if loss_type == "mse":
        return criterion(emb1, emb2)
    elif loss_type == "cosine":
        return criterion(emb1, emb2, torch.ones(emb1.size(0), device=device))
    elif loss_type == "triplet":
        batch_size = emb1.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)
        neg_idx = torch.randperm(batch_size, device=device)
        neg_idx = torch.where(neg_idx == torch.arange(batch_size, device=device), 
                             (neg_idx + 1) % batch_size, neg_idx)
        return criterion(emb1, emb2, emb2[neg_idx])
    raise ValueError(f"Unknown loss: {loss_type}")


def train_epoch(model, dataloader, optimizer, criterion, loss_type, device):
    model.train()
    total_loss, batch_count = 0, 0
    
    for inputs1, inputs2, _ in tqdm(dataloader, desc="Training", leave=False):
        inputs1 = {k: v.to(device) for k, v in inputs1.items()}
        inputs2 = {k: v.to(device) for k, v in inputs2.items()}
        
        emb1 = model(inputs1['input_ids'], inputs1['attention_mask'])
        emb2 = model(inputs2['input_ids'], inputs2['attention_mask'])
        loss = compute_loss(loss_type, criterion, emb1, emb2, device)
        
        if loss.item() > 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
    
    return total_loss / batch_count if batch_count > 0 else 0


def validate(model, dataloader, criterion, loss_type, device):
    model.eval()
    total_loss, batch_count = 0, 0
    
    with torch.no_grad():
        for inputs1, inputs2, _ in tqdm(dataloader, desc="Validation", leave=False):
            inputs1 = {k: v.to(device) for k, v in inputs1.items()}
            inputs2 = {k: v.to(device) for k, v in inputs2.items()}
            
            emb1 = model(inputs1['input_ids'], inputs1['attention_mask'])
            emb2 = model(inputs2['input_ids'], inputs2['attention_mask'])
            loss = compute_loss(loss_type, criterion, emb1, emb2, device)
            
            if loss.item() > 0:
                total_loss += loss.item()
                batch_count += 1
    
    return total_loss / batch_count if batch_count > 0 else 0


def train_with_loss(loss_type, config):
    print(f"\n{'='*60}\nTraining with {loss_type.upper()} Loss\n{'='*60}")
    
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    
    adapter_config = AdapterConfig.load(
        config["adapter_config"],
        reduction_factor=config["reduction_factor"],
        non_linearity=config["non_linearity"],
        leave_out=config["leave_out"]
    )
    
    print(f"Adapter Config: {adapter_config}")
    
    model = AdapterEmbeddingModel(
        model_name=config["model_name"],
        adapter_config=adapter_config,
        adapter_name=config["adapter_name"],
        input_dim=config["input_dim"],
        output_dim=config["output_dim"],
        hidden_dim=config["hidden_dim"]
    ).to(device)
    
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    criterion = get_loss_fn(loss_type, config)
    
    scheduler = (CosineAnnealingLR(optimizer, T_max=config["epochs"]) 
                if config["scheduler_type"] == "cosine" 
                else ReduceLROnPlateau(optimizer, mode='min', factor=config["scheduler_factor"], 
                                      patience=config["scheduler_patience"]))
    
    dataset = TextPairDataset(config["data_path"])
    val_size = int(len(dataset) * config["val_split"])
    train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                            collate_fn=lambda b: collate_fn(b, model.tokenizer, config["max_length"]))
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False,
                          collate_fn=lambda b: collate_fn(b, model.tokenizer, config["max_length"]))
    
    best_val_loss = float('inf')
    patience_counter = 0
    save_path = os.path.join(config["save_dir"], f"{config['save_model_name']}_{loss_type}")
    
    for epoch in tqdm(range(config["epochs"]), desc="Epochs"):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, loss_type, device)
        val_loss = validate(model, val_loader, criterion, loss_type, device)
        
        if config["scheduler_type"] == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save_adapter(save_path)
            print(f"Epoch {epoch+1}: val_loss={val_loss:.4f} (saved)")
        else:
            patience_counter += 1
            if patience_counter >= config.get("early_stop_patience", 10):
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"Best val loss: {best_val_loss:.4f}\n")


def main():
    config = get_adapter_config()
    torch.manual_seed(config["seed"])
    os.makedirs(config["save_dir"], exist_ok=True)
    
    print(f"Starting Adapter Training\nModel: {config['model_name']}\nLoss: {', '.join(config['loss_functions'])}")
    
    for loss_type in config["loss_functions"]:
        try:
            train_with_loss(loss_type, config)
        except Exception as e:
            print(f"Error with {loss_type}: {str(e)}")
            continue
    
    print("Training completed!")


if __name__ == '__main__':
    main()
