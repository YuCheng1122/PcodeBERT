import sys
import os
sys.path.append('/home/tommy/Project/PcodeBERT/src')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import get_linear_schedule_with_warmup
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


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss, batch_count = 0, 0
    
    for inputs1, inputs2, _ in tqdm(dataloader, desc="Training", leave=False):
        inputs1 = {k: v.to(device) for k, v in inputs1.items()}
        inputs2 = {k: v.to(device) for k, v in inputs2.items()}
        
        emb1 = model(inputs1['input_ids'], inputs1['attention_mask'])
        emb2 = model(inputs2['input_ids'], inputs2['attention_mask'])
        loss = criterion(emb1, emb2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        batch_count += 1
    
    return total_loss / batch_count if batch_count > 0 else 0


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss, batch_count = 0, 0
    
    with torch.no_grad():
        for inputs1, inputs2, _ in tqdm(dataloader, desc="Validation", leave=False):
            inputs1 = {k: v.to(device) for k, v in inputs1.items()}
            inputs2 = {k: v.to(device) for k, v in inputs2.items()}
            
            emb1 = model(inputs1['input_ids'], inputs1['attention_mask'])
            emb2 = model(inputs2['input_ids'], inputs2['attention_mask'])
            loss = criterion(emb1, emb2)
            
            total_loss += loss.item()
            batch_count += 1
    
    return total_loss / batch_count if batch_count > 0 else 0


def train_adapter(config):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    
    adapter_config = AdapterConfig.load(
        config["adapter_config"],
        reduction_factor=config["reduction_factor"],
        non_linearity=config["non_linearity"]
    )
    
    model = AdapterEmbeddingModel(
        model_name=config["model_name"],
        adapter_config=adapter_config,
        adapter_name=config["adapter_name"]
    ).to(device)
    
    model_name = os.path.basename(config["model_name"])
    save_dir = os.path.join(config["save_base_dir"], model_name)
    os.makedirs(save_dir, exist_ok=True)
    
    dataset = TextPairDataset(config["data_path"])
    val_size = int(len(dataset) * config["val_split"])
    train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                            collate_fn=lambda b: collate_fn(b, model.tokenizer, config["max_length"]))
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False,
                          collate_fn=lambda b: collate_fn(b, model.tokenizer, config["max_length"]))
    
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    steps_per_epoch = len(train_loader)
    num_training_steps = steps_per_epoch * config["epochs"]
    num_warmup_steps = int(num_training_steps * config["warmup_ratio"])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    criterion = nn.MSELoss()
    
    log_file = os.path.join(save_dir, "training_log.txt")
    with open(log_file, "w") as f:
        f.write("epoch,train_loss,val_loss\n")
    
    checkpoint_epochs = [10, 20, 30]
    
    for epoch in tqdm(range(config["epochs"]), desc="Epochs"):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        with open(log_file, "a") as f:
            f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f}\n")
        
        if (epoch + 1) in checkpoint_epochs:
            checkpoint_dir = os.path.join(save_dir, f"epoch_{epoch+1}")
            model.save_adapter(checkpoint_dir)
            print(f"Saved checkpoint at epoch {epoch+1}")


def main():
    config = get_adapter_config()
    torch.manual_seed(config["seed"])
    train_adapter(config)
    print("Training completed!")


if __name__ == '__main__':
    main()
