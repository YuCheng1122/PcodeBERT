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


def train_epoch(model, dataloader, optimizer, scheduler, criteria, device):
    model.train()
    total_loss, batch_count = 0, 0
    
    for inputs1, inputs2, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs1 = {k: v.to(device) for k, v in inputs1.items()}
        inputs2 = {k: v.to(device) for k, v in inputs2.items()}
        labels = labels.to(device)
        
        emb1 = model(inputs1['input_ids'], inputs1['attention_mask'])
        emb2 = model(inputs2['input_ids'], inputs2['attention_mask'])
        
        loss = 0
        for criterion in criteria:
            if isinstance(criterion, nn.CosineEmbeddingLoss):
                loss += criterion(emb1, emb2, labels)
            else:
                loss += criterion(emb1, emb2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        batch_count += 1
    
    return total_loss / batch_count if batch_count > 0 else 0


def validate(model, dataloader, criteria, device):
    model.eval()
    total_loss, batch_count = 0, 0
    
    with torch.no_grad():
        for inputs1, inputs2, labels in tqdm(dataloader, desc="Validation", leave=False):
            inputs1 = {k: v.to(device) for k, v in inputs1.items()}
            inputs2 = {k: v.to(device) for k, v in inputs2.items()}
            labels = labels.to(device)
            
            emb1 = model(inputs1['input_ids'], inputs1['attention_mask'])
            emb2 = model(inputs2['input_ids'], inputs2['attention_mask'])
            
            loss = 0
            for criterion in criteria:
                if isinstance(criterion, nn.CosineEmbeddingLoss):
                    loss += criterion(emb1, emb2, labels)
                else:
                    loss += criterion(emb1, emb2)
            
            total_loss += loss.item()
            batch_count += 1
    
    return total_loss / batch_count if batch_count > 0 else 0


def train_adapter_config(loss_type, save_base_dir, model_path, data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config_name = f"{loss_type}_6layers"
    
    print(f"\n{'='*60}")
    print(f"Training Configuration: {config_name}")
    print(f"Loss Function: {loss_type.upper()}")
    print(f"Total Epochs: 10 (saving checkpoints at epochs 1-10)")
    print(f"{'='*60}")
    
    # Create adapter config
    adapter_config = AdapterConfig.load(
        "pfeiffer",
        reduction_factor=64,
        non_linearity="gelu"
    )
    
    # Create model
    model = AdapterEmbeddingModel(
        model_name=model_path,
        adapter_config=adapter_config,
        adapter_name="pcode_adapter"
    ).to(device)
    
    # Setup optimizer
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-5,
        weight_decay=0.01
    )
    
    # Setup loss function
    criteria = []
    if loss_type == "mse":
        criteria.append(nn.MSELoss())
    elif loss_type == "cosine":
        criteria.append(nn.CosineEmbeddingLoss())
    
    # Load dataset
    dataset = TextPairDataset(data_path)
    val_size = int(len(dataset) * 0.2)
    train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, model.tokenizer, 512)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, model.tokenizer, 512)
    )
    
    # Setup scheduler
    steps_per_epoch = len(train_loader)
    num_training_steps = steps_per_epoch * 10  # 10 epochs
    num_warmup_steps = int(num_training_steps * 0.1)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    checkpoint_epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    for epoch in tqdm(range(10), desc="Epochs"):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criteria, device)
        val_loss = validate(model, val_loader, criteria, device)
        
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        if (epoch + 1) in checkpoint_epochs:
            checkpoint_dir = os.path.join(save_base_dir, f"{config_name}_epoch{epoch+1}")
            model.save_adapter(checkpoint_dir)
            print(f"  → Checkpoint saved at epoch {epoch+1}")
    
    print(f"Training completed for {config_name}\n")


def main():
    model_path = "/home/tommy/Project/PcodeBERT/outputs/models/RoBERTa/model_epoch_25"
    data_path = "/home/tommy/Project/PcodeBERT/outputs/data/Adapters/train_x86_64_arm_32_functions_deduped.pickle"
    save_dir = "/home/tommy/Project/PcodeBERT/outputs/models/Adapters/ablation"
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*60)
    print("Adapter Ablation Training")
    print("="*60)
    print(f"Pretrain Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Output: {save_dir}")
    print(f"Configurations: 2 loss types × 10 epochs = 20 checkpoints")
    print("="*60)
    
    # Train with both loss functions
    for loss_type in ['cosine', 'mse']:
        train_adapter_config(loss_type, save_dir, model_path, data_path)
    
    print("="*60)
    print("All Adapter Training Completed!")
    print("="*60)
    print(f"Total adapters created: 20 (10 checkpoints × 2 loss types)")
    print(f"Saved to: {save_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
