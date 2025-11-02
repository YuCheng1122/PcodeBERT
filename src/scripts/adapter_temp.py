import sys
import os
sys.path.append('/home/tommy/Project/PcodeBERT/src')
import torch 
import pickle
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from configs.adapter import get_adapter_config
from models.adapter_temp import AdapterMapper
from transformers import RobertaModel, AutoTokenizer

os.makedirs('/home/tommy/Project/PcodeBERT/outputs/adapter', exist_ok=True)

config = get_adapter_config()
tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
roberta = RobertaModel.from_pretrained(config["model_name"])
for param in roberta.parameters():
    param.requires_grad = False

def text_to_embedding(texts, model, tokenizer, device):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]

def load_and_cache_embeddings(data_path, cache_dir, batch_size=64):
    os.makedirs(cache_dir, exist_ok=True)
    
    data_file_name = os.path.basename(data_path).replace('.pickle', '')
    cache_file = os.path.join(cache_dir, f"{data_file_name}_embeddings.pt")
    
    if os.path.exists(cache_file):
        print(f"Loading from cache: {cache_file}")
        cached_data = torch.load(cache_file)
        return cached_data['v1'], cached_data['v2'], cached_data['labels']
    
    print(f"Computing embeddings...")
    
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    roberta.to(device)
    
    texts1 = [d[0] for d in data]
    texts2 = [d[1] for d in data]
    labels = [d[2] for d in data]
    
    v1_list = []
    v2_list = []
    for i in range(0, len(texts1), batch_size):
        batch_texts1 = texts1[i:i+batch_size]
        batch_texts2 = texts2[i:i+batch_size]
        v1_list.append(text_to_embedding(batch_texts1, roberta, tokenizer, device).cpu())
        v2_list.append(text_to_embedding(batch_texts2, roberta, tokenizer, device).cpu())
    
    v1 = torch.cat(v1_list, dim=0)
    v2 = torch.cat(v2_list, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    
    torch.save({'v1': v1, 'v2': v2, 'labels': labels}, cache_file)
    print(f"Saved to cache: {cache_file}")
    
    return v1, v2, labels

def load_data(path, cache_dir, batch_size=64, val_split=0.2):
    v1, v2, labels = load_and_cache_embeddings(path, cache_dir, batch_size)
    
    dataset = TensorDataset(v1, v2, labels)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def get_loss_function(loss_type, config):
    if loss_type == "mse":
        return torch.nn.MSELoss()
    elif loss_type == "cosine":
        return torch.nn.CosineEmbeddingLoss()
    elif loss_type == "triplet":
        return torch.nn.TripletMarginLoss(
            margin=config.get("triplet_margin", 1.0),
            p=config.get("triplet_p", 2)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def compute_loss(loss_type, criterion, v1_mapped, v2, device):
    if loss_type == "mse":
        return criterion(v1_mapped, v2)
    
    elif loss_type == "cosine":
        batch_size = v1_mapped.size(0)
        target = torch.ones(batch_size).to(device)
        return criterion(v1_mapped, v2, target)
    
    elif loss_type == "triplet":
        batch_size = v1_mapped.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        neg_indices = torch.randperm(batch_size).to(device)
        for i in range(batch_size):
            if neg_indices[i] == i:
                neg_indices[i] = (i + 1) % batch_size
        
        negatives = v2[neg_indices]
        return criterion(v1_mapped, v2, negatives)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def train_with_loss(loss_type):
    print(f"\n{'='*50}")
    print(f"Training with {loss_type.upper()} loss")
    print(f"{'='*50}\n")
    
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model = AdapterMapper(config["input_dim"], config["output_dim"], config["hidden_dim"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = get_loss_function(loss_type, config)
    
    if config["scheduler_type"] == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config["scheduler_factor"], 
                                      patience=config["scheduler_patience"])
    
    train_loader, val_loader = load_data(
        config["data_path"], 
        config["cache_dir"],
        config["batch_size"], 
        config["val_split"]
    )
    
    best_val_loss = float('inf')
    save_dir = os.path.dirname(config["save_path"])
    save_name = os.path.basename(config["save_path"]).replace('.pt', f'_{loss_type}.pt')
    loss_save_path = os.path.join(save_dir, save_name)
    
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0
        batch_count = 0
        for v1, v2, _ in train_loader:
            v1, v2 = v1.to(device), v2.to(device)
            optimizer.zero_grad()
            v1_mapped = model(v1)
            loss = compute_loss(loss_type, criterion, v1_mapped, v2, device)
            
            if loss.item() > 0:
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                batch_count += 1
        
        train_loss = train_loss / batch_count if batch_count > 0 else 0
        
        model.eval()
        val_loss = 0
        val_batch_count = 0
        with torch.no_grad():
            for v1, v2, _ in val_loader:
                v1, v2 = v1.to(device), v2.to(device)
                v1_mapped = model(v1)
                loss = compute_loss(loss_type, criterion, v1_mapped, v2, device)
                if loss.item() > 0:
                    val_loss += loss.item()
                    val_batch_count += 1
        
        val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0
        
        if config["scheduler_type"] == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{config["epochs"]}, Train: {train_loss:.4f}, Val: {val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), loss_save_path)
            print(f'Saved: {loss_save_path} (val_loss: {val_loss:.4f})')
    
    print(f'\nBest val loss: {best_val_loss:.4f}\n')

def train():
    loss_functions = config.get("loss_functions", ["mse"])
    
    print(f"\nTraining with: {', '.join(loss_functions)}")
    print(f"Cache: {config['cache_dir']}\n")
    
    for loss_type in loss_functions:
        try:
            train_with_loss(loss_type)
        except Exception as e:
            print(f"\nError with {loss_type}: {str(e)}\n")
            continue
    
    print("All training completed!")

if __name__ == '__main__':
    train()
