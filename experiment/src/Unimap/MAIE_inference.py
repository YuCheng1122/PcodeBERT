import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import PaperLSTM
from config import get_lstm_config
from train_utils import load_pkl_with_labels

class SequenceDataset(Dataset):
    def __init__(self, data_with_labels, max_len=500):
        self.samples = []
        for data, label in data_with_labels:
            if len(data) > 0:
                seq = np.array(data)
                if len(seq) > max_len:
                    seq = seq[:max_len]
                self.samples.append((seq, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.float32), len(seq)

def collate_fn(batch):
    seqs, labels, lens = zip(*batch)
    max_len = max(lens)
    padded_seqs = []
    for seq in seqs:
        if len(seq) < max_len:
            padding = torch.zeros(max_len - len(seq), seq.shape[1])
            seq = torch.cat([seq, padding], dim=0)
        padded_seqs.append(seq)
    return torch.stack(padded_seqs), torch.tensor(labels), torch.tensor(lens)

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for embeddings, labels, _ in dataloader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(embeddings).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for embeddings, labels, _ in dataloader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            
            outputs = model(embeddings).squeeze()
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total

def main():
    config = get_lstm_config()
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    source_cpu = config['source_cpus'][0]
    target_cpu = config['target_cpus'][0]
    
    print(f"Loading source data: {source_cpu}")
    source_data = load_pkl_with_labels(config['source_dir'], source_cpu, config['csv_path'], config['cache_file'])
    
    print(f"Loading target data: {target_cpu}")
    target_data = load_pkl_with_labels(config['source_dir'], target_cpu, config['csv_path'], config['cache_file'])
    
    train_dataset = SequenceDataset(source_data)
    test_dataset = SequenceDataset(target_data)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], collate_fn=collate_fn)
    
    model = PaperLSTM(input_dim=config['input_dim'], use_embedding=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.BCELoss()
    
    best_test_acc = 0
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Test - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
            os.makedirs(config['model_output_dir'], exist_ok=True)
            torch.save(model.state_dict(), f"{config['model_output_dir']}/best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"Training complete. Best test acc: {best_test_acc:.4f}")

if __name__ == "__main__":
    main()
