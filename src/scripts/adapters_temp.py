import torch 
import pickle
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from configs.adapter import get_adapter_config
from transformers import RobertaTokenizer, RobertaModel

config = get_adapter_config()
tokenizer = RobertaTokenizer.from_pretrained(config["model_name"])
model = RobertaModel.from_pretrained(config["model_name"])

for param in model.parameters():
    param.requires_grad = False


def load_data(path, batch_size=64, shuffle=True):
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