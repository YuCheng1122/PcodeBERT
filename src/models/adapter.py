import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """殘差塊用於深度特徵轉換"""
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += residual
        return F.relu(out)


class Adapter(nn.Module):
    """基於殘差塊的 Adapter 模型"""
    def __init__(self, dim=256, num_blocks=2):
        super().__init__()
        self.blocks = nn.Sequential(*[ResidualBlock(dim) for _ in range(num_blocks)])
        self.fc_out = nn.Linear(dim, dim)

    def forward(self, x):
        out = self.blocks(x)
        return self.fc_out(out)


class LSTMAdapter(nn.Module):
    """基於 LSTM 的 Adapter 模型"""
    def __init__(self, input_dim=256, hidden_dim=256, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        return self.fc(last_hidden)


class MLPAdapter(nn.Module):
    """基於 MLP 的 Adapter 模型"""
    def __init__(self, input_dim=256, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.mlp(x)


def create_adapter(adapter_type='mlp', input_dim=256, hidden_dim=256, num_blocks=2, num_layers=1):
    """
    創建指定類型的 Adapter 模型
    
    Args:
        adapter_type: adapter類型 ('mlp', 'lstm', 'residual')
        input_dim: 輸入維度
        hidden_dim: 隱藏層維度
        num_blocks: 殘差塊數量（僅用於 residual 類型）
        num_layers: LSTM層數（僅用於 lstm 類型）
    
    Returns:
        Adapter 模型實例
    """
    adapter_type = adapter_type.lower()
    
    if adapter_type == 'mlp':
        return MLPAdapter(input_dim=input_dim, hidden_dim=hidden_dim)
    elif adapter_type == 'lstm':
        return LSTMAdapter(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    elif adapter_type == 'residual':
        return Adapter(dim=input_dim, num_blocks=num_blocks)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}. Choose from 'mlp', 'lstm', 'residual'")
