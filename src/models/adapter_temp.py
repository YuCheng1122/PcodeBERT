import torch
import torch.nn as nn

class AdapterMapper(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        # self.mapper = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.LayerNorm(hidden_dim), 
        #     nn.Linear(hidden_dim, output_dim)
        # )

        self.mapper = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(), 
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, hidden_dim), 
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, output_dim)
        )   
        self.use_residual = (input_dim == output_dim)

    def forward(self, x):
        if self.use_residual:
            return x + self.mapper(x)
        else:
            return self.mapper(x)
