import torch
import torch.nn as nn

class AdapterMapper(nn.Module):
    """
    這就是論文 中的映射 F (Mapping F)。
    它學習將一個向量空間 (input_dim) 映射到另一個 (output_dim)。
    """
    def __init__(self, input_dim, output_dim, hidden_dim=1024):
        super().__init__()
        
        # 您可以使這個網路更深或更淺
        # 這實質上就是一個 bottleneck/ResNet 的簡化版
        self.mapper = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # 為了穩定性，可以加入 LayerNorm
            nn.LayerNorm(hidden_dim), 
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 論文 提到 F 是 x 和 f(x, V) 的組合
        # (即 x + f(x))。
        # 如果 input_dim == output_dim，您可以實作一個殘差連接 (residual connection)
        # 讓 Adapter 只學習 "偏差" (bias)，這有助於訓練穩定
        self.use_residual = (input_dim == output_dim)

    def forward(self, x):
        if self.use_residual:
            # F(x) = x + f(x)
            bias = self.mapper(x)
            return x + bias
        else:
            # F(x) = f(x)
            # 如果維度不同，則直接映射
            return self.mapper(x)