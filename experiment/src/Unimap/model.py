import torch
import torch.nn as nn

class PaperLSTM(nn.Module):
    def __init__(self, input_dim=64, use_embedding=True, vocab_size=682):
        super().__init__()
        self.use_embedding = use_embedding
        if use_embedding:
            self.embedding = nn.Embedding(vocab_size, input_dim)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=192, num_layers=2, 
                            batch_first=True, dropout=0.2, bidirectional=True)
        self.fc = nn.Linear(384, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.use_embedding:
            x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.sigmoid(self.fc(out))
