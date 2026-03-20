import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(
        self,
        char_size,
        d_model=512,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()

        self.char_size = char_size
        self.d_model = d_model
        self.dropout = dropout

        self.linear = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, char_size),
        )

    def forward(self, x):
        return self.linear(x)