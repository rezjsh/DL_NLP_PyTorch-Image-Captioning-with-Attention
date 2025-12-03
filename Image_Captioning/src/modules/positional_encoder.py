import torch
import torch.nn as nn
import math
from src.utils.logging_setup import logger

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.
    Inputs are expected to be (Batch, Seq_Len, Dim).
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant 'pe' matrix with values dependent on pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # register_buffer saves this to the state_dict but it is not a learnable parameter
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, Dim)
        """
        # Slice the PE matrix to the current sequence length
        # self.pe is (1, max_len, dim), x is (B, seq_len, dim)
        # Broadcasting automatically handles the batch dimension
        x = x + self.pe[:, :x.size(1), :].detach()
        return self.dropout(x)