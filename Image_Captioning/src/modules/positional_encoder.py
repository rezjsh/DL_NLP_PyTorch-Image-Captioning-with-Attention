import torch
import torch.nn as nn
import math
from src.utils.logging_setup import logger


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding from "Attention Is All You Need".
    Adds information about the position of the tokens in the sequence.
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        pe = torch.zeros(1, max_len, d_model)
        # Apply formula for sinusoidal positional encoding
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # Register as buffer so it's not a model parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape (batch_size, seq_len, d_model)
        Returns:
            Tensor, shape (batch_size, seq_len, d_model) with positional encoding added.
        """
        # Add positional encoding to the input. Truncate if input sequence is shorter than max_len.
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return self.dropout(x)
