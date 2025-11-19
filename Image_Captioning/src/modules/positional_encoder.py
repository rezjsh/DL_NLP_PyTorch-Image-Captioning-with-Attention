import torch
import torch.nn as nn
import math
from src.utils.logging_setup import logger


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding from "Attention Is All You Need".
    Adds information about the position of the tokens in the sequence to the embeddings.
    The positional encodings are pre-computed up to `max_len`. If an input sequence
    is longer than `max_len`, a runtime error will occur, so `max_len` should be
    chosen to cover the maximum expected sequence length.
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        """
        Initializes the PositionalEncoding layer.

        Args:
            d_model (int): The dimension of the model's embeddings.
            max_len (int): The maximum length of sequences this positional encoding
                           can handle. Positional encodings are pre-computed up to this length.
            dropout (float): Dropout probability to be applied to the output.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        pe = torch.zeros(1, max_len, d_model)

        # Apply formula for sinusoidal positional encoding:
        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        # Register as buffer so it's part of the model's state but not a learnable parameter.
        # It will be moved to the correct device with the model.
        self.register_buffer('pe', pe)

        logger.info(f"PositionalEncoding initialized with d_model={d_model} and max_len={max_len}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor, typically token embeddings,
                              shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Tensor with positional encodings added,
                          shape (batch_size, seq_len, d_model).
        """
        # Add positional encoding to the input.
        # Ensure that the sequence length of x does not exceed max_len.
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            logger.error(f"Input sequence length ({seq_len}) exceeds pre-computed max_len ({self.pe.size(1)}) for positional encoding.")
            raise ValueError(f"Sequence length ({seq_len}) exceeds max_len ({self.pe.size(1)}) in PositionalEncoding.")

        x = x + self.pe[:, :seq_len]
        return self.dropout(x)