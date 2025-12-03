import torch
import torch.nn as nn
from src.entity.config_entity import EncoderConfig
from src.modules.positional_encoder import PositionalEncoding
from src.modules.encoder_cnn import EncoderCNN
from src.utils.logging_setup import logger

class CNNTransformerEncoder(nn.Module):
    """
    End-to-End Encoder: Image -> CNN -> Projection -> Positional Encoding -> Transformer
    """
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        # 1. CNN Backbone
        # We enforce a 7x7 grid to keep sequence length reasonable (49 tokens)
        self.cnn = EncoderCNN(
            model_name=config.cnn_model_name,
            fixed_spatial_size=(7, 7)
        )

        # 2. Projection Layer
        # Projects CNN feature space (e.g., 2048) to Transformer embedding space (e.g., 512)
        # We ALWAYS project. It creates a learnable bridge between visual and latent space.
        self.projection = nn.Sequential(
            nn.Linear(self.cnn.out_channels, config.embed_dim),
            nn.LayerNorm(config.embed_dim),
            # nn.ReLU(inplace=True),
            nn.Dropout(config.dropout)
        )

        # 3. Positional Encoding
        # Max len = 7*7 = 49. We set it slightly higher for safety.
        self.pos_encoder = PositionalEncoding(
            d_model=config.embed_dim,
            max_len=100,
            dropout=config.dropout
        )

        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="relu",
            batch_first=True, # Critical: matches (Batch, Seq, Dim) format
            norm_first=True   # Pre-Norm is generally more stable for deeper transformers
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_transformer_layers)

        # 5. Fine-tuning Controls
        self._set_finetuning()

    def _set_finetuning(self):
        # Freeze/Unfreeze CNN
        for param in self.cnn.parameters():
            param.requires_grad = self.config.fine_tune_cnn

        # Freeze/Unfreeze Transformer (usually always True, but good option to have)
        for param in self.transformer.parameters():
            param.requires_grad = self.config.fine_tune_transformer

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (Batch, 3, H, W)
        Returns:
            encoded_features: (Batch, Seq_Len, Embed_Dim)
        """
        # 1. Get Visual Features (Batch, 49, CNN_Dim)
        features = self.cnn(images)

        # 2. Project to Embedding Dim (Batch, 49, Embed_Dim)
        features = self.projection(features)

        # 3. Add Spatial Info (Positional Encoding)
        features = self.pos_encoder(features)

        # 4. Contextualize with Transformer
        # Output: (Batch, 49, Embed_Dim)
        encoded_output = self.transformer(features)

        return encoded_output