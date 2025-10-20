# src/components/encoder.py
import torch
import torch.nn as nn
from src.entity.config_entity import EncoderConfig
from src.modules.positional_encoder import PositionalEncoding
from src.modules.encoder_cnn import EncoderCNN
from src.utils.logging_setup import logger

class CNNTransformerEncoder(nn.Module):
    """
    A combined CNN and Transformer Encoder for image feature extraction.
    The CNN extracts spatial features, which are then processed by a Transformer encoder.
    Supports fine-tuning specific parts of the model.
    """
    def __init__(self, config: EncoderConfig) -> None:
        """
        Initializes the CNN-Transformer encoder.

        Args:
            config (EncoderConfig): Configuration for the encoder.
        """
        super(CNNTransformerEncoder, self).__init__()
        self.config = config
        self.cnn_encoder = EncoderCNN(model_name=self.config.cnn_model_name)

        # Ensure embed_dim matches the CNN's output feature dimension or add a projection layer
        cnn_output_dim = self.cnn_encoder.encoder_dim
        if cnn_output_dim != self.config.embed_dim:
            logger.warning(f"Mismatch between CNN output dim ({cnn_output_dim}) "
                           f"and requested embed_dim ({self.config.embed_dim}). Adding a linear projection layer.")
            self.feature_projection = nn.Linear(cnn_output_dim, self.config.embed_dim)
            self.embed_dim = self.config.embed_dim
        else:
            self.feature_projection = None
            self.embed_dim = cnn_output_dim

        # Set requires_grad for CNN parameters based on fine_tune_cnn flag
        for param in self.cnn_encoder.parameters():
            param.requires_grad = self.config.fine_tune_cnn
        logger.info(f"CNN backbone fine-tuning set to: {self.config.fine_tune_cnn}")

        # Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, # Use the actual embed_dim after potential projection
            nhead=self.config.num_heads,
            dim_feedforward=self.config.ff_dim,
            dropout=self.config.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.config.num_transformer_layers)

        # Calculate the number of pixels for positional encoding max_len
        # If spatial output is (H, W), num_pixels = H * W
        num_pixels = 1 # Default for flattened output
        if len(self.cnn_encoder.output_spatial_size) == 2:
             num_pixels = self.cnn_encoder.output_spatial_size[0] * self.cnn_encoder.output_spatial_size[1]
        # max_len for positional encoding should be the number of positions in the sequence
        # This is num_pixels (for spatial features) + 1 (if using a CLS token, which is common)
        # Assuming we will use a CLS token for simplicity, adjust if not needed.
        self.positional_encoding = PositionalEncoding(self.embed_dim, max_len=num_pixels + 1) # +1 for potential CLS token

        self.dropout = nn.Dropout(self.config.dropout)

        # Set requires_grad for Transformer parameters based on fine_tune_transformer flag
        for param in self.transformer_encoder.parameters():
            param.requires_grad = self.config.fine_tune_transformer
        logger.info(f"Transformer encoder fine-tuning set to: {self.config.fine_tune_transformer}")

        logger.info(f"CNNTransformerEncoder initialized. Embed Dim: {self.embed_dim}, "
                    f"Transformer Layers: {self.config.num_transformer_layers}, "
                    f"Heads: {self.config.num_heads}, FF Dim: {self.config.ff_dim}, "
                    f"Positional Encoding Max Len: {num_pixels + 1}")


    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the combined CNN and Transformer encoder.
        Args:
            images (torch.Tensor): Input image batch (batch_size, 3, H, W).
        Returns:
            torch.Tensor: Encoded image features (batch_size, num_pixels, embed_dim).
        """
        # 1. CNN Feature Extraction
        # Output: (batch_size, num_pixels, cnn_output_dim) - reshaped inside EncoderCNN
        cnn_features = self.cnn_encoder(images)

        # 2. Optional Linear Projection
        if self.feature_projection:
            cnn_features = self.feature_projection(cnn_features)

        # 3. Add Positional Encoding
        # Positional encoding expects (batch_size, seq_len, d_model)
        # cnn_features is already (batch_size, num_pixels, embed_dim)
        # Add a CLS token if required by the model architecture (not explicitly in this code, but common pattern)
        # If a CLS token is needed, you'd prepend it here before PE.
        # For now, assuming PE is applied directly to flattened spatial features.
        features_with_pe = self.positional_encoding(cnn_features)
        features_with_pe = self.dropout(features_with_pe)

        # 4. Pass through Transformer Encoder
        # TransformerEncoder expects (src, src_mask)
        # Here src is features_with_pe
        encoder_output = self.transformer_encoder(features_with_pe)

        return encoder_output