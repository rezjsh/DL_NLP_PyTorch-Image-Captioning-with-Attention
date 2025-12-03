import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional
from src.utils.logging_setup import logger

class EncoderCNN(nn.Module):
    """
    CNN Backbone to extract spatial features.
    Includes Adaptive Pooling to ensure fixed sequence lengths for the Transformer.
    """
    def __init__(self, model_name: str, fixed_spatial_size: Optional[Tuple[int, int]] = (7, 7)):
        super(EncoderCNN, self).__init__()

        # 1. Load Backbone
        self.backbone, self.out_channels = self._get_backbone(model_name)

        # 2. Adaptive Pooling
        # This enforces a specific output grid (e.g., 7x7) regardless of input image size.
        # This is vital for Transformer memory stability.
        self.avg_pool = nn.AdaptiveAvgPool2d(fixed_spatial_size) if fixed_spatial_size else nn.Identity()

        self.output_spatial_size = fixed_spatial_size

        logger.info(f"EncoderCNN initialized: {model_name}, Output Channels: {self.out_channels}, Grid: {fixed_spatial_size}")

    def _get_backbone(self, model_name: str):
        """Helper to load model and remove classification heads."""
        base_models = {
            "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
            "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
            "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
            "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT),
            "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
            "efficientnet_b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT),
        }

        if model_name not in base_models:
            raise ValueError(f"Model {model_name} not supported. Choose from {list(base_models.keys())}")

        model_fn, weights = base_models[model_name]
        raw_model = model_fn(weights=weights)

        if "resnet" in model_name:
            # Remove AvgPool and FC layer
            modules = list(raw_model.children())[:-2]
            backbone = nn.Sequential(*modules)
            out_channels = raw_model.fc.in_features
        elif "efficientnet" in model_name:
            # EfficientNet .features returns the feature maps directly
            backbone = raw_model.features
            # Dummy pass to find out_channels
            with torch.no_grad():
                dummy = backbone(torch.randn(1, 3, 224, 224))
            out_channels = dummy.shape[1]
        else:
            raise NotImplementedError(f"Logic for {model_name} not implemented yet.")

        return backbone, out_channels

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            Tensor of shape (Batch, Seq_Len, Channels)
            where Seq_Len = H * W
        """
        # 1. Extract Features -> (B, C, H_raw, W_raw)
        features = self.backbone(images)

        # 2. Enforce Fixed Grid -> (B, C, 7, 7)
        features = self.avg_pool(features)

        # 3. Flatten Spatial Dimensions -> (B, C, 49)
        features = features.flatten(2)

        # 4. Permute for Transformer -> (B, 49, C)
        features = features.permute(0, 2, 1)

        return features