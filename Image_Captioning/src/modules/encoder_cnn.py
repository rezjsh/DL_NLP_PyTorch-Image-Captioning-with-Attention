from src.utils.logging_setup import logger
import torch.nn as nn
import torchvision.models as models
import torch


class EncoderCNN(nn.Module):
    """
    CNN Encoder to extract features from input images using various pre-trained backbones.
    Removes the final classification layer.
    """
    def __init__(self, model_name: str) -> None:
        """
        Initializes the CNN encoder by loading a pre-trained model based on model_name.

        Args:
            model_name (str): The name of the pre-trained model to load (e.g., "resnet50", "vgg16", "efficientnet_b0").
        """
        super(EncoderCNN, self).__init__()
        # Dictionary to map model names to their respective loading functions, weights,
        # and a lambda function to extract the feature backbone.
        self.model_configs = {
            # ResNets: Remove the last two layers (AvgPool and FC)
            "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, lambda m: nn.Sequential(*list(m.children())[:-2])),
            "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT, lambda m: nn.Sequential(*list(m.children())[:-2])),
            "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, lambda m: nn.Sequential(*list(m.children())[:-2])),
            "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT, lambda m: nn.Sequential(*list(m.children())[:-2])),
            "resnet152": (models.resnet152, models.ResNet152_Weights.DEFAULT, lambda m: nn.Sequential(*list(m.children())[:-2])),
            
            # VGGs: The 'features' module contains the convolutional layers
            "vgg16": (models.vgg16, models.VGG16_Weights.DEFAULT, lambda m: m.features),
            "vgg19": (models.vgg19, models.VGG19_Weights.DEFAULT, lambda m: m.features),
            
            # DenseNets: The 'features' module contains the convolutional layers
            "densenet121": (models.densenet121, models.DenseNet121_Weights.DEFAULT, lambda m: m.features),
            "densenet161": (models.densenet161, models.DenseNet161_Weights.DEFAULT, lambda m: m.features),
            
            # EfficientNets: The 'features' module contains the convolutional layers
            "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, lambda m: m.features),
            "efficientnet_b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT, lambda m: m.features),
            "efficientnet_b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT, lambda m: m.features),
            "efficientnet_b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT, lambda m: m.features),
            "efficientnet_b4": (models.efficientnet_b4, models.EfficientNet_B4_Weights.DEFAULT, lambda m: m.features),
            "efficientnet_b5": (models.efficientnet_b5, models.EfficientNet_B5_Weights.DEFAULT, lambda m: m.features),
            "efficientnet_b6": (models.efficientnet_b6, models.EfficientNet_B6_Weights.DEFAULT, lambda m: m.features),
            "efficientnet_b7": (models.efficientnet_b7, models.EfficientNet_B7_Weights.DEFAULT, lambda m: m.features),
            
            # MobileNets: The 'features' module contains the convolutional layers
            "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT, lambda m: m.features),
            "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT, lambda m: m.features),
            "mobilenet_v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT, lambda m: m.features),

            # ConvNeXts: The 'features' module contains the convolutional layers
            "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT, lambda m: m.features),
            "convnext_small": (models.convnext_small, models.ConvNeXt_Small_Weights.DEFAULT, lambda m: m.features),
            "convnext_base": (models.convnext_base, models.ConvNeXt_Base_Weights.DEFAULT, lambda m: m.features),
            "convnext_large": (models.convnext_large, models.ConvNeXt_Large_Weights.DEFAULT, lambda m: m.features),
        }

        if model_name not in self.model_configs:
            raise ValueError(f"Unsupported model name: {model_name}. "
                             f"Supported models are: {list(self.model_configs.keys())}")

        model_loader, model_weights, feature_extractor_fn = self.model_configs[model_name]
        
        # Load the pre-trained full model
        full_model = model_loader(weights=model_weights)
        
        # Extract the feature-only backbone using the specific function for this model type
        self.cnn_backbone = feature_extractor_fn(full_model)

        # Dynamically determine the output feature dimension and spatial size
        # Use a standard input size for classification models
        # Most models output 7x7 spatial features for 224x224 input due to downsampling by 32
        dummy_input = torch.randn(1, 3, 224, 224) 
        
        with torch.no_grad():
            dummy_output = self.cnn_backbone(dummy_input)

        if dummy_output.dim() == 4: # Output is (batch_size, channels, H, W)
            self.encoder_dim = dummy_output.size(1) # Channels
            self.output_spatial_size = (dummy_output.size(2), dummy_output.size(3))
        elif dummy_output.dim() == 2: # Output is (batch_size, features) - already pooled and flattened
            self.encoder_dim = dummy_output.size(1)
            self.output_spatial_size = (1, 1) # Treat as 1x1 spatial for consistency with attention
            logger.warning(f"Model {model_name} backbone output is already globally pooled "
                           f"to shape {dummy_output.shape}. This means num_pixels will be 1. "
                           "Ensure this is intended for your Transformer's attention mechanism.")
        else:
            raise RuntimeError(f"Unexpected output dimensions from {model_name} backbone: {dummy_output.shape}")

        logger.info(f"EncoderCNN initialized with {model_name} backbone. "
                    f"CNN output dimension (encoder_dim): {self.encoder_dim}. "
                    f"Output spatial size: {self.output_spatial_size}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.
        Args:
            images (torch.Tensor): Input image batch (batch_size, 3, H, W).
        Returns:
            torch.Tensor: Extracted image features (batch_size, num_pixels, encoder_dim).
        """
        features = self.cnn_backbone(images)
        
        if features.dim() == 2: # If backbone already returned flattened (B, C)
            # Unsqueeze to (B, 1, C) to fit (batch_size, num_pixels, encoder_dim)
            features = features.unsqueeze(1) 
        elif features.dim() == 4:
            # Permute from (batch_size, channels, H, W) to (batch_size, H, W, channels)
            features = features.permute(0, 2, 3, 1) 
            # Reshape to (batch_size, num_pixels, encoder_dim)
            features = features.contiguous().view(features.size(0), -1, features.size(3)) 
        else:
            raise RuntimeError(f"Unexpected feature dimensions after CNN backbone: {features.shape}. "
                               "Expected 2D (B, C) or 4D (B, C, H, W).")
        
        return features # Shape: (batch_size, num_pixels, encoder_dim)
    
