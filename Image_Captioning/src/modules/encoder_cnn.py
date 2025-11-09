from src.utils.logging_setup import logger
import torch.nn as nn
import torchvision.models as models
import torch


class EncoderCNN(nn.Module):
    """
    CNN Encoder to extract features from input images using various pre-trained backbones.
    Removes the final classification layer to get spatial features.

    Attributes:
        encoder_dim (int): The dimension of the output features from the CNN backbone.
        output_spatial_size (tuple): The spatial dimensions (H, W) of the feature maps.
        cnn_backbone (nn.Module): The loaded and modified pre-trained CNN model.
    """
    def __init__(self, model_name: str) -> None:
        """
        Initializes the CNN encoder by loading a pre-trained model based on model_name.

        Args:
            model_name (str): The name of the pre-trained model to load (e.g., "resnet50",
                              "vgg16", "efficientnet_b0").
        """
        super(EncoderCNN, self).__init__()

        # Dictionary to map model names to their respective loading functions, weights,
        # and a lambda function to extract the feature backbone.
        # The lambda function should typically remove the classification head.
        self.model_configs = {
            # ResNets: Remove the last two layers (AvgPool and FC)
            "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, lambda m: nn.Sequential(*list(m.children())[:-2])),
            "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT, lambda m: nn.Sequential(*list(m.children())[:-2])),
            "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, lambda m: nn.Sequential(*list(m.children())[:-2])),
            "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT, lambda m: nn.Sequential(*list(m.children())[:-2])),
            "resnet152": (models.resnet152, models.ResNet152_Weights.DEFAULT, lambda m: nn.Sequential(*list(m.children())[:-2])),

            # VGGs: Remove the classifier (last layer)
            "vgg16": (models.vgg16, models.VGG16_Weights.DEFAULT, lambda m: m.features),
            "vgg19": (models.vgg19, models.VGG19_Weights.DEFAULT, lambda m: m.features),

            # EfficientNets: Remove the classifier
            "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, lambda m: m.features),
            "efficientnet_b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT, lambda m: m.features),
            "efficientnet_b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT, lambda m: m.features),
            "efficientnet_b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT, lambda m: m.features),
            "efficientnet_b4": (models.efficientnet_b4, models.EfficientNet_B4_Weights.DEFAULT, lambda m: m.features),
            "efficientnet_b5": (models.efficientnet_b5, models.EfficientNet_B5_Weights.DEFAULT, lambda m: m.features),
        }

        if model_name not in self.model_configs:
            supported_models = ", ".join(self.model_configs.keys())
            logger.error(f"Unsupported model name: '{model_name}'. Supported models are: {supported_models}")
            raise ValueError(f"Unsupported model name: '{model_name}'. "
                             f"Please choose from: {supported_models}")

        model_func, weights, feature_extractor_lambda = self.model_configs[model_name]
        logger.info(f"Loading pre-trained {model_name} model with {weights.name} weights.")
        full_model = model_func(weights=weights)
        self.cnn_backbone = feature_extractor_lambda(full_model)

        # Determine output dimensions by passing a dummy input
        # Assuming typical image input size of (1, 3, 224, 224)
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            dummy_output = self.cnn_backbone(dummy_input)

        if dummy_output.dim() == 4: # (B, C, H_out, W_out)
            self.encoder_dim = dummy_output.size(1)  # C
            self.output_spatial_size = (dummy_output.size(2), dummy_output.size(3)) # H_out, W_out
            logger.debug(f"CNN backbone {model_name} output feature map shape: {dummy_output.shape}")
        elif dummy_output.dim() == 2: # (B, C) - already flattened by the backbone
            self.encoder_dim = dummy_output.size(1) # C
            self.output_spatial_size = (1, 1) # Represents a single "pixel" for flattened output
            logger.debug(f"CNN backbone {model_name} output flattened features shape: {dummy_output.shape}")
        else:
            logger.error(f"Unexpected dummy output dimension for {model_name}: {dummy_output.dim()}. Expected 2 or 4.")
            raise RuntimeError(f"Unexpected output dimension from CNN backbone: {dummy_output.dim()}")

        logger.info(f"EncoderCNN initialized with {model_name} backbone. "
                    f"Output features dimension: {self.encoder_dim}, spatial size: {self.output_spatial_size}.")


    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the encoder.

        Args:
            images (torch.Tensor): Input image batch (batch_size, 3, H, W).

        Returns:
            torch.Tensor: Extracted image features (batch_size, num_pixels, encoder_dim).
                          num_pixels is H * W if the output is spatial, or 1 if flattened.
        """
        features = self.cnn_backbone(images)

        if features.dim() == 2: # If backbone already returned flattened (B, C)
            # Unsqueeze to (B, 1, C) to fit (batch_size, num_pixels, encoder_dim) format where num_pixels=1
            features = features.unsqueeze(1)
        elif features.dim() == 4: # (batch_size, channels, H, W)
            # Permute to (batch_size, H, W, channels)
            features = features.permute(0, 2, 3, 1)
            # Reshape to (batch_size, num_pixels, encoder_dim)
            # .contiguous() is good practice before view/reshape if tensor might be non-contiguous
            features = features.contiguous().view(features.size(0), -1, features.size(3))
        else:
            raise RuntimeError(f"Unexpected feature tensor dimension from CNN backbone: {features.dim()}. Expected 2 or 4.")

        return features