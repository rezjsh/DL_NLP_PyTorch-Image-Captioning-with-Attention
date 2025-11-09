import torchvision.transforms as transforms
from PIL import Image
import torch
from src.utils.logging_setup import logger
from src.entity.config_entity import ImagePreprocessingConfig

class ImagePreprocessor:
    """
    A class to encapsulate image preprocessing transformations using torchvision.transforms.
    """
    def __init__(self, config: ImagePreprocessingConfig) -> None:
        """
        Initializes the ImagePreprocessor by building a torchvision transform pipeline.
        Args:
            config (ImagePreprocessingConfig): An instance of ImagePreprocessingConfig containing image preprocessing settings.
        """
        self.config = config

        self.transform_pipeline = self._build_transform_pipeline()
        logger.info("ImagePreprocessor initialized and transform pipeline built.")
        logger.info(f"Image preprocessing settings: {self.config}")

    def _build_transform_pipeline(self) -> transforms.Compose:
        """
        Builds the torchvision.transforms.Compose pipeline.
        """
        transforms_list = []
        # Apply the specified image transformations
        # Resize
        if self.config.resize_size:
            transforms_list.append(transforms.Resize(tuple(self.config.resize_size)))
            logger.debug(f"Added Resize: {self.config.resize_size}")

        # RandomCrop
        if self.config.random_crop_size:
            transforms_list.append(transforms.RandomCrop(tuple(self.config.random_crop_size)))
            logger.debug(f"Added RandomCrop: {self.config.random_crop_size}")

        # ToTensor
        transforms_list.append(transforms.ToTensor())
        logger.debug("Added ToTensor")

        # Normalize
        if self.config.normalize_mean and self.config.normalize_std:
            if len(self.config.normalize_mean) == 3 and len(self.config.normalize_std) == 3:  # Ensure RGB
                transforms_list.append(transforms.Normalize(mean=self.config.normalize_mean, std=self.config.normalize_std))
                logger.debug(f"Added Normalize: mean={self.config.normalize_mean}, std={self.config.normalize_std}")
            else:
                logger.warning("Normalize mean/std are not 3-element lists. Skipping normalization.")
        
        return transforms.Compose(transforms_list)


    def preprocess_image(self, pil_image: Image.Image) -> torch.Tensor:
        """
        Applies the defined transform pipeline to a PIL Image.
        Args:
            pil_image (PIL.Image.Image): The input PIL Image.
        Returns:
            torch.Tensor: The transformed image tensor.
        Raises:
            TypeError: If the input is not a PIL Image.
        """
        if not isinstance(pil_image, Image.Image):
            logger.error(f"Input is not a PIL Image. Got: {type(pil_image)}")
            raise TypeError("Input to preprocess_image must be a PIL Image.")
        
        try:
            transformed_tensor = self.transform_pipeline(pil_image)
            logger.debug(f"Image transformed to tensor of shape: {transformed_tensor.shape}")
            return transformed_tensor
        except Exception as e:
            logger.error(f"Error applying image transformations: {e}", exc_info=True)
            raise

