import os
import torch
import pandas as pd
from PIL import Image
from typing import Optional, Tuple
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from src.utils.logging_setup import logger
from src.entity.config_entity import CaptioningDatasetConfig
from src.components.text_preprocessing import TextPreprocessor
from src.components.image_preprocessing import ImagePreprocessor

class CaptioningDataset(Dataset):
    """
    PyTorch Dataset for loading image-caption pairs for a specific split (train/val/test).
    It correctly handles the Flickr8k structure of a single caption file and separate split files.
    """
    def __init__(self, config: CaptioningDatasetConfig, text_preprocessor: TextPreprocessor, image_preprocessor: ImagePreprocessor, split: str) -> None:
        """
        Args:
            config (CaptioningDatasetConfig): Configuration object for the dataset (must contain split file paths).
            text_preprocessor (TextPreprocessor): An initialized TextPreprocessor instance.
            image_preprocessor (ImagePreprocessor): An initialized ImagePreprocessor instance.
            split (str): The dataset split to load ('train', 'dev', or 'test').
        """
        self.config = config
        self.vocab = text_preprocessor
        self.image_preprocessor = image_preprocessor
        self.split = split

        # Determine the correct image ID file path based on the split
        if split == 'train':
            split_file = os.path.join(self.config.dataset_base_dir, config.train_images_file)
        elif split == 'dev':
            split_file = os.path.join(self.config.dataset_base_dir, config.dev_images_file)
        elif split == 'test':
            split_file = os.path.join(self.config.dataset_base_dir, config.test_images_file)
        else:
            logger.error(f"Invalid split value: {split}. Must be 'train', 'dev', or 'test'.")
            raise ValueError(f"Invalid split value: {split}")

        # 1. Load the list of image IDs for the current split
        try:
            with open(split_file, 'r') as f:
                # Read all image file names (e.g., 1000268201_693b08cb0e.jpg)
                self.split_image_ids = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(self.split_image_ids)} image IDs for the '{split}' split from {split_file}.")
        except FileNotFoundError:
            logger.error(f"Split file not found: {split_file}. Exiting.")
            raise FileNotFoundError(f"Split file not found: {split_file}")

        # 2. Load the master captions file (which contains all image IDs and captions)
        try:
            # The Flickr8k captions file is tokenized with image ID and caption separated by a tab.
            all_captions_df = pd.read_csv(self.config.caption_path, delimiter="\t", names=["image_caption_index", "caption"])

            # The 'image_caption_index' column contains 'image_name#index' (e.g., '1000268201_693b08cb0e.jpg#0')
            all_captions_df['image_id'] = all_captions_df['image_caption_index'].apply(lambda x: x.split('#')[0])

            logger.info(f"Successfully loaded all captions from: {self.config.caption_path}. Total entries: {len(all_captions_df)}")
        except FileNotFoundError:
            logger.error(f"Master captions file not found: {self.config.caption_path}. Exiting.")
            raise

        # 3. Filter the master captions DataFrame to only include images in the current split
        self.df = all_captions_df[all_captions_df['image_id'].isin(self.split_image_ids)].reset_index(drop=True)

        logger.info(f"Final {split} dataset size (image-caption pairs): {len(self.df)}")
        if len(self.df) == 0:
            logger.warning(f"The filtered DataFrame for '{split}' is empty. Check your split files and master caption file.")


    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Loads and preprocesses one sample (image and caption).
        Returns:
            A tuple of (image_tensor, numericalized_caption_tensor) or None on error.
        """
        # Get image ID and caption text
        sample = self.df.iloc[index]
        img_id = sample['image_id'] # Use the cleaned image_id column
        caption_text = sample['caption']

        img_path = os.path.join(self.config.images_dir, img_id)

        # Open image
        try:
            pil_image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            logger.error(f"Image file not found for index {index}: {img_path}. Returning None for this sample.")
            return None
        except Exception as e:
            logger.error(f"Error loading image for index {index} ({img_path}): {e}. Returning None for this sample.", exc_info=True)
            return None

        # Preprocess image using ImagePreprocessor
        image_tensor = None
        if self.image_preprocessor:
            try:
                # The image_preprocessor should have the transformation logic
                image_tensor = self.image_preprocessor.preprocess_image(pil_image)
            except Exception as e:
                logger.error(f"Error during image preprocessing for {img_path}: {e}. Returning None for this sample.", exc_info=True)
                return None
        else:
            # Fallback (shouldn't happen in a proper pipeline)
            logger.warning(f"No image preprocessor provided. Falling back to basic ToTensor for {img_path}.")
            image_tensor = transforms.ToTensor()(pil_image)

        # Numericalize caption using vocab
        # 1. Tokenize and numericalize the main body of the caption
        numericalized_caption = self.vocab.numericalize(caption_text)

        # 2. Add <SOS> and <EOS> tokens
        numericalized_caption.insert(0, self.vocab.stoi["<SOS>"])
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        # Convert list of integers to PyTorch tensor
        caption_tensor = torch.tensor(numericalized_caption, dtype=torch.long)

        return image_tensor, caption_tensor