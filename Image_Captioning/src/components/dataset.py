import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from src.utils.logging_setup import logger
from src.entity.config_entity import CaptioningDatasetConfig
import torchvision.transforms as transforms
from src.components.image_preprocessing import ImagePreprocessor
from src.components.text_preprocessing import TextPreprocessor
from typing import Optional, Tuple

class CaptioningDataset(Dataset):
    """
    PyTorch Dataset for loading image-caption pairs for a specific split (train/val/test).
    It correctly handles the Flickr8k structure of a single caption file and separate split files.
    """
    def __init__(self, config: CaptioningDatasetConfig, text_preprocessor: TextPreprocessor, image_preprocessor: ImagePreprocessor, split: str) -> None:
        """
        Args:
            config (CaptioningDatasetConfig): Configuration object for the dataset.
            text_preprocessor (TextPreprocessor): An initialized TextPreprocessor instance (handling vocabulary and numericalization).
            image_preprocessor (ImagePreprocessor): An initialized ImagePreprocessor instance (handling image transformations).
            split (str): The dataset split to load ('train', 'dev', or 'test').
        """
        self.config = config
        self.text_preprocessor = text_preprocessor  # Renamed self.vocab for clarity
        self.image_preprocessor = image_preprocessor
        self.split = split
        self.df: pd.DataFrame = pd.DataFrame() # Initialize df to silence potential IDE warnings

        # 1. Determine the correct image ID file path based on the split
        split_map = {
            'train': config.train_images_file,
            'dev': config.dev_images_file,
            'test': config.test_images_file,
        }

        if split not in split_map:
            logger.error(f"Invalid split value: {split}. Must be 'train', 'dev', or 'test'.")
            raise ValueError(f"Invalid split value: {split}")

        split_file = os.path.join(self.config.dataset_base_dir, split_map[split])

        # 2. Load the list of image IDs for the current split
        try:
            with open(split_file, 'r') as f:
                self.split_image_ids = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(self.split_image_ids)} image IDs for the '{split}' split from {split_file}.")
        except FileNotFoundError:
            logger.error(f"Split file not found: {split_file}. Exiting.")
            raise

        # 3. Load the master captions file
        try:
            # Using header=None and names directly for cleaner column assignment
            all_captions_df = pd.read_csv(
                config.caption_path,
                delimiter="\t",
                header=None,
                names=["image_caption_index", "caption"],
                encoding='utf-8' # Explicit encoding for robustness
            )

            # Extract the actual image ID (filename) from 'image_name#index'
            all_captions_df['image_id'] = all_captions_df['image_caption_index'].str.split('#').str[0]

            logger.info(f"Successfully loaded all captions from: {config.caption_path}. Total entries: {len(all_captions_df)}")
        except FileNotFoundError:
            logger.error(f"Master captions file not found: {config.caption_path}. Exiting.")
            raise

        # 4. Filter the master captions DataFrame
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
        # Retrieve image ID and caption text directly
        try:
            img_id = self.df.loc[index, 'image_id']
            caption_text = self.df.loc[index, 'caption']
        except KeyError:
             logger.error(f"DataFrame error: Could not retrieve sample at index {index}.")
             return None

        img_path = os.path.join(self.config.images_dir, img_id)

        # 1. Image Loading and Preprocessing
        try:
            # Context manager for PIL Image handles resource cleanup
            with Image.open(img_path) as pil_image:
                # Ensure 3 color channels
                pil_image = pil_image.convert("RGB")

                if self.image_preprocessor:
                    image_tensor = self.image_preprocessor.preprocess_image(pil_image)
                else:
                    logger.warning(f"No image preprocessor provided. Falling back to basic ToTensor for {img_id}.")
                    image_tensor = transforms.ToTensor()(pil_image)

        except FileNotFoundError:
            logger.error(f"Image file not found for index {index}: {img_path}. Skipping this sample.")
            return None
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {e}", exc_info=True)
            return None


        # 2. Caption Numericalization
        try:
            # 1. Tokenize and numericalize
            numericalized_caption = self.text_preprocessor.numericalize(caption_text)

            # 2. Add <SOS> and <EOS> tokens using the text preprocessor's vocabulary
            numericalized_caption.insert(0, self.text_preprocessor.stoi["<SOS>"])
            numericalized_caption.append(self.text_preprocessor.stoi["<EOS>"])

            # 3. Convert list of integers to PyTorch tensor
            caption_tensor = torch.tensor(numericalized_caption, dtype=torch.long)
        except Exception as e:
            logger.error(f"Error during caption numericalization for index {index} ('{caption_text}'): {e}", exc_info=True)
            return None

        return image_tensor, caption_tensor




# # Example test usage
# if __name__ == "__main__":
    # class DummyConfig:
    #     dataset_base_dir= "data/"
    #     train_images_file= "Flickr_8k.trainImages.txt"
    #     dev_images_file= "Flickr_8k.devImages.txt"
    #     test_images_file= "Flickr_8k.testImages.txt"
    #     images_dir= "data/Flicker8k_Dataset"
    #     caption_path= "data/captions.txt"

    # # Mock preprocessors with dummy passthrough for test
    # class DummyImagePreprocessor:
    #     def preprocess_image(self, pil_image):
    #         return transforms.ToTensor()(pil_image)

    # class DummyTextPreprocessor:
    #     stoi = {"<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    #     def numericalize(self, text):
    #         return [1, 2, 3]  # dummy numericalization

    # config = DummyConfig()
    # dataset = CaptioningDataset(config, DummyTextPreprocessor(), DummyImagePreprocessor(), 'train')

    # print(f"Dataset length: {len(dataset)}")
    # sample = dataset[23]
    # if sample:
    #     img_tensor, cap_tensor = sample
    #     print(f"Image tensor shape: {img_tensor.shape}")
    #     print(f"Caption tensor: {cap_tensor}")
    # else:
    #     print("Sample is None (error in loading)")