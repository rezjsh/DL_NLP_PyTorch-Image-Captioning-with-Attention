import os
import requests
import zipfile
import shutil
from src.entity.config_entity import DataIngestionConfig
from src.utils.logging_setup import logger
from src.utils.helpers import download_file, extract_zip
class DataIngestion:
    """
    A utility class to download and extract the dataset.
    This includes both the image files and the captions file.
    """

    def __init__(self, config: DataIngestionConfig):
        """
        Initializes the downloader.
        Args:
            config (DataIngestionConfig): Configuration object for data ingestion.
        """
        self.config = config

    def download_dataset(self):
        """
        Downloads and extracts the dataset into the specified download directory.
        """
        # --- Download Images ---
        images_zip_path = os.path.join(self.config.download_dir, self.config.images_zip_name)
        if not os.path.exists(images_zip_path):
            if not download_file(self.config.images_zip_url, images_zip_path):
                logger.error("Failed to download image dataset. Exiting.")
                return
        else:
            logger.info(f"Image zip file already exists: {images_zip_path}")

        # --- Download Captions ---
        captions_zip_path = os.path.join(self.config.download_dir, self.config.captions_zip_name)
        if not os.path.exists(captions_zip_path):
            if not download_file(self.config.captions_url, captions_zip_path):
                logger.error("Failed to download captions dataset. Exiting.")
                return
        else:
            logger.info(f"Captions zip file already exists: {captions_zip_path}")

        # --- Extract Images ---
        images_extract_path = os.path.join(self.config.download_dir, "images")
        if not os.path.exists(images_extract_path):
            if not extract_zip(images_zip_path, images_extract_path):
                logger.error("Failed to extract image dataset. Exiting.")
                return
        else:
            logger.warning(f"Image directory already exists: {images_extract_path}")

        # --- Extract and Process Captions ---
        captions_temp_extract_path = os.path.join(self.config.download_dir, "temp_captions")
        if not os.path.exists(captions_temp_extract_path):
            if not extract_zip(captions_zip_path, captions_temp_extract_path):
                logger.error("Failed to extract captions dataset. Exiting.")
                return

            # Find the actual captions file
            found_caption_file = None
            for root, _, files in os.walk(captions_temp_extract_path):
                if self.config.captions_file_name in files:
                    found_caption_file = os.path.join(root, self.config.captions_file_name)
                    break
            
            if found_caption_file:
                # Move the captions file to the main download directory
                target_captions_path = os.path.join(self.config.download_dir, "captions.txt")
                shutil.move(found_caption_file, target_captions_path)
                logger.info(f"Moved captions file to: {target_captions_path}")
            else:
                logger.warning(f"Could not find '{self.config.captions_file_name}' inside the captions zip.")

            # Clean up temporary captions extraction directory
            shutil.rmtree(captions_temp_extract_path)
            logger.info(f"Cleaned up temporary captions directory: {captions_temp_extract_path}")
        else:
            logger.warning(f"Captions directory already exists (or captions.txt already moved).")


        logger.info("\nFlickr8k dataset download and extraction complete!")
        logger.info(f"Images are in: {os.path.join(self.config.download_dir, 'images')}")
        logger.info(f"Captions are in: {os.path.join(self.config.download_dir, 'captions.txt')}")