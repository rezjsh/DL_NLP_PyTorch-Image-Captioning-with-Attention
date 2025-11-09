# src/components/data_ingestion.py
import os
import shutil
import pandas as pd 
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

    def _download_images(self, images_zip_path=None) -> None:
        '''
        Download the images zip file if it doesn't exist.
        '''
        if images_zip_path is None:
            images_zip_path = os.path.join(self.config.download_dir, self.config.images_zip_name)

        if not os.path.exists(images_zip_path):
            if not download_file(self.config.images_zip_url, images_zip_path):
                logger.error("Failed to download image dataset. Exiting.")
                return
        else:
            logger.info(f"Image zip file already exists: {images_zip_path}")

    def _extract_images(self, images_zip_path) -> None:
        '''
        Extract the images zip file if it doesn't exist.
        '''
        images_extract_path = os.path.join(self.config.download_dir, "Flicker8k_Dataset") 
        if not os.path.exists(images_extract_path):
            logger.info(f"Extracting images from {images_zip_path} to {images_extract_path}...")
            if not extract_zip(images_zip_path, self.config.download_dir): # Extract to download_dir
                logger.error("Failed to extract image dataset. Exiting.")
                return
        else:
            logger.info(f"Images already extracted to: {images_extract_path}")

    def _download_captions(self, captions_zip_path=None) -> None:
        '''
        Download the captions zip file if it doesn't exist.
        '''
        if captions_zip_path is None:
            captions_zip_path = os.path.join(self.config.download_dir, self.config.captions_zip_name)

        if not os.path.exists(captions_zip_path):
            if not download_file(self.config.captions_url, captions_zip_path):
                logger.error("Failed to download captions dataset. Exiting.")
                return
        else:
            logger.info(f"Captions zip file already exists: {captions_zip_path}")
            

    def download_dataset(self) -> None:
        """
        Downloads and extracts the dataset into the specified download directory.
        Also processes the raw token file into a usable captions.txt.
        """
        images_zip_path = os.path.join(self.config.download_dir, self.config.images_zip_name)
        captions_zip_path = os.path.join(self.config.download_dir, self.config.captions_zip_name)
        # --- Download Images ---
        self._download_images(images_zip_path)

        # --- Download Captions ---
        self._download_captions(captions_zip_path)

        # --- Extract Images ---
        self._extract_images(images_zip_path)


        # --- Extract and Process Captions and Split Files ---
        raw_token_file_name = self.config.captions_file_name # Flickr8k.token.txt
        processed_captions_file_name = "captions.txt" # The desired output file name
        processed_captions_file_path = os.path.join(self.config.download_dir, processed_captions_file_name)

        # Check if the final processed captions file already exists
        if not os.path.exists(processed_captions_file_path):
            logger.info(f"Processed captions file not found: {processed_captions_file_path}. Processing raw token file.")

            # Create a temporary directory for extraction of captions zip
            captions_temp_extract_path = os.path.join(self.config.download_dir, "temp_captions_extract")
            os.makedirs(captions_temp_extract_path, exist_ok=True) # Ensure temp dir exists

            logger.info(f"Extracting captions from {captions_zip_path} to {captions_temp_extract_path}...")
            if not extract_zip(captions_zip_path, captions_temp_extract_path):
                logger.error("Failed to extract captions dataset. Exiting.")
                shutil.rmtree(captions_temp_extract_path, ignore_errors=True) # Clean up temp dir on failure
                return

            # Find the actual directory containing the files after extraction
            # It might be directly `captions_temp_extract_path` or a subfolder within it.
            actual_extracted_captions_dir = captions_temp_extract_path
            potential_subdirs = [d for d in os.listdir(captions_temp_extract_path) if os.path.isdir(os.path.join(captions_temp_extract_path, d))]
            for sub_dir in potential_subdirs:
                if raw_token_file_name in os.listdir(os.path.join(captions_temp_extract_path, sub_dir)):
                    actual_extracted_captions_dir = os.path.join(captions_temp_extract_path, sub_dir)
                    break # Found the folder containing the files

            raw_token_file_path = os.path.join(actual_extracted_captions_dir, raw_token_file_name)

            if not os.path.exists(raw_token_file_path):
                 logger.error(f"Raw token file not found after extraction: {raw_token_file_path}. Cannot process captions.")
                 shutil.rmtree(captions_temp_extract_path, ignore_errors=True) # Clean up temp dir on failure
                 return

            logger.info(f"Processing raw token file: {raw_token_file_path}")

            # Read the raw token file and process it
            try:
                # The format is typically "image_name#caption_number\tcaption_text"
                df_raw = pd.read_csv(raw_token_file_path, delimiter='\t', header=None, names=['image_caption_id', 'caption'])
                df_raw['image'] = df_raw['image_caption_id'].apply(lambda x: x.split('#')[0])
                df_processed = df_raw[['image', 'caption']]

                # Save the processed DataFrame to captions.txt
                df_processed.to_csv(processed_captions_file_path, sep='\t', index=False, header=False) # Save without header or index
                logger.info(f"Processed captions saved to: {processed_captions_file_path}")

            except Exception as e:
                logger.error(f"Error processing raw token file {raw_token_file_path}: {e}", exc_info=True)
                shutil.rmtree(captions_temp_extract_path, ignore_errors=True) # Clean up temp dir on failure
                return


            # List of split files to move from the extracted temp directory to download_dir
            split_files_to_move = [
                self.config.train_images_file,
                self.config.dev_images_file,
                self.config.test_images_file,
            ]

            # Move split files
            for file_name in split_files_to_move:
                source_path = os.path.join(actual_extracted_captions_dir, file_name)
                target_path = os.path.join(self.config.download_dir, file_name)
                if os.path.exists(source_path):
                    shutil.move(source_path, target_path)
                    logger.info(f"Moved {file_name} to: {target_path}")
                else:
                    logger.warning(f"Could not find split file '{file_name}' inside the captions zip extraction: {source_path}")


            # Clean up temporary captions extraction directory
            shutil.rmtree(captions_temp_extract_path)
            logger.info(f"Cleaned up temporary captions directory: {captions_temp_extract_path}")
        else:
            logger.info(f"Processed captions file already exists: {processed_captions_file_path}. Skipping processing.")

        # Check if all split files are present in the download directory
            split_files_to_check = [
                self.config.train_images_file,
                self.config.dev_images_file,
                self.config.test_images_file,
            ]
            for file_name in split_files_to_check:
                 target_path = os.path.join(self.config.download_dir, file_name)
                 if not os.path.exists(target_path):
                      logger.warning(f"Split file '{file_name}' not found in download directory: {target_path}. It might need manual placement or re-extraction.")


        logger.info("\nFlickr8k dataset download and processing complete!")
        logger.info(f"Images are in: {os.path.join(self.config.download_dir, 'Flicker8k_Dataset')}")
        logger.info(f"Processed captions file is in: {processed_captions_file_path}")
        logger.info(f"Split files are in: {self.config.download_dir}")