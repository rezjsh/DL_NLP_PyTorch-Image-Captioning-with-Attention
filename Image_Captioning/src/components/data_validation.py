import json
import os
from pathlib import Path
import random
import pandas as pd
from PIL import Image
from src.utils.logging_setup import logger
from src.entity.config_entity import DataValidationConfig 

class DataValidation:
    """
    A utility class to validate the downloaded dataset.
    It checks directory structure, file existence, and captions file format.
    """

    def __init__(self, config: DataValidationConfig):
        """
        Initializes the validator.
        Args:
            dataset_base_dir (str): The base directory where the Flickr8k dataset is located.
                                This should be the same as the download_dir used by the downloader.

        """
        self.config = config
        self.validation_report = {}
        self.all_image_paths = [] 
        logger.info(f"Validator initialized for dataset at: {self.config.dataset_base_dir}")

    def _check_directory_structure(self):
        """Checks if the main dataset directory and its 'images' subdirectory exist."""
        logger.info("Checking Directory Structure.")
        if not os.path.exists(self.config.dataset_base_dir):
            logger.error(f"Dataset base directory not found: {self.config.dataset_base_dir}")
            self.validation_report["directory_structure_check"] = {
                "status": "failed",
                "message": f"Dataset base directory not found: {self.config.dataset_base_dir}"
            }
            return False
        logger.info(f"Dataset base directory exists: {self.config.dataset_base_dir}")

        
        if not os.path.exists(self.config.images_dir):
            logger.error(f"Images directory not found: {self.config.images_dir}")
            self.validation_report["directory_structure_check"] = {
                "status": "failed",
                "message": f"Images directory not found: {self.config.images_dir}"
            }
            return False
        logger.info(f"Images directory exists: {self.config.images_dir}")
        self.validation_report["directory_structure_check"] = {
            "status": "passed",
            "message": f"Images directory exists: {self.config.images_dir}"
        }
        return True

    def _check_images(self):
        """Checks if the images directory contains files and if they are valid images."""
        logger.info("Checking Images")
        image_files = [f for f in os.listdir(self.config.images_dir) if f.lower().endswith(tuple(self.config.image_extensions))]
        
        if not image_files:
            logger.warning(f"No image files found in {self.config.images_dir}.")
            self.validation_report["image_validation_check"] = {
                "status": "failed",
                "message": f"No image files found in {self.config.images_dir}"
            }
            return False

        logger.info(f"Found {len(image_files)} potential image files in {self.config.images_dir}.")
        self.validation_report["image_validation_check"] = {
            "status": "passed",
            "message": f"Found {len(image_files)} potential image files in {self.config.images_dir}"
        }
        
        # Check a sample of images for validity to avoid iterating through all if there are many
        sample_size = min(10, len(image_files))
        logger.info(f"Checking validity of {sample_size} sample images...")
        valid_sample_count = 0
        for i, img_name in enumerate(random.sample(image_files, sample_size)):
            img_path = os.path.join(self.config.images_dir, img_name)
            try:
                with Image.open(img_path) as img:
                    img.verify()
                valid_sample_count += 1
            except Exception as e:
                logger.error(f"Could not open or verify sample image '{img_name}': {e}")
                
        if valid_sample_count == sample_size:
            logger.info(f"All {sample_size} sample images are valid.")
            self.validation_report["image_validation_check"]["message"] = f"All {sample_size} sample images are valid."
            self.validation_report["image_validation_check"]["status"] = "passed"

            return True
        else:
            logger.warning(f"Some sample images were invalid. Total valid samples: {valid_sample_count}/{sample_size}")
            self.validation_report["image_validation_check"]["message"] = f"Some sample images were invalid. Total valid samples: {valid_sample_count}/{sample_size}"
            self.validation_report["image_validation_check"]["status"] = "failed"
            return False


    def _check_captions_file(self):
        """Checks if the captions file exists and its basic CSV structure."""
        logger.info("Checking Captions File.")
        if not os.path.exists(self.config.captions_file):
            logger.error(f"Captions file not found: {self.config.captions_file}")
            self.validation_report["captions_file_check"] = {
                "status": "failed",
                "message": f"Captions file not found: {self.config.captions_file}"
            }
            return False
        logger.info(f"Captions file exists: {self.config.captions_file}")

        try:
            df = pd.read_csv(self.config.captions_file, delimiter='\t',header=None)
            df['image'] = df[0].apply(lambda x: x.split('#')[0])
            df['caption'] = df[1]
            df = df[['image', 'caption']]
            logger.info(f"Captions file loaded successfully. It contains {len(df)} entries.")

            
            # Check for expected columns
            expected_columns = ['image', 'caption']
            if not all(col in df.columns for col in expected_columns):
                logger.error(f"Captions file missing expected columns. Found: {df.columns.tolist()}, Expected: {expected_columns}")
                self.validation_report["captions_file_check"] = {
                    "status": "failed",
                    "message": f"Captions file missing expected columns. Found: {df.columns.tolist()}, Expected: {expected_columns}"
                }
                return False
            logger.info(f"Captions file contains 'image' and 'caption' columns.")

            # Check for non-empty entries
            if df.isnull().any().any():
                print("Captions file contains missing (NaN) values.")
                self.validation_report["captions_file_check"] = {
                    "status": "warning",
                    "message": f"Captions file contains missing (NaN) values."
                }
            # Check for duplicate captions for the same image (can be valid, but good to note)
            if df.duplicated(subset=['image', 'caption']).any():
                logger.info("Captions file contains duplicate (image, caption) pairs.")
                self.validation_report["captions_file_check"] = {
                    "status": "info",
                    "message": f"Captions file contains duplicate (image, caption) pairs."
                }

            return True
        except pd.errors.EmptyDataError:
            logger.error("Captions file is empty.")
            self.validation_report["captions_file_check"] = {
                "status": "failed",
                "message": f"Captions file is empty."
            }
            return False
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing captions file (CSV format issue): {e}")
            self.validation_report["captions_file_check"] = {
                "status": "failed",
                "message": f"Error parsing captions file (CSV format issue): {e}"
            }
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while checking captions file: {e}")
            self.validation_report["captions_file_check"] = {
                "status": "failed",
                "message": f"An unexpected error occurred while checking captions file: {e}"
            }
            return False

    def _cross_validate_images_and_captions(self):
        """
        Performs a cross-validation between image filenames in caption file
        and actual image files present in the images directory.
        """
        logger.info("Cross-validating Images and Captions")
        try:
            df = pd.read_csv(self.config.captions_file, delimiter='\t',header=None)
            df['image'] = df[0].apply(lambda x: x.split('#')[0])
            df['caption'] = df[1]
            df = df[['image', 'caption']]
            captioned_images = set(df['image'].unique())

            actual_image_files = {f for f in os.listdir(self.config.images_dir) if f.lower().endswith(tuple(self.config.image_extensions))}

            images_in_captions_not_on_disk = captioned_images - actual_image_files
            images_on_disk_not_in_captions = actual_image_files - captioned_images

            if not images_in_captions_not_on_disk and not images_on_disk_not_in_captions:
                logger.info("All images referenced in caption file exist on disk, and vice-versa (for recognized image types).")
                self.validation_report["image_validation_check"] = {
                    "status": "passed",
                    "message": f"All sample images are valid."
                }
                return True
            else:
                if images_in_captions_not_on_disk:
                    logger.warning(f"{len(images_in_captions_not_on_disk)} images refcaption file are NOT found on disk. Sample: {list(images_in_captions_not_on_disk)[:5]}")
                    self.validation_report["image_validation_check"] = {
                        "status": "warning",
                        "message": f"{len(images_in_captions_not_on_disk)} images refcaption file are NOT found on disk."
                    }
                if images_on_disk_not_in_captions:
                    logger.warning(f"{len(images_on_disk_not_in_captions)} images found on disk are NOT refcaption file. Sample: {list(images_on_disk_not_in_captions)[:5]}")
                    self.validation_report["image_validation_check"] = {
                        "status": "warning",
                        "message": f"{len(images_on_disk_not_in_captions)} images found on disk are NOT refcaption file."
                    }
                return False
        except FileNotFoundError:
            logger.error("Captions file or images directory not found for cross-validation.")
            self.validation_report["image_validation_check"] = {
                "status": "failed",
                "message": f"Captions file or images directory not found for cross-validation."
            }
            return False
        except Exception as e:
            logger.error(f"An error occurred during cross-validation: {e}")
            self.validation_report["image_validation_check"] = {
                "status": "failed",
                "message": f"An error occurred during cross-validation: {e}"
            }
            return False

    def _save_validation_report(self) -> bool:
        """Save validation report to a JSON file."""
        try:
            report_file_path = Path(self.config.validation_report_file)
            report_file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(report_file_path, 'w') as f:
                json.dump(self.validation_report, f, indent=4)

            logger.info(f"Validation report saved to {report_file_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving validation report: {e}")
            return False

    def validate_dataset(self):
        """
        Runs all validation checks on the dataset.
        Returns:
            bool: True if all critical checks pass, False otherwise.
        """
        logger.info(f"Starting validation of dataset at: {self.config.dataset_base_dir}")
        
        overall_status = True

        # Critical checks
        if not self._check_directory_structure():
            overall_status = False
        
        if not self._check_images():
            overall_status = False

        if not self._check_captions_file():
            overall_status = False

        self._cross_validate_images_and_captions()
        
        is_saved = self._save_validation_report()
        if not is_saved:
           logger.warning("Failed to save validation report.")

        logger.info("Validation Summary")
        if overall_status:
            logger.info("All critical dataset validation checks passed!")
            logger.info(f"Validation report: {self.validation_report}")
        else:
            logger.error("Some critical dataset validation checks failed. Please review errors above.")
            logger.error(f"Validation report: {self.validation_report}")

        return overall_status
