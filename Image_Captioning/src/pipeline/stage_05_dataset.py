from src.utils.logging_setup import logger
from src.config.configuration import ConfigurationManager
from src.components.image_preprocessing import ImagePreprocessor
from src.components.text_preprocessing import TextPreprocessor
from src.components.dataset import CaptioningDataset
from typing import Dict

class DatasetPipeline:
    """
    Pipeline stage for creating the CaptioningDataset objects for all splits (train/dev/test).
    It returns a dictionary containing all three datasets.
    """
    def __init__(self, config: ConfigurationManager) -> None:
        self.config = config.get_dataset_config()

    def run_pipeline(self, text_preprocessor: TextPreprocessor, image_preprocessor: ImagePreprocessor) -> Dict[str, CaptioningDataset]:
        """
        Runs the dataset creation pipeline for all three splits.

        Args:
            text_preprocessor (TextPreprocessor): The initialized text preprocessor (with vocabulary built).
            image_preprocessor (ImagePreprocessor): The initialized image preprocessor.

        Returns:
            Dict[str, CaptioningDataset]: A dictionary containing the 'train', 'dev', and 'test' datasets.
        """
        logger.info("Running CaptioningDatasetPipeline to create all splits...")

        datasets = {}
        splits = ['train', 'dev', 'test']

        for split in splits:
            logger.info(f"Creating CaptioningDataset for split: '{split}'...")
            try:
                dataset = CaptioningDataset(
                    config=self.config,
                    text_preprocessor=text_preprocessor,
                    image_preprocessor=image_preprocessor,
                    split=split
                )
                datasets[split] = dataset
                logger.info(f"Dataset for split '{split}' created successfully. Size: {len(dataset)}.")
            except Exception as e:
                logger.error(f"Error creating dataset for split '{split}': {e}")
                raise e

        logger.info("CaptioningDatasetPipeline completed. All three datasets (train, dev, test) are ready!")
        return datasets