from src.utils.logging_setup import logger
from src.components.image_preprocessing import ImagePreprocessor
from src.config.configuration import ConfigurationManager


class ImageProcessingPipeline:
    def __init__(self, config: ConfigurationManager):
        self.config = config.get_image_preprocessing_config()

    def run_pipeline(self):
        try:
            logger.info("Running image processing pipeline...")
            image_preprocessor = ImagePreprocessor(self.config)
            return image_preprocessor
        except Exception as e:
            logger.error(f"Error occurred in image processing pipeline: {e}")
            raise e