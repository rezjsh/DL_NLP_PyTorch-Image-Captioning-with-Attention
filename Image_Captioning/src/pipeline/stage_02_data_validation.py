
from src.components.data_validation import DataValidation
from src.utils.logging_setup import logger
from src.config.configuration import ConfigurationManager

class ValidationPipeline:
    def __init__(self, config: ConfigurationManager):
        self.config = config
        self.data_validation_config = self.config.get_data_validation_config()
        self.data_validation = DataValidation(self.data_validation_config)

    def run_pipeline(self):
        try:
            logger.info("Starting validation pipeline...")
            self.data_validation.validate_dataset()
            logger.info("Validation pipeline completed.")
        except Exception as e:
            logger.error(f"Validation pipeline failed: {e}")
            raise e

if __name__ == '__main__':
    try:
        config = ConfigurationManager()
        pipeline = ValidationPipeline(config)
        pipeline.run_pipeline()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")