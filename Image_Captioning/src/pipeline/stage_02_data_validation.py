from src.components.data_validation import DataValidation
from src.utils.logging_setup import logger
from src.config.configuration import ConfigurationManager
from src.entity.config_entity import DataValidationConfig

class ValidationPipeline:
    """
    Pipeline stage for data validation. Validates the integrity and structure
    of the downloaded dataset.
    """
    def __init__(self, config_manager: ConfigurationManager):
        """
        Initializes the ValidationPipeline.

        Args:
            config_manager (ConfigurationManager): The configuration manager instance.
        """
        self.config = config_manager.get_data_validation_config()
        self.data_validation = DataValidation(self.config)

    def run_pipeline(self) -> bool:
        """
        Executes the data validation pipeline.

        Returns:
            bool: True if data validation passes all critical checks, False otherwise.
        """
        try:
            logger.info("Starting data validation pipeline...")
            validation_status = self.data_validation.validate_dataset()
            if validation_status:
                logger.info("Data validation pipeline completed successfully: All critical checks passed.")
            else:
                logger.warning("Data validation pipeline completed with warnings/failures. Please check the report.")
            return validation_status
        except Exception as e:
            logger.error(f"Data validation pipeline failed: {e}", exc_info=True)
            raise e
        
        
if __name__ == '__main__':
    try:
        config = ConfigurationManager()
        pipeline = ValidationPipeline(config)
        pipeline.run_pipeline()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")