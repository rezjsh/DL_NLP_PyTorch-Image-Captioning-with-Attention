from src.components.data_ingestion import DataIngestion
from src.utils.logging_setup import logger
from src.config.configuration import ConfigurationManager
from src.entity.config_entity import DataIngestionConfig


class DataIngestionPipeline:
    """
    Pipeline stage for data ingestion. Downloads and extracts the dataset.
    """
    def __init__(self, config_manager: ConfigurationManager):
        """
        Initializes the DataIngestionPipeline.

        Args:
            config_manager (ConfigurationManager): The configuration manager instance.
        """
        self.config = config_manager.get_data_ingestion_config()
        self.data_ingestion = DataIngestion(config=self.config)

    def run_pipeline(self) -> DataIngestionConfig:
        """
        Executes the data ingestion pipeline.

        Returns:
            DataIngestionConfig: The configuration used for data ingestion,
                                 which contains paths to the ingested data.
        """
        try:
            logger.info("Starting data ingestion pipeline")
            self.data_ingestion.download_dataset()
            logger.info(f"Data ingestion completed successfully. Data downloaded to: {self.config.download_dir}")
            return self.config
        except Exception as e:
            logger.error(f"Error in data ingestion pipeline: {e}", exc_info=True)
            raise e

if __name__ == '__main__':
    try:
        config_manager_ingestion = ConfigurationManager()
        data_ingestion_pipeline = DataIngestionPipeline(config=config_manager_ingestion)
        data_ingestion_pipeline.run_pipeline()
    except Exception as e:
        logger.error(f"Error in data ingestion pipeline: {e}")
        raise e