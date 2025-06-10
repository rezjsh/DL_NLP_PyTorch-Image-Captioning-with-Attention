from src.components.data_ingestion import DataIngestion
from src.utils.logging_setup import logger
from src.config.configuration import ConfigurationManager

class DataIngestionPipeline:
    def __init__(self, config: ConfigurationManager):
        self.config = config
        self.data_ingestion_config = self.config.get_data_ingestion_config()
        self.data_ingestion = DataIngestion(config=self.data_ingestion_config)

    def run_pipeline(self):
        try:
            logger.info("Starting data ingestion pipeline")
            self.data_ingestion.initiate_data_ingestion()
            logger.info(f"Data ingestion completed successfully.")
        except Exception as e:
            logger.error(f"Error in data ingestion pipeline: {e}")
            raise e

if __name__ == '__main__':
    try:
        config_manager_ingestion = ConfigurationManager()
        data_ingestion_pipeline = DataIngestionPipeline(config=config_manager_ingestion)
        data_ingestion_pipeline.run_pipeline()
    except Exception as e:
        logger.error(f"Error in data ingestion pipeline: {e}")
        raise e