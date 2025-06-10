from pathlib import Path
from src.constants.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig
from src.utils.helpers import create_directory, read_yaml_file
from src.utils.logging_setup import logger

class ConfigurationManager:
    def __init__(self, config_file_path: str = CONFIG_FILE_PATH, params_file_path: str = PARAMS_FILE_PATH):
        self.config = read_yaml_file(config_file_path)
        self.params = read_yaml_file(params_file_path)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        logger.info("Getting data ingestion config")
        config = self.config.dataset
        # params = self.params.dataset
        logger.info(f"Data ingestion config: {config}")
        
        dirs_to_create = [config.download_dir]
        logger.info(f"Dirs to create: {dirs_to_create}")
        create_directory(dirs_to_create)
        logger.info("Creating data ingestion config")

        data_ingestion_config = DataIngestionConfig(
            download_dir=config.download_dir,
            images_zip_url=config.images_zip_url,
            captions_url=config.captions_url,
            images_zip_name=config.images_zip_name,
            captions_zip_name=config.captions_zip_name,
            captions_file_name=config.captions_file_name
        )
        logger.info(f"Data ingestion config created: {data_ingestion_config}")
        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        logger.info("Getting data validation config")
        config = self.config.validation

        dirs_to_create = [Path(config.validation_report_file).parent]
        create_directory(dirs_to_create)

        data_validation_config = DataValidationConfig(
            dataset_base_dir=config.dataset_base_dir,
            images_dir=config.images_dir,
            captions_file=config.captions_file,
            image_extensions=config.image_extensions,
            validation_report_file=config.validation_report_file
        )
        logger.info(f"Data validation config created: {data_validation_config}")

        return data_validation_config
