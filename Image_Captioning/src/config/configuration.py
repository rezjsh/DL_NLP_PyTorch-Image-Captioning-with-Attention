from src.constants.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH

from Image_Captioning.src.entity.config_entity import DataIngestionConfig
from Image_Captioning.src.utils.helpers import create_directory, read_yaml_file


class ConfigurationManager:
    def __init__(self, config_file_path: str = CONFIG_FILE_PATH, params_file_path: str = PARAMS_FILE_PATH):
        self.config = read_yaml_file(config_file_path)
        self.params = read_yaml_file(params_file_path)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.dataset
        # params = self.params.dataset

        dirs_to_create = [config.download_dir]
        create_directory(dirs_to_create)

        data_ingestion_config = DataIngestionConfig(
            download_dir=config.download_dir,
            images_zip_url=config.images_zip_url,
            captions_url=config.captions_url,
            images_zip_name=config.images_zip_name,
            captions_zip_name=config.captions_zip_name,
            captions_file_name=config.captions_file_name
        )

        return data_ingestion_config
