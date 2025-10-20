from torch.utils.data import DataLoader
from src.components.data_loader import MyDataLoader
from src.components.text_preprocessing import TextPreprocessor
from src.utils.logging_setup import logger
from src.config.configuration import ConfigurationManager
from src.components.dataset import CaptioningDataset
from typing import Dict

class DataLoaderPipeline:
    '''Pipeline for creating all data loaders (train, dev, test)'''
    def __init__(self, config: ConfigurationManager) -> None:
        self.config_manager = config
        self.config = config.get_data_loader_config()

    def run_pipeline(self, datasets: Dict[str, CaptioningDataset], text_preprocessor: TextPreprocessor) -> Dict[str, DataLoader]:
        '''
        Creates and returns all three DataLoader instances (train, dev, test).

        Args:
            datasets (Dict[str, CaptioningDataset]): Dictionary containing 'train', 'dev', and 'test' datasets.
            text_preprocessor (TextPreprocessor): The initialized text preprocessor (to get the PAD token ID).

        Returns:
            Dict[str, DataLoader]: A dictionary containing the 'train_loader', 'val_loader', and 'test_loader'.
        '''
        logger.info("Running DataLoaderPipeline to create all loaders...")

        loaders = {}

        # Configuration mapping for each split
        split_configs = {
            'train': {'dataset': datasets['train'], 'shuffle': True, 'drop_last': True, 'key_name': 'train_loader'},
            'dev':   {'dataset': datasets['dev'],   'shuffle': False, 'drop_last': False, 'key_name': 'val_loader'},
            'test':  {'dataset': datasets['test'],  'shuffle': False, 'drop_last': False, 'key_name': 'test_loader'},
        }

        for split, cfg in split_configs.items():
            try:
                # 1. Instantiate the MyDataLoader component
                my_data_loader = MyDataLoader(
                    config=self.config,
                    dataset=cfg['dataset'],
                    text_preprocessor=text_preprocessor
                )

                # 2. Call load_data with specific settings for the split
                data_loader = my_data_loader.load_data(
                    shuffle=cfg['shuffle'],
                    drop_last=cfg['drop_last']
                )

                loaders[cfg['key_name']] = data_loader
                logger.info(f"DataLoader for split '{split}' created and saved as '{cfg['key_name']}'.")

            except Exception as e:
                logger.error(f"Error creating DataLoader for split '{split}': {e}")
                raise e

        logger.info("DataLoaderPipeline completed. All three loaders are ready!")
        return loaders