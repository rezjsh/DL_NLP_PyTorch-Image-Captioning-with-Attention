from typing import Any, Optional
from torch.utils.data import DataLoader
from src.utils.logging_setup import logger
from src.entity.config_entity import DataLoaderConfig
from src.modules.collate import MyCollate

class MyDataLoader:
    '''Custom DataLoader for image captioning'''
    def __init__(self, config: DataLoaderConfig, dataset, text_preprocessor) -> None:
        self.config = config
        self.dataset = dataset
        # Initialize the custom collate function using the PAD token ID from TextPreprocessor
        self.collate_fn = MyCollate(pad_idx=text_preprocessor.stoi["<PAD>"])

    def load_data(self, shuffle: Optional[bool] = None, drop_last: Optional[bool] = None) -> DataLoader:
        '''
        Load data, allowing specific overrides for shuffle and drop_last (useful for val/test splits).

        Args:
            shuffle (Optional[bool]): Overrides the config shuffle setting. True for train, False for val/test.
            drop_last (Optional[bool]): Overrides the config drop_last setting. True for train, False for val/test.
        '''

        # Use provided arguments, otherwise fall back to configuration file settings
        final_shuffle = shuffle if shuffle is not None else self.config.shuffle
        final_drop_last = drop_last if drop_last is not None else self.config.drop_last

        data_loader_name = "Training" if final_shuffle else ("Validation/Test")
        logger.info(f"Initializing {data_loader_name} DataLoader with batch size {self.config.batch_size}, workers {self.config.num_workers}, shuffle={final_shuffle}, drop_last={final_drop_last}.")

        return DataLoader(self.dataset,
                            batch_size=self.config.batch_size,
                            shuffle=final_shuffle,
                            num_workers=self.config.num_workers,
                            pin_memory=self.config.pin_memory,
                            drop_last=final_drop_last,
                            persistent_workers=True,
                            collate_fn=self.collate_fn)
