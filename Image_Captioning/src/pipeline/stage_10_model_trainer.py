from src.utils.logging_setup import logger
from src.config.configuration import ConfigurationManager
from src.components.model_trainer import Trainer
from src.components.encoder_decoder import TransformerImageCaptioningModel
from src.components.text_preprocessing import TextPreprocessor
from torch.utils.data import DataLoader

class TrainingPipeline:
    def __init__(self, config: ConfigurationManager):
        self.config = config.get_model_trainer_config()

    def run_pipeline(self,
                     model: TransformerImageCaptioningModel,
                     train_loader: DataLoader,
                     val_loader: DataLoader,
                     text_preprocessor: TextPreprocessor) -> None:
        '''
        Initializes and runs the training component.

        Args:
            model (TransformerImageCaptioningModel): The final Encoder-Decoder model.
            train_loader (DataLoader): The DataLoader for the training set.
            val_loader (DataLoader): The DataLoader for the validation set.
            text_preprocessor (TextPreprocessor): The initialized text preprocessor (for PAD index).
        '''
        logger.info("Trainer initialized. Starting training...")

        # Instantiate and run the Trainer component
        trainer = Trainer(
            config=self.config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader, # Pass the validation loader
            text_preprocessor=text_preprocessor
        )

        trainer.train()
        logger.info("Training process completed.")