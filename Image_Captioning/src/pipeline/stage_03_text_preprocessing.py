# src/pipeline/stage_03_text_preprocessing.py
from src.utils.logging_setup import logger
from src.components.text_preprocessing import TextPreprocessor
from src.config.configuration import ConfigurationManager
import os
import pandas as pd


class TextPreprocessingPipeline:
    def __init__(self, config: ConfigurationManager):
        self.config = config.get_text_preprocessing_config()
        self.data_ingestion_config = config.get_data_ingestion_config()
        self.dataset_config = config.get_dataset_config() # To get caption path

    def run_pipeline(self) -> TextPreprocessor:
        try:
            preprocessor = TextPreprocessor(self.config)

            # Define path for vocabulary file
            vocab_path = os.path.join(self.data_ingestion_config.download_dir, "vocab.json")

            # Try to load existing vocabulary
            preprocessor.load_vocab(vocab_path)

            # If vocabulary is not yet built (e.g., first run or vocab file missing)
            if len(preprocessor.stoi) <= 4: # Only special tokens exist
                logger.info("Vocabulary not found or incomplete. Building new vocabulary...")
                # Load captions to build vocabulary
                caption_file_path = self.dataset_config.caption_path
                try:
                    # Assuming the caption file is the processed captions.txt from data ingestion
                    # which contains all image-caption pairs, regardless of split.
                    df = pd.read_csv(caption_file_path, delimiter="\t", names=["image", "caption"])
                    captions = df["caption"].tolist()
                    preprocessor.build_vocabulary(captions)
                    preprocessor.save_vocab(vocab_path)
                except FileNotFoundError:
                    logger.error(f"Caption file not found at {caption_file_path}. Cannot build vocabulary.")
                    raise
                except Exception as e:
                    logger.error(f"Error loading captions or building vocabulary: {e}")
                    raise
            else:
                logger.info("Vocabulary loaded successfully from file.")

            return preprocessor
        except Exception as e:
            logger.error(f"Error occurred during text preprocessing pipeline: {e}")
            raise