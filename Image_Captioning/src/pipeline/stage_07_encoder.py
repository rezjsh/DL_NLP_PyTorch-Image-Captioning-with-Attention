# src/pipeline/stage_07_encoder.py
from src.utils.logging_setup import logger
from src.config.configuration import ConfigurationManager
from src.components.encoder import CNNTransformerEncoder # Assuming this is the main encoder

class EncoderPipeline:
    def __init__(self, config: ConfigurationManager):
        self.config = config.get_encoder_config()

    def run_pipeline(self) -> CNNTransformerEncoder:
        try:
            logger.info("Running Encoder Pipeline...")
            encoder = CNNTransformerEncoder(self.config)
            logger.info("Encoder Pipeline completed.")
            return encoder
        except Exception as e:
            logger.error(f"Error occurred in Encoder Pipeline: {e}")
            raise e