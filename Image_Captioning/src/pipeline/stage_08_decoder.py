# src/pipeline/stage_08_decoder.py
from src.utils.logging_setup import logger
from src.config.configuration import ConfigurationManager
from src.components.decoder import TransformerDecoder

class DecoderPipeline:
    def __init__(self, config: ConfigurationManager, vocab_size: int):
        self.config = config.get_decoder_config(vocab_size=vocab_size)

    def run_pipeline(self) -> TransformerDecoder:
        try:
            logger.info("Running Decoder Pipeline...")
            decoder = TransformerDecoder(self.config)
            logger.info("Decoder Pipeline completed.")
            return decoder
        except Exception as e:
            logger.error(f"Error occurred in Decoder Pipeline: {e}")
            raise e