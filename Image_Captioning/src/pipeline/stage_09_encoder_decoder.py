# src/pipeline/stage_09_encoder_decoder.py
from src.utils.logging_setup import logger
from src.config.configuration import ConfigurationManager
from src.components.encoder_decoder import TransformerImageCaptioningModel
from src.components.encoder import CNNTransformerEncoder
from src.components.decoder import TransformerDecoder

class EncoderDecoderPipeline:
    def __init__(self, config: ConfigurationManager):
        self.config = config.get_encoder_decoder_config()

    def run_pipeline(self, vocab_size: int, transformer_encoder: CNNTransformerEncoder, transformer_decoder: TransformerDecoder) -> TransformerImageCaptioningModel:
        try:
            logger.info("Running Encoder-Decoder Model Pipeline...")
            model = TransformerImageCaptioningModel(
                config=self.config,
                vocab_size=vocab_size,
                transformer_encoder=transformer_encoder,
                transformer_decoder=transformer_decoder
            )
            logger.info("Encoder-Decoder Model Pipeline completed.")
            return model
        except Exception as e:
            logger.error(f"Error occurred in Encoder-Decoder Model Pipeline: {e}")
            raise e