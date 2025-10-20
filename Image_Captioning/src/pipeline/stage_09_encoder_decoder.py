
from src.components.encoder import CNNTransformerEncoder
from src.components.decoder import  TransformerDecoder
from src.utils.logging_setup import logger
from src.config.configuration import ConfigurationManager
from src.components.encoder_decoder import TransformerImageCaptioningModel

class EncoderDecoderPipeline:
    def __init__(self, config:ConfigurationManager, vocab_size:int, transformer_encoder: CNNTransformerEncoder, transformer_decoder: TransformerDecoder) -> None:
        self.config = config.get_encoder_decoder_config()
        self.vocab_size = vocab_size
        self.encoder = transformer_encoder
        self.decoder = transformer_decoder

    def run_pipeline(self):
        logger.info("Running Encoder-Decoder Pipeline")
        model = TransformerImageCaptioningModel(config=self.config, vocab_size=self.vocab_size, transformer_encoder=self.encoder, transformer_decoder=self.decoder)
        logger.info(f"Encoder-Decoder model created: {model}")
        return model