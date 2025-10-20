import torch
import torch.nn as nn
from typing import Any, List, Tuple, Union
from src.utils.logging_setup import logger
from src.entity.config_entity import EncoderDecoderConfig
from src.components.encoder import CNNTransformerEncoder, EncoderCNN
from src.components.decoder import TransformerDecoder
from src.utils.device import DEVICE

class TransformerImageCaptioningModel(nn.Module):
    """
    Combines a Transformer-based Encoder (or just CNN) and a TransformerDecoder
    into a single end-to-end image captioning model.
    """
    def __init__(self, config: EncoderDecoderConfig, vocab_size: int, transformer_encoder: CNNTransformerEncoder, transformer_decoder: TransformerDecoder) -> None:
        """
        Args:
            config (EncoderDecoderConfig): Configuration for the model.
            vocab_size (int): Size of the vocabulary for the decoder.
            transformer_encoder (CNNTransformerEncoder): The encoder component of the model.
            transformer_decoder (TransformerDecoder): The decoder component of the model.
        """
        super(TransformerImageCaptioningModel, self).__init__()
        self.config = config

        encoder_output_d_model = None # To hold the actual d_model coming out of the encoder

        # Initialize the chosen encoder (CNN or CNNTransformerEncoder)
        if self.config.encoder_type == "cnn":
            # Assuming EncoderCNN is defined elsewhere and has an encoder_dim attribute
            self.encoder = EncoderCNN(model_name=self.config.cnn_backbone)
            encoder_output_d_model = self.encoder.encoder_dim
            for param in self.encoder.parameters():
                param.requires_grad = self.config.fine_tune_cnn
            # logger.info(f"CNN encoder parameters trainable: {self.config.fine_tune_cnn}")

        elif self.config.encoder_type == "cnn_transformer":
            self.encoder = transformer_encoder
            encoder_output_d_model = self.encoder.embed_dim
        else:
            raise ValueError(f"Unknown encoder_type: {self.config.encoder_type}")

        # Initialize the Transformer Decoder
        # Crucially, the decoder's `d_model` MUST match the encoder's output dimension

        if self.config.d_model != encoder_output_d_model:
            raise ValueError(f"Decoder d_model ({self.config.d_model}) does not match "
                             f"encoder output d_model ({encoder_output_d_model}). Please set decoder d_model to encoder's.")

        self.decoder = transformer_decoder

        logger.info("TransformerImageCaptioningModel initialized.")
        logger.info(f"Total trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    # 1. Removed `pad_idx` from arguments.
    # 2. Changed return type hint to `torch.Tensor` as TransformerDecoder.forward only returns one tensor.
    def forward(self, images: torch.Tensor, captions: torch.Tensor, caption_lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training the Transformer-based captioning model.
        Args:
            images (torch.Tensor): Batch of input images.
            captions (torch.Tensor): Batch of padded captions (shifted right, e.g., <SOS> w1 w2).
            caption_lengths (torch.Tensor): Original lengths of captions in the batch (not directly used by decoder.forward here).
        Returns:
            torch.Tensor: Predicted word scores (batch_size, max_decode_length, vocab_size).
        """
        self.train() # Set model to training mode (important for dropout)
        encoder_out = self.encoder(images) # (batch_size, num_pixels, encoder_dim)

        # Corrected call to self.decoder:
        # - Pass `captions` (LongTensor) as the first argument (`trg`).
        # - Pass `encoder_out` (FloatTensor) as the second argument (`memory`).
        # - `trg_mask` is optional, let the decoder handle it or explicitly pass None.
        # - Removed `caption_lengths` and `pad_idx` as they are not arguments to TransformerDecoder.forward.
        # - Expecting only one return value from TransformerDecoder.forward.
        predictions = self.decoder(captions, encoder_out)

        # TransformerDecoder.forward only returns the output tensor, no alphas or sort_ind.
        return predictions
    # --- MODIFIED FORWARD METHOD END ---


    def generate_caption(self, image_tensor: torch.Tensor, vocab: Any, beam_size: int = 3) -> Tuple[List[str], Union[torch.Tensor, None]]:
        """
        Generates a caption for a single image using beam search.
        Args:
            image_tensor (torch.Tensor): Single input image tensor (1, C, H, W).
            vocab (object): The vocabulary object (e.g., TextPreprocessor instance).
            beam_size (int): The number of sequences to keep at each step.
        Returns:
            tuple: (caption_words, attention_alphas)
                caption_words (list): List of generated words (excluding special tokens).
                attention_alphas (None): TransformerDecoder does not directly return attention alphas from this method.
        """
        self.eval()
        # The decoder's generate_caption already handles no_grad internally
        image_tensor = image_tensor.to(DEVICE)
        encoder_out = self.encoder(image_tensor) # (1, num_pixels, encoder_dim)

        # The TransformerDecoder's generate_caption method requires max_len
        caption_words, alphas = self.decoder.generate_caption(
            encoder_out=encoder_out,
            vocab=vocab,
            beam_size=beam_size,
            max_len=self.config.max_caption_length # Pass the configured max_caption_length
        )
        return caption_words, alphas