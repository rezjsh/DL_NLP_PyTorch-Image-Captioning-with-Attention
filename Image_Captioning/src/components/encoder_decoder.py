import torch
import torch.nn as nn
from typing import Any, List, Tuple
from src.utils.logging_setup import logger
from src.entity.config_entity import EncoderDecoderConfig
from src.components.encoder import CNNTransformerEncoder
from src.components.decoder import TransformerDecoder
from src.utils.device import DEVICE

class TransformerImageCaptioningModel(nn.Module):
    """
    Combines a Transformer-based Encoder (or just CNN) and a TransformerDecoder
    into a single end-to-end image captioning model.
    """
    def __init__(self, config: EncoderDecoderConfig,
                 vocab_size: int,
                 transformer_encoder: CNNTransformerEncoder,
                 transformer_decoder: TransformerDecoder,
                 pad_idx: int = 0
                 ) -> None:
        """
        Args:
            config (EncoderDecoderConfig): Configuration for the model.
            vocab_size (int): Size of the vocabulary for the decoder.
            transformer_encoder (CNNTransformerEncoder): The encoder component.
            transformer_decoder (TransformerDecoder): The decoder component.
            pad_idx (int): The vocabulary ID for the padding token.
        """
        super(TransformerImageCaptioningModel, self).__init__()
        self.config = config
        self.pad_idx = pad_idx

        # 1. Initialize Encoder
        encoder_output_d_model = None
        if self.config.encoder_type == "cnn":
            # Assuming EncoderCNN is a simple feature extractor
            self.encoder = EncoderCNN(model_name=self.config.cnn_backbone)
            # The CNN's output channels determine its embedding dimension
            encoder_output_d_model = self.encoder.out_channels # Corrected: Use out_channels for CNN
            # Fine-tuning logic for pure CNN
            for param in self.encoder.parameters():
                param.requires_grad = self.config.fine_tune_cnn
            logger.debug(f"EncoderCNN initialized with fine_tune_cnn={self.config.fine_tune_cnn}.")
        elif self.config.encoder_type == "cnn_transformer":
            self.encoder = transformer_encoder
            encoder_output_d_model = self.encoder.config.embed_dim # Corrected: Access from encoder's config
            logger.debug(f"CNNTransformerEncoder initialized with embed_dim={self.encoder.config.embed_dim}.") # Corrected
        else:
            logger.error(f"Unknown encoder_type: {self.config.encoder_type}")
            raise ValueError(f"Unknown encoder_type: {self.config.encoder_type}")

        # 2. Dimension Check (CRUCIAL)
        if self.config.d_model != encoder_output_d_model:
            logger.error(f"Decoder d_model ({self.config.d_model}) does not match "
                         f"encoder output d_model ({encoder_output_d_model}).")
            raise ValueError(f"Decoder d_model ({self.config.d_model}) must match "
                             f"encoder output d_model ({encoder_output_d_model}).")

        # 3. Initialize Decoder
        self.decoder = transformer_decoder

        # Store the decoder's mask generation method for easy access
        self._generate_causal_mask = self.decoder._generate_square_subsequent_mask
        logger.debug("Decoder mask generation method stored for easy access.")

    def forward(self, images: torch.Tensor, captions: torch.Tensor, caption_lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training the Transformer-based captioning model.
        Args:
            images (torch.Tensor): Batch of input images.
            captions (torch.Tensor): Batch of padded captions (right-shifted: <SOS> w1 w2 ...).
            caption_lengths (torch.Tensor): Original lengths of captions in the batch (not used directly).
        Returns:
            torch.Tensor: Predicted word scores (batch_size, max_decode_length, vocab_size).
        """
        logger.debug("Starting forward pass of TransformerImageCaptioningModel.") # Changed to debug
        self.train()

        # 1. Encoder Pass

        # [Image of CNN-Transformer Encoder Architecture]

        # Output: (batch_size, num_pixels, encoder_dim)
        encoder_out = self.encoder(images)

        # 2. Decoder Masks (CRUCIAL for training)
        tgt_seq_len = captions.size(1);

        # Causal Mask (Look-ahead mask) - expects float with -inf or 0.0
        trg_mask = self._generate_causal_mask(tgt_seq_len)
        logger.debug(f"Causal mask generated with shape: {trg_mask.shape}") # Changed to debug
        # Key Padding Mask (Masks padding tokens in the target sequence)
        # Prevents the decoder's self-attention from attending to padding
        # Should be boolean where True means being ignored.
        # Convert boolean mask to float mask as suggested by PyTorch warning.
        tgt_key_padding_mask = (captions == self.pad_idx).float().masked_fill(captions == self.pad_idx, float('-inf')).masked_fill(captions != self.pad_idx, float(0.0))
        logger.debug(f"Key padding mask generated with shape: {tgt_key_padding_mask.shape}") # Changed to debug
        #  Decoder Pass
        predictions = self.decoder(
            trg=captions,
            memory=encoder_out,
            trg_mask=trg_mask,
            tgt_key_padding_mask=tgt_key_padding_mask # Pass the float mask
        )
        logger.debug(f"Decoder output generated with shape: {predictions.shape}") # Changed to debug
        return predictions


    def generate_caption(self, image_tensor: torch.Tensor, vocab: Any, beam_size: int = 3) -> Tuple[List[str], torch.Tensor]:
        """
        Generates a caption for a single image using beam search (via the decoder).
        """
        logger.debug("Starting caption generation (inference) for a single image.")
        self.eval()
        image_tensor = image_tensor.to(DEVICE)
        logger.debug(f"Image tensor moved to device: {DEVICE}")
        # 1. Encoder Pass (Inference)
        encoder_out = self.encoder(image_tensor) # (1, num_pixels, encoder_dim)
        logger.debug(f"Encoder output generated with shape: {encoder_out.shape}")
        # 2. Decoder Inference (Handles max_len internally)
        caption_words, alphas = self.decoder.generate_caption(
            encoder_out=encoder_out,
            vocab=vocab,
            beam_size=beam_size,
            max_len=self.config.max_caption_length
        )
        logger.debug(f"Caption generated with length: {len(caption_words)}")
        return caption_words, alphas