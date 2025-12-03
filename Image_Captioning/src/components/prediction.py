# src/components/prediction.py
from __future__ import annotations
from typing import Optional
from PIL import Image
import torch

from src.components.image_preprocessing import ImagePreprocessor
from src.components.text_preprocessing import TextPreprocessor
from src.components.encoder_decoder import TransformerImageCaptioningModel
from src.components.encoder import CNNTransformerEncoder
from src.components.decoder import TransformerDecoder
from src.config.configuration import ConfigurationManager
from src.utils.device import DEVICE
from src.utils.logging_setup import logger


class ImageCaptionPredictor:
    """
    High-level predictor that wires preprocessing, model construction, weight loading,
    and caption generation. Designed for single-image inference in the Flask app.
    """

    def __init__(self,
                 config_manager: Optional[ConfigurationManager] = None,
                 model_ckpt_path: Optional[str] = None,
                 vocab_json_path: Optional[str] = None,
                 ) -> None:
        # Load configuration
        self.cm = config_manager or ConfigurationManager()
        self.encoder_decoder_cfg = self.cm.get_encoder_decoder_config()
        self.img_cfg = self.cm.get_image_preprocessing_config()
        self.txt_cfg = self.cm.get_text_preprocessing_config()
        vocab_json_path = 'artifacts\\models\\vocab.json'
        # Initialize preprocessors
        self.image_preprocessor = ImagePreprocessor(self.img_cfg)
        self.text_preprocessor = TextPreprocessor(self.txt_cfg)

        # Load vocab if provided via config/param or optional arg
        # Expect a path to a JSON file saved by TextPreprocessor.save_vocab
        if vocab_json_path is None:
            # Try to read from params if present
            try:
                vocab_json_path = self.cm.params.inference.vocab_path
            except Exception:
                vocab_json_path = None
        if vocab_json_path:
            try:
                self.text_preprocessor.load_vocab(vocab_json_path)
                logger.info(f"Loaded vocabulary from: {vocab_json_path}")
            except Exception as e:
                logger.warning(f"Failed to load vocab from {vocab_json_path}: {e}. Using default minimal vocab.")

        vocab_size = len(self.text_preprocessor)

        # Build encoder and decoder from configs
        encoder_cfg = self.cm.get_encoder_config()
        decoder_cfg = self.cm.get_decoder_config(vocab_size=vocab_size)

        transformer_encoder = CNNTransformerEncoder(encoder_cfg)
        transformer_decoder = TransformerDecoder(decoder_cfg)

        # Compose full model
        self.model = TransformerImageCaptioningModel(
            config=self.encoder_decoder_cfg,
            vocab_size=vocab_size,
            transformer_encoder=transformer_encoder,
            transformer_decoder=transformer_decoder,
            pad_idx=self.text_preprocessor.stoi["<PAD>"]
        ).to(DEVICE)

        # Load weights for the full model from .pth file
        if model_ckpt_path is None:
            try:
                model_ckpt_path = self.cm.config.inference.model_ckpt_path
            except Exception:
                model_ckpt_path = None

        self._load_model_weights(model_ckpt_path)
        self.model.eval()
        logger.info("ImageCaptionPredictor initialized and ready.")

    def _load_model_weights(self, model_ckpt_path: Optional[str]) -> None:
            """Load full model weights from a .pth file."""
            try:
                if model_ckpt_path is not None:
                    state_dict = torch.load(model_ckpt_path, map_location=DEVICE)
                    # If using torch.save(model.state_dict(), ...) this is just the state_dict
                    if isinstance(state_dict, dict) and all(isinstance(k, str) for k in state_dict.keys()):
                        self.model.load_state_dict(state_dict, strict=False)
                        logger.info(f"Loaded full model weights from {model_ckpt_path}")
                    else:
                        logger.error(f"Unexpected checkpoint format in {model_ckpt_path}")
                else:
                    logger.warning("No model checkpoint path provided; using randomly initialized model.")
            except Exception as e:
                logger.error(f"Failed to load model weights: {e}")

    def predict(self, image_path: str, beam_size: int = 5, max_repeats: int = 2) -> str:
        """
        Predict a caption and remove repetitions, especially trailing repeats.
        """
        pil_img = Image.open(image_path).convert('RGB')
        img_tensor = self.image_preprocessor.preprocess_image(pil_img).unsqueeze(0).to(DEVICE)

        # Generate caption
        words, _ = self.model.generate_caption(img_tensor, vocab=self.text_preprocessor, beam_size=beam_size)
        
        # Remove trailing repetitions and excessive repeats
        cleaned_words = []
        seen_count = {}
        
        for word in words:
            seen_count[word] = seen_count.get(word, 0) + 1
            if seen_count[word] <= max_repeats:
                cleaned_words.append(word)
        
        # Remove trailing repeated words specifically
        if len(cleaned_words) > 1:
            last_word = cleaned_words[-1]
            trailing_count = 1
            for i in range(len(cleaned_words)-2, -1, -1):
                if cleaned_words[i] == last_word:
                    trailing_count += 1
                else:
                    break
            # Remove trailing repeats beyond max_repeats
            if trailing_count > max_repeats:
                cleaned_words = cleaned_words[:-trailing_count]
        
        caption = " ".join(cleaned_words)
        logger.info(f"Cleaned caption: '{caption}' (removed trailing repeats)")
        return caption

