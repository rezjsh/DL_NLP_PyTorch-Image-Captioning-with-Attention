import sys
from src.entity.config_entity import TextPreprocessingConfig
import spacy
from collections import Counter
import re
from src.utils.logging_setup import logger


class TextPreprocessor:
    """
    A class to encapsulate text preprocessing, including tokenization,
    vocabulary building, and numericalization of captions.
    It applies preprocessing steps based on a provided configuration.
    """
    def __init__(self, config: TextPreprocessingConfig) -> None:
        """
        Initializes the TextPreprocessor with special tokens.
        Args:
            config (TextPreprocessingConfig): A configuration object containing text preprocessing
        """
        self.config = config
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"} # ID to word mapping
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3} # Word to ID mapping
        self.idx = 4 # Starting index for new words (after special tokens)

        logger.info(f"TextPreprocessor initialized with frequency threshold: {self.config.freq_threshold}")
        logger.info(f"Text preprocessing settings: {self.config}")

        self.spacy_eng = self.check_and_load_spacy_model()

    def tokenizer_eng(self, text: str) -> list[str]:
        """
        Tokenizes an English sentence using spaCy and applies configured preprocessing steps.
        Args:
            text (str): The input sentence.
        Returns:
            list: A list of processed tokens.
        """
        if self.config.remove_special_characters:
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
            # Replace multiple spaces with a single space
            text = re.sub(r'\s+', ' ', text).strip()
            
        doc = self.spacy_eng(text)
        
        tokens = []
        for tok in doc:
            token_text = tok.lemma_.lower() if self.config.lemmatization else tok.text.lower()
            if self.config.remove_stopwords and tok.is_stop:
                continue 
            if token_text.strip():
                    tokens.append(token_text)
        
        return tokens

    def build_vocabulary(self, sentences: list[str]) -> None:
        """
        Builds the vocabulary from a list of sentences by applying the configured
        tokenizer and frequency threshold.
        Args:
            sentences (list): A list of text sentences (captions).
        """
        counts = Counter()
        for sentence in sentences:
            for word in self.tokenizer_eng(sentence):
                counts[word] += 1
        
        initial_vocab_size = len(self.itos)
        # Add words to vocabulary if they meet the frequency threshold
        for word, count in counts.items():
            if count >= self.config.freq_threshold:
                self.stoi[word] = self.idx
                self.itos[self.idx] = word
                self.idx += 1
        logger.info(f"Vocabulary built. Added {len(self.itos) - initial_vocab_size} new words. Total vocabulary size: {len(self.itos)}")


    def numericalize(self, text: str) -> list[int]:
        """
        Converts a text sentence into a list of numerical IDs using the built vocabulary.
        Applies configured preprocessing steps via tokenizer_eng.
        Args:
            text (str): The input sentence.
        Returns:
            list: A list of numerical IDs corresponding to the tokens.
        """
        tokenized_text = self.tokenizer_eng(text)
        numerical_tokens = [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]
        logger.debug(f"Numericalized '{text[:30]}...': {numerical_tokens[:5]}...") 
        return numerical_tokens

    def decode(self, numerical_tokens: list[int]) -> str:
        """
        Converts a list of numerical IDs back into a text sentence using the vocabulary.
        Args:
            numerical_tokens (list): A list of numerical IDs.
        Returns:
            str: The reconstructed text sentence.
        """
        # Map numerical IDs back to words
        words = [self.itos[idx] for idx in numerical_tokens if idx in self.itos]
        return " ".join(words)

    def check_and_load_spacy_model(self) -> spacy.language.Language:
        """
        Checks if the spaCy model is loaded and ready for use.
        """
        try:
            spacy_eng = spacy.load("en_core_web_sm")
            logger.info("spaCy 'en_core_web_sm' model loaded.")
        except OSError:
            logger.warning("spaCy 'en_core_web_sm' model not found. Attempting download...", exc_info=True)
            try:
                # Use subprocess.run for more controlled execution and error handling
                import subprocess
                subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
                spacy_eng = spacy.load("en_core_web_sm")
                logger.info("spaCy 'en_core_web_sm' model downloaded and loaded successfully.")
                return spacy_eng
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to download spaCy model via subprocess: {e}", exc_info=True)
                logger.error("Please try running 'python -m spacy download en_core_web_sm' manually.")
                sys.exit(1)
            except Exception as e:
                logger.error(f"An unexpected error occurred during spaCy model loading: {e}", exc_info=True)
                sys.exit(1)

    def __len__(self) -> int:
        """Returns the current size of the vocabulary."""
        return len(self.itos)
