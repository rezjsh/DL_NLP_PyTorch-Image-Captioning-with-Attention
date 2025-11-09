# src/components/text_preprocessing.py
import sys
from src.entity.config_entity import TextPreprocessingConfig
import spacy
from collections import Counter
import re
from src.utils.logging_setup import logger
import json 

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
            text (str): The input text sentence.
        Returns:
            list[str]: A list of processed tokens.
        """
        if not isinstance(text, str):
            logger.warning(f"Input to tokenizer_eng is not a string: {type(text)}. Attempting conversion.")
            text = str(text) # Attempt to convert to string

        text = text.lower() if self.config.lowercase else text # Convert to lowercase
        if self.config.remove_special_characters:
            text = re.sub(r'[^\w\s]', '', text) # Remove non-alphanumeric characters

        # Tokenize
        tokens = [token.text for token in self.spacy_eng.tokenizer(text)]

        # Remove stopwords
        if self.config.remove_stopwords:
            # spaCy's default stop words list
            tokens = [token for token in tokens if token not in self.spacy_eng.Defaults.stop_words]

        # Perform lemmatization
        if self.config.lemmatization:
            # Re-process with nlp to get lemma_ attribute
            doc = self.spacy_eng(" ".join(tokens))
            tokens = [token.lemma_ for token in doc]

        return tokens


    def build_vocabulary(self, sentences: list[str]) -> None:
        """
        Builds the vocabulary from a list of sentences.
        Args:
            sentences (list[str]): A list of raw caption sentences.
        """
        logger.info("Building vocabulary...")
        counts = Counter()
        for sentence in sentences:
            # Ensure sentence is a string before tokenizing
            if not isinstance(sentence, str):
                logger.warning(f"Skipping non-string sentence during vocab build: {type(sentence)} - {sentence}")
                continue
            tokens = self.tokenizer_eng(sentence)
            counts.update(tokens)

        # Add words to vocabulary based on frequency threshold
        for word, count in counts.items():
            if count >= self.config.freq_threshold:
                self.stoi[word] = self.idx
                self.itos[self.idx] = word
                self.idx += 1
        logger.info(f"Vocabulary built. Total unique words: {len(self.stoi)}")
        logger.debug(f"Vocabulary STOI size: {len(self.stoi)}, ITOS size: {len(self.itos)}")
        # logger.debug(f"Sample ITOS: {list(self.itos.items())[:10]}")
        # logger.debug(f"Sample STOI: {list(self.stoi.items())[:10]}")


    def numericalize(self, text: str) -> list[int]:
        """
        Converts a sentence into a list of numericalized tokens.
        Args:
            text (str): The input text sentence.
        Returns:
            list[int]: A list of token IDs.
        """
        tokenized_text = self.tokenizer_eng(text)
        # Use .get() with default <UNK> for safer lookup
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]


    def denumericalize(self, tokens: list[int]) -> str:
        """
        Converts a list of numericalized tokens back into a readable sentence.
        Args:
            tokens (list[int]): A list of token IDs.
        Returns:
            str: The reconstructed sentence.
        """
        # Use .get() with default <UNK> for safer lookup
        words = [self.itos.get(idx, "<UNK>") for idx in tokens]
        return " ".join(words)

    def __len__(self) -> int:
        return len(self.stoi)

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
        return spacy_eng

    def save_vocab(self, path: str) -> None:
        """Saves the vocabulary (stoi and itos) to a JSON file."""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({'stoi': self.stoi, 'itos': self.itos}, f, indent=4)
            logger.info(f"Vocabulary saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save vocabulary to {path}: {e}")
            raise

    def load_vocab(self, path: str) -> None:
        """Loads the vocabulary (stoi and itos) from a JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Ensure keys in itos are integers
                self.stoi = data['stoi']
                self.itos = {int(k): v for k, v in data['itos'].items()}
                # Update idx to be the next available index
                self.idx = max(self.itos.keys()) + 1 if self.itos else 4
            logger.info(f"Vocabulary loaded from {path}. Vocabulary size: {len(self.stoi)}")
        except FileNotFoundError:
            logger.warning(f"Vocabulary file not found at {path}. A new vocabulary will be built if build_vocabulary is called.")
            # Reset to initial state if file not found
            self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
            self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
            self.idx = 4
        except Exception as e:
            logger.error(f"Failed to load vocabulary from {path}: {e}")
            raise