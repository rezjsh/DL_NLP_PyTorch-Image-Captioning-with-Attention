import logging
import os
import sys
from src.core.singleton import SingletonMeta

class Logger(metaclass=SingletonMeta):
    """
    A singleton class for managing application-wide logging.
    It configures and provides a pre-configured logger instance.
    Ensures that only one logger instance is created and configured.
    """
    def __init__(self, logger_name: str = "Image_Captioning",
                 log_dir: str = "logs",
                 log_file_name: str = "running_logs.log",
                 log_level: int = logging.INFO): # Make log level configurable
        """
        Initializes the Logger. This constructor will only be called once
        due to the SingletonMeta.

        Args:
            logger_name (str): The name of the logger to be used.
            log_dir (str): The directory where log files will be stored.
            log_file_name (str): The name of the log file.
            log_level (int): The minimum logging level (e.g., logging.DEBUG, logging.INFO).
        """
        self._logger = logging.getLogger(logger_name)

        # Remove all existing handlers to prevent duplicate log messages
        # This ensures that even if the initialization is called multiple times
        # handlers are not duplicated.
        for handler in list(self._logger.handlers):
            self._logger.removeHandler(handler)

        self._logger.setLevel(log_level) # Set the overall minimum level for the logger

        # Define the formatter for all handlers
        logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
        formatter = logging.Formatter(logging_str)

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        log_filepath = os.path.join(log_dir, log_file_name)

        # File Handler for writing logs to a file
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

        # Stream Handler for printing logs to the console (sys.stdout)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        self._logger.addHandler(stream_handler)

        self._logger.info(f"Logging system initialized for '{logger_name}' at level {logging.getLevelName(log_level)}.")

    @property
    def logger(self) -> logging.Logger:
        """
        Provides access to the configured logger instance.
        """
        return self._logger

# Global instance for easy access throughout the application
# When you import 'logger' from this file, you get the same pre-configured instance.
logger = Logger().logger