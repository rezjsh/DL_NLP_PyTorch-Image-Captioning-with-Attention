import logging
import os
import sys
from src.core.singleton import SingletonMeta

class Logger(metaclass=SingletonMeta):
    """
    A singleton class for managing application-wide logging.
    It configures and provides a pre-configured logger instance.
    """
    def __init__(self, logger_name="Image_Captioning", log_dir="logs", log_file_name="running_logs.log"):
        """
        Initializes the Logger. This constructor will only be called once
        due to the SingletonMeta.

        Args:
            logger_name (str): The name of the logger to be used.
            log_dir (str): The directory where log files will be stored.
            log_file_name (str): The name of the log file.
        """
        self._logger = None
        self._logger_name = logger_name
        self._log_dir = log_dir
        self._log_file_name = log_file_name
        self._setup_logging()

    def _setup_logging(self):
        """
        Configures the logging system.
        """
        logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
        log_filepath = os.path.join(self._log_dir, self._log_file_name)
        os.makedirs(self._log_dir, exist_ok=True)

        # Basic configuration applied only once
        if not self._logger: # Check if logger is already set up (important for potential re-init calls if not truly singleton)
            logging.basicConfig(
                level=logging.INFO,
                format=logging_str,
                handlers=[
                    logging.FileHandler(log_filepath),
                    logging.StreamHandler(sys.stdout)
                ]
            )
            self._logger = logging.getLogger(self._logger_name)
            self._logger.info("Logging initialized successfully.")

    @property
    def logger(self):
        """
        Provides access to the configured logger instance.
        """
        return self._logger

logger = Logger().logger
