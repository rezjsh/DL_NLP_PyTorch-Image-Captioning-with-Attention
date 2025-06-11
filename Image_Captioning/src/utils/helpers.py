import json
import os
import zipfile
from box import ConfigBox
import requests
import yaml
from src.utils.logging_setup import logger

def read_yaml_file(file_path: str)-> ConfigBox:
    """Read a YAML file and return its content as a ConfigBox object"""
    try:
        logger.info(f"Reading YAML file: {file_path}")
        with open(file_path, 'r') as file:
            content = yaml.safe_load(file)
            logger.info(f"YAML file {file_path} loaded successfully")
        return ConfigBox(content)
    except Exception as e:
        logger.error(f"Error reading YAML file {file_path}: {e}")
        raise e

def create_directory(dirs: list)-> None:
    """Create a list of directories"""
    try:
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    except Exception as e:
        logger.error(f"Error creating directory {dir_path}: {e}")
        raise e

def save_json(file_path: str, content: dict)-> None:
    """Save a dictionary to a JSON file"""
    try:
        with open(file_path, 'w') as file:
            json.dump(content, file, indent=4)
        logger.info(f"JSON file saved to: {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON file to {file_path}: {e}")
        raise e

def download_file(url: str, filename: str) -> bool:
    """
    Downloads a file from a given URL.
    Args:
        url (str): The URL of the file to download.
        filename (str): The local filename to save the downloaded content.
    Returns:
        bool: True if download is successful, False otherwise.
    """
    logger.info(f"Downloading {filename} from {url}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info(f"Successfully downloaded {filename}.")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading {filename}: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while downloading {filename}: {e}")
        return False
    

def extract_zip(zip_path: str, extract_to: str) -> bool:
        """
        Extracts a zip file to a specified directory.
        Args:
            zip_path (str): Path to the zip file.
            extract_to (str): Directory where contents will be extracted.
        Returns:
            bool: True if extraction is successful, False otherwise.
        """
        logger.info(f"Extracting {zip_path} to {extract_to}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            logger.info(f"Successfully extracted {zip_path}.")
            return True
        except zipfile.BadZipFile:
            logger.error(f"Error: {zip_path} is not a valid zip file.")
            return False
        except Exception as e:
            logger.error(f"Error extracting {zip_path}: {e}")
            return False