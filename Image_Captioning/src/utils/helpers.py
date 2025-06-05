import json
import logging
import os
from box import ConfigBox
import yaml


def read_yaml_file(file_path: str)-> ConfigBox:
    """Read a YAML file and return its content as a ConfigBox object"""
    try:
        logging.info(f"Reading YAML file: {file_path}")
        with open(file_path, 'r') as file:
            content = yaml.safe_load(file)
            logging.info(f"YAML file {file_path} loaded successfully")
        return ConfigBox(content)
    except Exception as e:
        logging.error(f"Error reading YAML file {file_path}: {e}")
        raise e

def create_directory(dirs: list)-> None:
    """Create a list of directories"""
    try:
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Created directory: {dir_path}")
    except Exception as e:
        logging.error(f"Error creating directory {dir_path}: {e}")
        raise e

def save_json(file_path: str, content: dict)-> None:
    """Save a dictionary to a JSON file"""
    try:
        with open(file_path, 'w') as file:
            json.dump(content, file, indent=4)
        logging.info(f"JSON file saved to: {file_path}")
    except Exception as e:
        logging.error(f"Error saving JSON file to {file_path}: {e}")
        raise e
