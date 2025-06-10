from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
    download_dir: str 
    images_zip_url: str 
    captions_url: str 
    images_zip_name: str 
    captions_zip_name: str 
    captions_file_name: str 

@dataclass
class DataValidationConfig:
    dataset_base_dir: Path
    images_dir: Path
    captions_file: str 
    image_extensions: list[str]
    validation_report_file: Path

@dataclass
class ModelTrainingConfig:
    model_name: str 
    model_path: str 
    model_type: str 
    model_weights: str 
