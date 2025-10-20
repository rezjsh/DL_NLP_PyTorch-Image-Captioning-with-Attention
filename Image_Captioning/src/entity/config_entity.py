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
class TextPreprocessingConfig:
    freq_threshold: int
    remove_special_characters: bool
    lowercase: bool
    remove_stopwords: bool
    lemmatization: bool

@dataclass
class ImagePreprocessingConfig:
    resize_size: list[int]
    random_crop_size: list[int]
    normalize_mean: list[float]
    normalize_std: list[float]



@dataclass
class CaptioningDatasetConfig:
    images_dir: Path
    caption_path: Path
    dataset_base_dir: Path        
    train_images_file: str       
    dev_images_file: str         
    test_images_file: str
   

@dataclass
class DataLoaderConfig:
    batch_size: int
    shuffle: bool
    num_workers: int
    pin_memory: bool
    drop_last: bool


@dataclass
class EncoderConfig:
    cnn_model_name: str
    embed_dim: int 
    num_transformer_layers: int
    num_heads: int
    ff_dim: int
    dropout: float
    fine_tune_cnn: bool
    fine_tune_transformer: bool

@dataclass
class DecoderConfig:
    vocab_size: int
    dropout: float
    d_model: int
    max_len: int
    num_heads: int
    ff_dim: int
    num_transformer_layers: int


@dataclass
class EncoderDecoderConfig:
    encoder_type: str
    cnn_backbone: str
    fine_tune_cnn: bool
    d_model: int
    max_caption_length: int
    vocab_size: int



@dataclass
class ModelTrainingConfig:
    model_dir: Path
    model_name: str
    report_dir: Path
    model_save_prefix: str
    num_epochs: int
    learning_rate: float
    weight_decay: float
    save_every_epochs: int
    early_stop_patience: int
    gradient_accumulation_steps: int
    clip_grad_norm: float