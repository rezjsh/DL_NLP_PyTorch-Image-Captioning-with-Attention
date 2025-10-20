from pathlib import Path
from src.constants.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.entity.config_entity import DecoderConfig, EncoderConfig, ImagePreprocessingConfig, TextPreprocessingConfig, DataIngestionConfig, DataValidationConfig, DataLoaderConfig, CaptioningDatasetConfig, EncoderDecoderConfig, ModelTrainingConfig
from src.utils.helpers import create_directory, read_yaml_file
from src.utils.logging_setup import logger
from src.core.singlton import SingletonMeta

class ConfigurationManager(metaclass=SingletonMeta):
    def __init__(self, config_file_path: str = CONFIG_FILE_PATH, params_file_path: str = PARAMS_FILE_PATH):
        self.config = read_yaml_file(config_file_path)
        self.params = read_yaml_file(params_file_path)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        logger.info("Getting data ingestion config")
        config = self.config.dataset
        logger.info(f"Data ingestion config: {config}")
        
        dirs_to_create = [config.download_dir]
        logger.info(f"Dirs to create: {dirs_to_create}")
        create_directory(dirs_to_create)
        logger.info("Creating data ingestion config")

        data_ingestion_config = DataIngestionConfig(
            download_dir=config.download_dir,
            images_zip_url=config.images_zip_url,
            captions_url=config.captions_url,
            images_zip_name=config.images_zip_name,
            captions_zip_name=config.captions_zip_name,
            captions_file_name=config.captions_file_name
        )
        logger.info(f"Data ingestion config created: {data_ingestion_config}")
        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        logger.info("Getting data validation config")
        config = self.config.validation

        dirs_to_create = [Path(config.validation_report_file).parent]
        create_directory(dirs_to_create)

        data_validation_config = DataValidationConfig(
            dataset_base_dir=config.dataset_base_dir,
            images_dir=config.images_dir,
            captions_file=config.captions_file,
            image_extensions=config.image_extensions,
            validation_report_file=config.validation_report_file
        )
        logger.info(f"Data validation config created: {data_validation_config}")

        return data_validation_config

    def get_text_preprocessing_config(self) -> TextPreprocessingConfig:
        logger.info("Getting text preprocessing config")
        params = self.params.text_preprocessing
        logger.info(f"Text preprocessing config: {params}")

        text_preprocessing_config = TextPreprocessingConfig(
            remove_special_characters=params.remove_special_characters,
            lowercase=params.lowercase,
            remove_stopwords=params.remove_stopwords,
            lemmatization=params.lemmatization,
            freq_threshold=params.freq_threshold
        )
        logger.info(f"Text preprocessing config created: {text_preprocessing_config}")

        return text_preprocessing_config

    def get_image_preprocessing_config(self) -> ImagePreprocessingConfig:
        logger.info("Getting image preprocessing config")
        params = self.params.image_preprocessing
        logger.info(f"Image preprocessing config: {params}")

        image_preprocessing_config = ImagePreprocessingConfig(
            resize_size=params.resize_size,
            random_crop_size=params.random_crop_size,
            normalize_mean=params.normalize_mean,
            normalize_std=params.normalize_std
        )
        logger.info(f"Image preprocessing config created: {image_preprocessing_config}")

        return image_preprocessing_config

    def get_dataset_config(self) -> CaptioningDatasetConfig:
        logger.info("Getting dataset config")
        config = self.config.create_dataset
        logger.info(f"Dataset config: {config}")

        dataset_config = CaptioningDatasetConfig(
            dataset_base_dir=Path(config.dataset_base_dir),
            train_images_file=config.train_images_file,
            dev_images_file=config.dev_images_file,
            test_images_file=config.test_images_file,
            images_dir=Path(config.images_dir),
            caption_path=Path(config.caption_path)
        )
        logger.info(f"Dataset config created: {dataset_config}")

        return dataset_config
    
    def get_data_loader_config(self) -> DataLoaderConfig:
        logger.info("Getting data loader config")
        params = self.params.data_loader
        logger.info(f"Data loader config params: {params}")

        data_loader_config = DataLoaderConfig(
            batch_size=params.batch_size,
            shuffle=params.shuffle,
            num_workers=params.num_workers,
            pin_memory=params.pin_memory,
            drop_last=params.drop_last
        )
        logger.info(f"DataLoaderConfig config created: {data_loader_config}")
        return data_loader_config

    def get_decoder_config(self) -> DecoderConfig:
        logger.info("Getting Decoder config")
        params = self.params.decoder
        logger.info(f"Decoder config: {params}")

        decoder_config = DecoderConfig(
            vocab_size=params.vocab_size,
            d_model=params.d_model,
            max_len=params.max_len,
            dropout=params.dropout,
            ff_dim=params.ff_dim,
            num_heads=params.num_heads,
            num_transformer_layers=params.num_transformer_layers,

        )
        logger.info(f"Decoder config created: {decoder_config}")

        return decoder_config

    def get_encoder_config(self) -> EncoderConfig:
        logger.info("Getting Encoder config")
        params = self.params.encoder
        logger.info(f"Encoder config: {params}")

        encoder_config = EncoderConfig(
            cnn_model_name=params.cnn_model_name,
            embed_dim=params.embed_dim,
            num_transformer_layers=params.num_transformer_layers,
            num_heads=params.num_heads,
            ff_dim=params.ff_dim,
            dropout=params.dropout,
            fine_tune_cnn=params.fine_tune_cnn,
            fine_tune_transformer=params.fine_tune_transformer
        )
        logger.info(f"Encoder config created: {encoder_config}")
        return encoder_config

    def get_encoder_decoder_config(self) -> EncoderDecoderConfig:
        logger.info("Getting EncoderDecoder config")
        params = self.params.encoder_decoder
        logger.info(f"EncoderDecoderConfig config: {params}")

        encoder_decoder_config = EncoderDecoderConfig(
            vocab_size=params.vocab_size,
            encoder_type=params.encoder_type,
            cnn_backbone=params.cnn_backbone,
            d_model=params.d_model,
            fine_tune_cnn=params.fine_tune_cnn,
            max_caption_length=params.max_caption_length
        )
        logger.info(f"EncoderDecoderConfig config created: {encoder_decoder_config}")

        return encoder_decoder_config
    
    def get_model_trainer_config(self) -> ModelTrainingConfig:
        logger.info("Getting model training config")
        config = self.config.training
        params = self.params.training
        logger.info(f"Model training config params: {params}")

        dirs_to_create= [config.model_dir, config.report_dir]
        create_directory(dirs_to_create)

        model_training_config = ModelTrainingConfig(
            model_dir=Path(config.model_dir),
            model_name=config.model_name,
            report_dir=Path(config.report_dir),
            model_save_prefix=config.model_save_prefix,
            num_epochs=params.num_epochs,
            learning_rate=params.learning_rate,
            weight_decay=float(params.weight_decay),
            save_every_epochs=params.save_every_epochs,
            early_stop_patience=params.early_stop_patience,
            gradient_accumulation_steps=params.gradient_accumulation_steps,
            clip_grad_norm=params.clip_grad_norm
        )
        logger.info(f"ModelTrainingConfig config created: {model_training_config}")
        return model_training_config
