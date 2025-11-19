from src.utils.logging_setup import logger
from src.config.configuration import ConfigurationManager
from src.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from src.pipeline.stage_02_data_validation import ValidationPipeline
from src.pipeline.stage_03_text_preprocessing import TextPreprocessingPipeline
from src.pipeline.stage_04_image_preprocessing import ImageProcessingPipeline
from src.pipeline.stage_05_dataset import DatasetPipeline
from src.pipeline.stage_06_data_loader import DataLoaderPipeline
from src.pipeline.stage_07_encoder import EncoderPipeline
from src.pipeline.stage_08_decoder import DecoderPipeline
from src.pipeline.stage_09_encoder_decoder import EncoderDecoderPipeline
from src.pipeline.stage_10_model_trainer import TrainingPipeline
from src.pipeline.stage_11_model_evaluation import EvaluationPipeline


import torch
torch.cuda.empty_cache()
if __name__ == '__main__':
    try:
        config_manager = ConfigurationManager()

        # --- Data Ingestion Stage ---
        STAGE_NAME = "Data Ingestion Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_ingestion_pipeline = DataIngestionPipeline(config_manager) # Pass as positional argument
        data_ingestion_pipeline.run_pipeline()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        # --- Data Validation Stage ---
        STAGE_NAME = "Data Validation Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_validation_pipeline = ValidationPipeline(config_manager) # Pass as positional argument
        data_validation_pipeline.run_pipeline()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        # --- Text Preprocessing Stage ---
        STAGE_NAME = "Text Preprocessing Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        text_preprocessing_pipeline = TextPreprocessingPipeline(config_manager) # Pass as positional argument
        text_preprocessor = text_preprocessing_pipeline.run_pipeline()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        # --- Image Preprocessing Stage ---
        STAGE_NAME = "Image Preprocessing Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        image_processing_pipeline = ImageProcessingPipeline(config_manager) # Pass as positional argument
        image_processor = image_processing_pipeline.run_pipeline()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        # --- Dataset Preparation Stage ---
        STAGE_NAME = "Dataset Preparation Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        dataset_pipeline = DatasetPipeline(config_manager) # Pass as positional argument
        datasets = dataset_pipeline.run_pipeline(text_preprocessor, image_processor)
        # datasets is now {'train': ..., 'dev': ..., 'test': ...}
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        # --- DataLoader Stage (Creates all 3 loaders) ---
        STAGE_NAME = "DataLoader Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_loader_pipeline = DataLoaderPipeline(config_manager) # Pass as positional argument
        loaders = data_loader_pipeline.run_pipeline(datasets, text_preprocessor)
        # loaders is now {'train_loader': ..., 'val_loader': ..., 'test_loader': ...}

        # Unpack loaders for use in the training stage
        train_loader = loaders['train_loader']
        val_loader = loaders['val_loader']
        test_loader = loaders['test_loader']

        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        # --- Encoder Stage ---
        STAGE_NAME = "Encoder Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        encoder_pipeline = EncoderPipeline(config_manager) # Pass as positional argument
        encoder = encoder_pipeline.run_pipeline()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        # --- Decoder Stage ---
        STAGE_NAME = "Decoder Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        decoder_pipeline = DecoderPipeline(config_manager, vocab_size=len(text_preprocessor)) # Pass as positional argument, keep vocab_size keyword
        decoder = decoder_pipeline.run_pipeline()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        # --- Encoder-Decoder Stage ---
        STAGE_NAME = "Encoder-Decoder Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        encoder_decoder_pipeline = EncoderDecoderPipeline(config_manager) # Pass as positional argument
        model = encoder_decoder_pipeline.run_pipeline(vocab_size=len(text_preprocessor), transformer_encoder=encoder, transformer_decoder=decoder) # Keep other arguments as keywords
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")


        # --- Model Training Stage ---
        STAGE_NAME = "Model Training Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        trainer = TrainingPipeline(config_manager) # Pass as positional argument
        trainer.run_pipeline(model=model, train_loader=train_loader, val_loader=val_loader, text_preprocessor=text_preprocessor) # Keep other arguments as keywords
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")


        # --- Model Evaluation Stage ---
        STAGE_NAME = "Model Evaluation Stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        evaluation_pipeline = EvaluationPipeline(config=config_manager)
        evaluation_metrics = evaluation_pipeline.run_pipeline(
            model=model,
            test_loader=test_loader,
            text_preprocessor=text_preprocessor
        )
        logger.info(f"Evaluation metrics: {evaluation_metrics}")
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")



    except Exception as e:
        logger.error(f"Error occurred during {STAGE_NAME} stage: {e}")
        raise e