# src/pipeline/stage_11_model_evaluation.py
from typing import Dict
from src.utils.logging_setup import logger
from src.config.configuration import ConfigurationManager
from src.components.model_evaluation import  ModelEvaluator 
from src.components.text_preprocessing import TextPreprocessor
from src.components.data_loader import MyDataLoader
from src.components.encoder_decoder import TransformerImageCaptioningModel

class EvaluationPipeline:
    """
    Pipeline stage for model evaluation on the test set.
    """
    def __init__(self, config: ConfigurationManager) -> None:
        self.config_manager = config
        self.eval_config = self.config_manager.get_model_evaluation_config()

    def run_pipeline(self, 
                     model: TransformerImageCaptioningModel, test_loader: MyDataLoader,
                     text_preprocessor: TextPreprocessor) -> Dict[str, float]:
        """
        Runs the evaluation pipeline by creating the test data loader and running the evaluator.
        
        Args:
            model (TransformerImageCaptioningModel): The final, trained model instance.
            text_preprocessor (TextPreprocessor): The trained TextPreprocessor (for vocabulary).
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        try:
            logger.info("Starting model evaluation pipeline...")
            evaluator = ModelEvaluator(
                config=self.eval_config,
                model=model,
                test_loader=test_loader,
                text_preprocessor=text_preprocessor,
            )
            
            metrics = evaluator.evaluate()
            
            logger.info("Evaluation pipeline completed successfully.")
            return metrics
            
        except Exception as e:
            logger.error(f"Error in evaluation pipeline: {e}")
            raise e



# class Evaluator:
#     """
#     A class to encapsulate evaluation and inference logic for the image captioning model.
#     It takes a trained model and preprocessors to generate captions for new images.
#     """
#     def __init__(self, model, text_preprocessor, device=None):
#         """
#         Initializes the Evaluator.
#         Args:
#             model (nn.Module): The trained captioning model (e.g., EncoderDecoder instance).
#             text_preprocessor (object): The TextPreprocessor instance (for vocabulary lookups).
#             device (torch.device, optional): The device (CPU/GPU) to run the model on.
#                                              Defaults to global DEVICE if None.
#         """
#         self.model = model
#         self.text_preprocessor = text_preprocessor
#         self.device = device # Use global DEVICE if not specified

#         # Ensure the model is on the correct device
#         self.model.to(self.device)
#         self.model.eval() # Set model to evaluation mode (important for inference)

#         logger.info(f"Evaluator initialized. Model set to evaluation mode on device: {self.device}.")

#     def generate_caption(self, image_path: str, beam_size: int = 3, image_transform: transforms.Compose = None):
#         """
#         Generates a caption for a single image using the trained model.
#         Args:
#             image_path (str): Path to the input image file.
#             beam_size (int): The number of sequences to consider in beam search.
#             image_transform (transforms.Compose): Image transformation pipeline required for inference.
#         Returns:
#             list: List of generated words (excluding special tokens), or an empty list if generation fails.
#         """
#         if image_transform is None:
#             logger.error("Image transform pipeline is required for inference. Cannot generate caption.")
#             return []

#         # Load and preprocess image
#         try:
#             img = Image.open(image_path).convert("RGB")
#             # Apply transformation and add a batch dimension (for a single image)
#             img_tensor = image_transform(img).unsqueeze(0).to(self.device)
#         except FileNotFoundError:
#             logger.error(f"Error: Image file not found at {image_path}", exc_info=True)
#             return []
#         except Exception as e:
#             logger.error(f"Error loading or transforming image {image_path}: {e}", exc_info=True)
#             return []

#         logger.info(f"Generating caption for image: {os.path.basename(image_path)} using beam size {beam_size}...")

#         with torch.no_grad(): # Disable gradient calculations for inference
#             # The generate_caption method is part of EncoderDecoder.decoder
#             # It expects the text_preprocessor (vocab) for token conversion
#             caption_words, attention_alphas = self.model.generate_caption(img_tensor, self.text_preprocessor, beam_size)

#         return caption_words
