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

