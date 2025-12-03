# services/predictor_service.py
"""
Lazy-initialized singleton for the image caption predictor to be reused across requests.
Replace the placeholder implementation with a real model using your components when ready.
"""
import threading

from src.config.configuration import ConfigurationManager

config_manager = ConfigurationManager()
class _PredictorSingleton:
    _instance_lock = threading.Lock()
    _predictor = None

    @classmethod
    def get_predictor(cls):
        """Return a singleton predictor instance using the real model."""
        if cls._predictor is None:
            with cls._instance_lock:
                if cls._predictor is None:
                    from src.components.prediction import ImageCaptionPredictor
                    # Let ImageCaptionPredictor load configs/weights from ConfigurationManager
                    cls._predictor = ImageCaptionPredictor(config_manager=config_manager)
        return cls._predictor


def predict_caption(image_path: str) -> str:
    predictor = _PredictorSingleton.get_predictor()
    return predictor.predict(image_path)
