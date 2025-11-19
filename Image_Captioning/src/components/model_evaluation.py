import torch
import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Any, List
from src.utils.device import DEVICE


# Placeholder function to calculate evaluation metrics
def calculate_metrics(references: List[List[str]], hypotheses: List[str]) -> dict:
    """
    Placeholder for actual metric calculation (e.g., using pycocoevalcap, NLTK).
    References are list[list[str]] (e.g., 5 captions per image).
    Hypotheses are list[str] (1 generated caption per image).
    """
    # Simple length-based check placeholder
    avg_ref_len = sum(len(ref[0].split()) for ref in references) / len(references) if references else 0
    avg_hyp_len = sum(len(hyp.split()) for hyp in hypotheses) / len(hypotheses) if hypotheses else 0

    return {
        "Num_Samples": len(hypotheses),
        "Avg_Reference_Length": round(avg_ref_len, 2),
        "Avg_Hypothesis_Length": round(avg_hyp_len, 2),
        "BLEU_4_Score": 0.0, 
        "CIDEr_Score": 0.0    
    }
# --- End Placeholders ---


class ModelEvaluator:
    """
    Class to handle the evaluation of the trained Image Captioning Model
    on a test dataset using beam search inference.
    """
    def __init__(self, config: Any, model: Any, test_loader: DataLoader, text_preprocessor: Any) -> None:
        """
        Initializes the ModelEvaluator.
        """
        self.config = config
        self.model = model.to(DEVICE)
        self.test_loader = test_loader
        self.vocab = text_preprocessor # Vocabulary object for decoding

        # Ensure the report directory exists
        os.makedirs(self.config.report_dir, exist_ok=True)

        self._load_model_weights()
        # logger.info(f"ModelEvaluator initialized. Will evaluate on test_loader split.")


    def _load_model_weights(self):
        """
        Loads the trained model weights from the configured path,
        removing '_orig_mod.' prefix if present in the state_dict keys.
        """
        model_path = self.config.model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at: {model_path}")

        try:
            state_dict = torch.load(model_path, map_location=DEVICE)

            new_state_dict = {}
            for k, v in state_dict.items():
                # Remove the '_orig_mod.' prefix if it exists (common with torch.compile)
                if k.startswith('_orig_mod.'):
                    new_key = k[len('_orig_mod.'):]
                else:
                    new_key = k
                new_state_dict[new_key] = v

            self.model.load_state_dict(new_state_dict, strict=True)
            self.model.eval() # Set model to evaluation mode
            # logger.info(f"Successfully loaded model weights from: {model_path}")

        except Exception as e:
            raise RuntimeError(f"Error loading model weights after remapping: {e}")

    @torch.no_grad()
    def evaluate(self) -> dict:
        """
        Runs inference on the test set, generates captions, and calculates metrics.

        Assumes DataLoader yields: (images: Tensor, targets: list[list[str]], lengths: Tensor)
        where targets[i] is a list of N ground-truth string captions for image i.
        """
        # logger.info("Starting model evaluation and caption generation...")

        hypotheses = []  # Generated captions (list[str])
        references = []  # Ground truth captions (list[list[str]])

        self.model.eval() # Ensure model is in evaluation mode

        # Iterate over the test dataset
        for images, captions_tensor, lengths in tqdm(self.test_loader, desc="Generating Captions"):

            # Move images to the device
            images = images.to(DEVICE)

            # --- Inference/Caption Generation ---
            # Iterate through each image in the batch
            for i, img_tensor in enumerate(images):
                # The generate_caption method expects a single image (1, C, H, W)
                img_tensor_single = img_tensor.unsqueeze(0)

                # Call the model's generation method (using beam search)
                # Assumes self.model.generate_caption returns (list_of_words, attention_weights)
                caption_words, _ = self.model.generate_caption(
                    image_tensor=img_tensor_single,
                    vocab=self.vocab,
                    beam_size=self.config.beam_size
                )

                # Convert the list of words back to a sentence string
                generated_caption = " ".join(caption_words)
                hypotheses.append(generated_caption)

                # --- Collect References ---
                # `captions_tensor` from the DataLoader contains the numericalized, padded target caption.
                # We need to denumericalize this tensor to get a string reference.
                # `calculate_metrics` expects List[List[str]], so we provide a list containing one string reference.

                numericalized_ref_caption = captions_tensor[i] # This is a 1D torch.Tensor

                # Convert to list of integers, filtering out padding tokens
                ref_word_ids = [idx.item() for idx in numericalized_ref_caption if idx.item() != self.vocab.stoi["<PAD>"]]

                # Denumericalize the sequence of valid IDs
                # The `denumericalize` method returns a string including <SOS> and <EOS>.
                raw_ref_sentence = self.vocab.denumericalize(ref_word_ids)

                # Filter out special tokens from the string (e.g., <SOS>, <EOS>)
                # by splitting the sentence and then joining again.
                filtered_ref_sentence = " ".join([word for word in raw_ref_sentence.split() if word not in ["<SOS>", "<EOS>"]])

                # Append a list containing this single reference string to the references list
                references.append([filtered_ref_sentence])

        # --- Final Validation & Metric Calculation ---
        if not hypotheses or len(hypotheses) != len(references):
            # logger.error(f"Mismatch: Collected {len(hypotheses)} hypotheses and {len(references)} references.")
            metrics = {}
        else:
            # calculate_metrics expects list[list[str]] for references and list[str] for hypotheses
            metrics = calculate_metrics(references, hypotheses)
            # logger.info(f"Evaluation Metrics: {metrics}")

        # --- Save Report ---
        report_data = {
            "Generated_Caption": hypotheses,
            # Join multiple references with a delimiter for CSV readability
            "Reference_Captions": [" ||| ".join(refs) for refs in references]
        }

        df = pd.DataFrame(report_data)
        report_file_path = os.path.join(self.config.report_dir, "evaluation_results.csv")
        df.to_csv(report_file_path, index=False)
        # logger.info(f"Detailed evaluation results saved to: {report_file_path}")

        return {"metrics": metrics, "report_path": report_file_path}