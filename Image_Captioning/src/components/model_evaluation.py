import torch
import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Any, List, Dict

# Detect device automatically
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Metric Calculation Placeholder ---
def calculate_metrics(references: List[List[str]], hypotheses: List[str]) -> Dict[str, float]:
    """
    Computes evaluation metrics.
    Args:
        references: List of list of strings (ground truth captions).
        hypotheses: List of strings (generated captions).
    """
    # Placeholder: Average length calculation
    avg_ref_len = sum(len(ref[0].split()) for ref in references) / len(references) if references else 0
    avg_hyp_len = sum(len(hyp.split()) for hyp in hypotheses) / len(hypotheses) if hypotheses else 0

    return {
        "Num_Samples": len(hypotheses),
        "Avg_Reference_Length": round(avg_ref_len, 2),
        "Avg_Hypothesis_Length": round(avg_hyp_len, 2),
        "BLEU_4_Score": 0.0, # Replace with actual NLTK/COCO calls
        "CIDEr_Score": 0.0   # Replace with actual NLTK/COCO calls
    }
# --------------------------------------

class ModelEvaluator:
    """
    Handles evaluation of the Image Captioning Model using efficient Batch Inference.
    """
    def __init__(self, config: Any, model: Any, test_loader: DataLoader, text_preprocessor: Any) -> None:
        self.config = config
        self.model = model.to(DEVICE)
        self.test_loader = test_loader
        self.vocab = text_preprocessor

        # Create output directory
        os.makedirs(self.config.report_dir, exist_ok=True)
        self._load_model_weights()

    def _load_model_weights(self):
        """Loads weights, handling potential key prefixes from DataParallel."""
        model_path = self.config.model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at: {model_path}")

        try:
            state_dict = torch.load(model_path, map_location=DEVICE)

            # clean keys
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace('_orig_mod.', '').replace('module.', '')
                new_state_dict[k] = v

            self.model.load_state_dict(new_state_dict, strict=True)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Error loading model weights: {e}")

    @torch.no_grad()
    def evaluate(self) -> dict:
        """
        Runs batch inference on the test set.
        Significantly faster than single-image iteration.
        """
        hypotheses: List[str] = []
        references: List[List[str]] = []

        self.model.eval()

        print(f"Starting evaluation on {DEVICE}...")

        for images, captions_tensor, lengths in tqdm(self.test_loader, desc="Evaluating"):

            # 1. Move batch to GPU
            images = images.to(DEVICE)
            batch_size = images.size(0)

            # 2. Run Batch Inference
            # Checks if the optimized method exists, otherwise falls back to a slower loop
            if hasattr(self.model, 'batch_generate_caption'):
                batch_words, _ = self.model.batch_generate_caption(
                    image_tensor=images,
                    vocab=self.vocab,
                    beam_size=self.config.beam_size,
                    max_len=self.config.max_len if hasattr(self.config, 'max_len') else 20
                )
            else:
                # FAST FALLBACK: Loop on GPU (if batch method not implemented)
                batch_words = []
                for i in range(batch_size):
                    # Unsqueeze creates (1, C, H, W) without moving memory to CPU
                    words, _ = self.model.generate_caption(
                        images[i].unsqueeze(0),
                        self.vocab,
                        self.config.beam_size
                    )
                    batch_words.append(words)

            # 3. Process Results
            # Join words into sentences
            hypotheses.extend([" ".join(words) for words in batch_words])

            # 4. Process References (Ground Truth)
            for i in range(batch_size):
                numericalized_ref = captions_tensor[i]

                # Filter <PAD>
                ref_ids = [idx.item() for idx in numericalized_ref if idx.item() != self.vocab.stoi["<PAD>"]]

                # Convert to string
                raw_ref = self.vocab.denumericalize(ref_ids)

                # Clean <SOS>/<EOS>
                clean_ref = " ".join([w for w in raw_ref.split() if w not in ["<SOS>", "<EOS>"]])

                references.append([clean_ref])

        # --- Metrics & Saving ---
        if len(hypotheses) != len(references):
            print(f"Warning: Size mismatch. Hypotheses: {len(hypotheses)}, References: {len(references)}")
            metrics = {}
        else:
            metrics = calculate_metrics(references, hypotheses)

        # Save CSV
        report_data = {
            "Generated_Caption": hypotheses,
            "Reference_Captions": [" ||| ".join(refs) for refs in references]
        }
        df = pd.DataFrame(report_data)
        report_path = os.path.join(self.config.report_dir, "evaluation_results.csv")
        df.to_csv(report_path, index=False)

        print(f"Evaluation complete. Results saved to {report_path}")
        return {"metrics": metrics, "report_path": report_path}