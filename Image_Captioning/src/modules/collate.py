from src.utils.logging_setup import logger
import torch


class MyCollate:
    """
    Custom collate function for DataLoader to handle variable-length captions
    by padding them to the maximum length within each batch.
    """
    def __init__(self, pad_idx: int) -> None:
        """
        Args:
            pad_idx (int): The vocabulary ID for the padding token.
        """
        self.pad_idx = pad_idx
        logger.info(f"MyCollate initialized with padding index: {pad_idx}")


    def __call__(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collates a list of samples into a batch.
        Args:
            batch (list): A list of tuples (image, caption_tensor).
        Returns:
            tuple: (images_tensor, padded_captions_tensor, caption_lengths_tensor)
                   Returns (None, None, None) if the batch is empty after filtering.
        """
        # Filter out any samples that might have failed to load (e.g., image not found from __getitem__)
        batch = [item for item in batch if item is not None]
        if not batch:
            logger.warning("Batch is empty after filtering out failed samples.")
            return None, None, None 

        # Stack images (each item[0] is an image tensor, unsqueeze(0) adds batch dim)
        imgs = [item[0].unsqueeze(0) for item in batch] # imgs will be (batch_size, C, H, W)
        
        imgs = torch.cat(imgs, dim=0)
        
        # Get caption tensors and their original lengths
        targets = [item[1] for item in batch]
        lengths = [len(cap) for cap in targets] # Store original lengths for correct loss calculation

        # Pad captions to the longest sequence in the batch
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        
        logger.debug(f"Collated batch shapes: Images {imgs.shape}, Captions {targets.shape}, Lengths {len(lengths)}")
        return imgs, targets, torch.tensor(lengths)
