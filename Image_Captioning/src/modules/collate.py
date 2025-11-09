from src.utils.logging_setup import logger
import torch


class MyCollate:
    """
    Custom collate function for DataLoader to handle variable-length captions
    by padding them to the maximum length within each batch.
    """
    def __init__(self, pad_idx: int) -> None:
        """
        Initializes the custom collate function.

        Args:
            pad_idx (int): The vocabulary ID for the padding token.
        """
        self.pad_idx = pad_idx
        logger.info(f"MyCollate initialized with padding index: {pad_idx}")


    def __call__(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collates a list of samples into a batch.

        Filters out any samples that might have failed to load and pads captions
        to the maximum length within the batch.

        Args:
            batch (list): A list of tuples, where each tuple contains
                          (image_tensor, caption_tensor).

        Returns:
            tuple: A tuple containing:
                   - images_tensor (torch.Tensor): A batch of image tensors (batch_size, C, H, W).
                   - padded_captions_tensor (torch.Tensor): A batch of padded caption tensors (batch_size, max_seq_len).
                   - caption_lengths_tensor (torch.Tensor): A tensor of original caption lengths (batch_size,).
                   Returns (None, None, None) if the batch is empty after filtering.
        """
        # Filter out any samples that might have failed to load (e.g., image not found from __getitem__)
        # This ensures that only valid samples are processed.
        batch = [item for item in batch if item is not None]

        if not batch:
            logger.warning("Batch is empty after filtering out failed samples. Returning None for all outputs.")
            return None, None, None

        # Separate images and captions from the batch
        # imgs will be a list of individual image tensors (C, H, W)
        # targets will be a list of individual caption tensors (seq_len,)
        imgs = [item[0] for item in batch]
        targets = [item[1] for item in batch]

        # Stack images along a new batch dimension to create a single tensor
        # Resulting shape: (batch_size, C, H, W)
        imgs_tensor = torch.stack(imgs, dim=0)

        # Get original lengths of captions before padding.
        # These lengths are crucial for packed padded sequences or loss calculations.
        lengths_tensor = torch.tensor([len(cap) for cap in targets])

        # Pad captions to the longest sequence in the current batch.
        # `batch_first=True` makes the output shape (batch_size, max_seq_len).
        # `padding_value` is the ID for the padding token.
        padded_captions_tensor = torch.nn.utils.rnn.pad_sequence(
            targets, batch_first=True, padding_value=self.pad_idx
        )

        logger.info(f"Collated batch shapes: Images {imgs_tensor.shape}, Captions {padded_captions_tensor.shape}")

        return imgs_tensor, padded_captions_tensor, lengths_tensor