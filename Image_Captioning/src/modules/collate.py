import torch
from src.utils.logging_setup import logger
from typing import List, Tuple, Optional


class MyCollate:
    """
    Custom collate function for DataLoader to handle variable-length captions
    by padding them to the maximum length within each batch.

    It also sorts the batch by caption length in descending order, which is
    required when using torch.nn.utils.rnn.pack_padded_sequence later on.
    """
    def __init__(self, pad_idx: int) -> None:
        """
        Initializes the custom collate function.

        Args:
            pad_idx (int): The vocabulary ID for the padding token.
        """
        self.pad_idx = pad_idx
        # logger.info(f"MyCollate initialized with padding index: {pad_idx}")


    def __call__(self, batch: List[Optional[Tuple[torch.Tensor, torch.Tensor]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collates a list of samples into a batch.

        Args:
            batch (list): A list of tuples, where each tuple contains
                          (image_tensor, caption_tensor) or None if the sample failed to load.

        Returns:
            tuple: A tuple containing:
                    - images_tensor (torch.Tensor): A batch of image tensors (batch_size, C, H, W).
                    - padded_captions_tensor (torch.Tensor): A batch of padded caption tensors (batch_size, max_seq_len).
                    - caption_lengths_tensor (torch.Tensor): A tensor of original caption lengths (batch_size,).
        """

        # 1. Filter and Sort
        # Filter out any samples that might have failed to load (e.g., image not found)
        # Type: List[Tuple[torch.Tensor, torch.Tensor]]
        valid_batch = [item for item in batch if item is not None]

        if not valid_batch:
            logger.warning("Batch is empty after filtering out failed samples. Returning empty Tensors.")
            # Return empty tensors with correct dtype for safety in downstream code
            return torch.tensor([], dtype=torch.float), torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

        # CRUCIAL: Sort the batch by caption length in descending order.
        # This is a requirement for using `pack_padded_sequence` which is standard in captioning models.
        valid_batch.sort(key=lambda x: len(x[1]), reverse=True)

        # 2. Separate Tensors
        # imgs: list of individual image tensors (C, H, W)
        # targets: list of individual caption tensors (seq_len,)
        imgs = [item[0] for item in valid_batch]
        targets = [item[1] for item in valid_batch]

        # 3. Stack Images
        # Resulting shape: (batch_size, C, H, W)
        imgs_tensor = torch.stack(imgs, dim=0)

        # 4. Get Original Lengths
        # These lengths are crucial for sequence models (RNN/Transformer)
        lengths_tensor = torch.tensor([len(cap) for cap in targets], dtype=torch.long)

        # 5. Pad Captions
        # `pad_sequence` takes a list of variable-length tensors and pads them to the max length.
        # Resulting shape: (batch_size, max_seq_len)
        padded_captions_tensor = torch.nn.utils.rnn.pad_sequence(
            targets, batch_first=True, padding_value=self.pad_idx
        )

        logger.debug(f"Collated batch shapes: Images {imgs_tensor.shape}, Captions {padded_captions_tensor.shape}")

        return imgs_tensor, padded_captions_tensor, lengths_tensor