# src/components/decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List, Tuple, Union
from src.modules.positional_encoder import PositionalEncoding
from src.entity.config_entity import DecoderConfig
from src.utils.logging_setup import logger
from src.utils.device import DEVICE

# Using a named tuple or simple dataclass for beam search candidates can improve readability
from collections import namedtuple

# Beam search candidate structure
BeamCandidate = namedtuple('BeamCandidate', ['score', 'sequence', 'last_hidden', 'last_state', 'attention_weights'])

class TransformerDecoder(nn.Module):
    """Transformer-based Decoder for image captioning."""
    def __init__(self, config: DecoderConfig) -> None:
        super(TransformerDecoder, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.positional_encoding = PositionalEncoding(self.config.d_model, max_len=self.config.max_len)
        self.dropout = nn.Dropout(self.config.dropout)

        decoder_layer = nn.TransformerDecoderLayer(d_model=self.config.d_model, nhead=self.config.num_heads, dim_feedforward=self.config.ff_dim, dropout=self.config.dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, self.config.num_transformer_layers)

        self.fc = nn.Linear(self.config.d_model, self.config.vocab_size)
        logger.info(f"TransformerDecoder initialized. d_model={self.config.d_model}")

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generates an upper-triangular matrix of -inf, with 0 on diagonal."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(DEVICE)

    def forward(self, trg: torch.Tensor, memory: torch.Tensor, trg_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the Transformer Decoder.
        Args:
            trg (torch.Tensor): Target sequence (captions) (batch_size, trg_seq_len).
            memory (torch.Tensor): Encoded image features (batch_size, num_pixels, encoder_dim).
            trg_mask (torch.Tensor, optional): Mask for target sequence (trg_seq_len, trg_seq_len).
        Returns:
            torch.Tensor: Logits for the next word (batch_size, trg_seq_len, vocab_size).
        """
        trg_embed = self.dropout(self.positional_encoding(self.embedding(trg)))

        # memory_key_padding_mask should mask the padding in encoder_out
        # If encoder_out already has num_pixels, there's usually no padding for image features
        # unless images themselves were padded to a fixed size and then features extracted.
        # Assuming encoder_out does not contain padding that needs masking.
        output = self.transformer_decoder(trg_embed, memory, tgt_mask=trg_mask)
        output = self.fc(output)
        return output

    def generate_caption(self, encoder_out: torch.Tensor, vocab, max_len: int = 50, beam_size: int = 1) -> Tuple[List[str], torch.Tensor]:
        """
        Generates a caption for a single image using greedy search or beam search.
        Args:
            encoder_out (torch.Tensor): Encoded image features (1, num_pixels, encoder_dim).
            vocab (object): The vocabulary object (e.g., TextPreprocessor instance).
            max_len (int): Maximum length of the generated caption.
            beam_size (int): The number of sequences to keep at each step. Set to 1 for greedy search.
        Returns:
            tuple: (caption_words, attention_alphas)
                caption_words (list[str]): List of generated words (excluding special tokens).
                attention_alphas (torch.Tensor): Stacked attention weights for visualization (seq_len, num_pixels).
                                                Returns None if attention weights are not tracked or available.
        """
        with torch.no_grad():
            # Ensure encoder_out is on the correct device
            encoder_out = encoder_out.to(DEVICE)
            batch_size = encoder_out.size(0) # Should be 1 for single image inference

            if batch_size != 1:
                logger.error("`generate_caption` expects a single image tensor (batch_size=1).")
                raise ValueError("Batch size must be 1 for single image caption generation.")

            sos_idx = vocab.stoi["<SOS>"]
            eos_idx = vocab.stoi["<EOS>"]
            pad_idx = vocab.stoi["<PAD>"]

            if beam_size == 1: # Greedy Search
                sequence = [sos_idx]
                attention_alphas = [] # Not directly available from standard TransformerDecoder output
                                    # (requires custom attention layer inside TransformerDecoderLayer)
                                    # For now, will return empty.

                for _ in range(max_len - 1): # -1 because we start with <SOS>
                    trg_tensor = torch.LongTensor(sequence).unsqueeze(0).to(DEVICE) # (1, current_seq_len)
                    trg_mask = self._generate_square_subsequent_mask(trg_tensor.size(1))

                    # Forward pass
                    # predictions: (1, current_seq_len, vocab_size)
                    predictions = self.forward(trg_tensor, encoder_out, trg_mask)

                    # Get the last token's prediction logits
                    next_token_logits = predictions[:, -1, :] # (1, vocab_size)
                    next_word_idx = next_token_logits.argmax(dim=-1).item() # Get the word with max probability

                    sequence.append(next_word_idx)

                    if next_word_idx == eos_idx:
                        break

                # Convert numerical IDs back to words, excluding special tokens
                caption_words = [vocab.itos[word_id] for word_id in sequence
                                if word_id not in [pad_idx, sos_idx, eos_idx]]

                # Return empty attention alphas for now as standard TransformerDecoder doesn't expose them easily
                return caption_words, torch.empty(0)

            else: # Beam Search
                # Initialize beam with the start token
                # score, sequence (list), last_decoder_output (for subsequent steps), attention_weights (list of alpha tensors)
                # For Transformer, we need the entire sequence and the memory (encoder_out)

                # Initial candidate: (log_prob, [SOS_idx])
                beam = [BeamCandidate(score=0.0, sequence=[sos_idx], last_hidden=None, last_state=None, attention_weights=[])]
                completed_sequences = []

                for _ in range(max_len - 1): # Iterate for max_len - 1 steps to generate words
                    if not beam: # If all beams have finished
                        break

                    current_beam_candidates = []
                    for candidate in beam:
                        current_score, current_sequence, _, _, current_alphas = candidate

                        # Prepare input for the decoder
                        trg_tensor = torch.LongTensor(current_sequence).unsqueeze(0).to(DEVICE) # (1, current_seq_len)
                        trg_mask = self._generate_square_subsequent_mask(trg_tensor.size(1))

                        # Forward pass: predictions (1, current_seq_len, vocab_size)
                        predictions = self.forward(trg_tensor, encoder_out, trg_mask)

                        # Get logits for the last generated token
                        next_token_logits = predictions[:, -1, :] # (1, vocab_size)
                        next_token_log_probs = F.log_softmax(next_token_logits, dim=-1).squeeze(0) # (vocab_size)

                        # Get top-k (beam_size) candidates
                        topk_log_probs, topk_indices = torch.topk(next_token_log_probs, k=beam_size, dim=-1)

                        for i in range(beam_size):
                            next_word_log_prob = topk_log_probs[i].item()
                            next_word_idx = topk_indices[i].item()

                            new_sequence = current_sequence + [next_word_idx]
                            new_score = current_score + next_word_log_prob
                            # Attention weights are not directly exposed by nn.TransformerDecoder, so keep empty for now
                            new_alphas = current_alphas

                            new_candidate = BeamCandidate(score=new_score, sequence=new_sequence,
                                                            last_hidden=None, last_state=None, attention_weights=new_alphas)

                            if next_word_idx == eos_idx:
                                completed_sequences.append(new_candidate)
                            else:
                                current_beam_candidates.append(new_candidate)

                    # Sort and select top beam_size candidates for the next iteration
                    beam = sorted(current_beam_candidates, key=lambda x: x.score, reverse=True)[:beam_size]

                    # Stop early if enough completed sequences are found
                    if len(completed_sequences) >= beam_size:
                        break

                # Add any remaining incomplete sequences to completed list (they might be the best if no EOS)
                for candidate in beam:
                     if candidate.sequence[-1] != eos_idx: # Only add if not already completed
                        completed_sequences.append(candidate)

                # Sort all completed sequences by their total score (log probability sum)
                completed_sequences.sort(key=lambda x: x.score, reverse=True)

                # Select the best sequence
                if completed_sequences:
                    best_candidate = completed_sequences[0]
                    best_sequence = best_candidate.sequence
                    best_alphas = best_candidate.attention_weights # This will be empty for now
                else:
                    logger.warning("Beam search completed without finding any valid sequences. Returning empty caption.")
                    return [], torch.empty(0) # Return empty attention alphas

                # Convert numerical IDs back to words, excluding special tokens
                caption_words = [vocab.itos[word_id] for word_id in best_sequence
                                if word_id not in [pad_idx, sos_idx, eos_idx]]

                return caption_words, torch.empty(0) # Return empty attention alphas