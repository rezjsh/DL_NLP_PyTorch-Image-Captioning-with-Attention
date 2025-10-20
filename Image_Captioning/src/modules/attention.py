from src.utils.logging_setup import logger
import torch.nn as nn

class Attention(nn.Module):
    """
    Bahdanau-style (additive) Attention mechanism.
    It computes attention weights over the encoder's output (image features)
    based on the decoder's previous hidden state.
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        Args:
            encoder_dim (int): Dimension of the encoder's output features (e.g., 2048 from ResNet).
            decoder_dim (int): Dimension of the decoder's hidden state (e.g., Config.hidden_size).
            attention_dim (int): Dimension of the intermediate attention layer.
        """
        super(Attention, self).__init__()
        
        # Linear layer to transform encoder output
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  
        # Linear layer to transform decoder's previous hidden state
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  
        # Linear layer to compute attention scores (energies)
        self.full_att = nn.Linear(attention_dim, 1)              
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Softmax over the pixels (dim=1)

        logger.info(f"Attention mechanism initialized with encoder_dim={encoder_dim}, decoder_dim={decoder_dim}, attention_dim={attention_dim}.")

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward pass of the attention mechanism.
        Args:
            encoder_out (torch.Tensor): Image features from encoder (batch_size, num_pixels, encoder_dim).
            decoder_hidden (torch.Tensor): Decoder's previous hidden state (batch_size, decoder_dim).
        Returns:
            tuple: (context_vector, alpha_weights)
                context_vector (torch.Tensor): Weighted average of encoder features (batch_size, encoder_dim).
                alpha_weights (torch.Tensor): Attention weights (batch_size, num_pixels).
        """
        # Transform encoder output: (batch_size, num_pixels, attention_dim)
        att1 = self.encoder_att(encoder_out)  
        
        # Transform decoder hidden state: (batch_size, 1, attention_dim)
        # unsqueeze(1) adds a dimension for broadcasting with att1
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)  
        
        # Compute attention energies: (batch_size, num_pixels, 1) then squeeze to (batch_size, num_pixels)
        e = self.full_att(self.relu(att1 + att2)).squeeze(2)  
        
        # Apply softmax to get attention weights (alpha): (batch_size, num_pixels)
        alpha = self.softmax(e)  
        
        # Compute context vector: weighted sum of encoder outputs
        # (batch_size, num_pixels, encoder_dim) * (batch_size, num_pixels, 1) -> (batch_size, num_pixels, encoder_dim)
        # then sum over num_pixels to get (batch_size, encoder_dim)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  
        
        return context, alpha
