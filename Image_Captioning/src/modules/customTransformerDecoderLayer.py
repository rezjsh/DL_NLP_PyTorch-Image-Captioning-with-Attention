import torch
import torch.nn as nn
from typing import Optional

# Custom TransformerDecoderLayer to force return of attention weights
class CustomTransformerDecoderLayer(nn.TransformerDecoderLayer):
    '''
    Custom TransformerDecoderLayer that forces the return of attention weights
    from the cross-attention mechanism. This is useful for visualization or
    analysis purposes where attention maps are needed.
    '''
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> torch.Tensor:
        
        # Self-attention block
        x = tgt
        if self.norm_first:
            _x = self.norm1(x)
            _x, _ = self.self_attn(_x, _x, _x,
                                        attn_mask=tgt_mask,
                                        key_padding_mask=tgt_key_padding_mask,
                                        need_weights=False, # Self-attention weights usually not needed for cross-attention vis
                                        is_causal=tgt_is_causal
                                        )
            x = x + self.dropout1(_x)
            _x = self.norm2(x)
            # Cross-attention block
            _x, _ = self.multihead_attn(_x, memory, memory,
                                                attn_mask=memory_mask,
                                                key_padding_mask=memory_key_padding_mask,
                                                need_weights=True, # Force True here
                                                average_attn_weights=False, # Force False to get all heads
                                                is_causal=memory_is_causal # Pass new argument
                                                )
            x = x + self.dropout2(_x)
            _x = self.norm3(x)
            
            # Feedforward block
            _x = self.linear2(self.dropout(self.activation(self.linear1(_x))))
            x = x + self.dropout3(_x)
        else: # Default norm_first=False
            _x, _ = self.self_attn(x, x, x,
                                        attn_mask=tgt_mask,
                                        key_padding_mask=tgt_key_padding_mask,
                                        need_weights=False,
                                        is_causal=tgt_is_causal # Pass new argument
                                        )
            x = x + self.dropout1(_x)
            x = self.norm1(x)
            # Cross-attention block
            _x, _ = self.multihead_attn(x, memory, memory,
                                                attn_mask=memory_mask,
                                                key_padding_mask=memory_key_padding_mask,
                                                need_weights=True, # Force True here
                                                average_attn_weights=False, # Force False to get all heads
                                                is_causal=memory_is_causal # Pass new argument
                                                )
            x = x + self.dropout2(_x)
            x = self.norm2(x)
            # Feedforward block
            _x = self.linear2(self.dropout(self.activation(self.linear1(x))))
            x = x + self.dropout3(_x)
            x = self.norm3(x)

        return x
