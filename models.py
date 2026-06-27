import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(torch.nn.Module):
    """
    A very simple transformer-based model for morphological parsing.
    Framed as a sequence-to-sequence problem.
    """
    
    def __init__(self, d_input, num_glosses, max_len=50):
        """
        d_input: Input dimensionality. For example, it could be the dimensionality of the input language alphabet.
        num_glosses: Dimensionality of the output - the number of possible glosses.
        max_len: Maximum length of the input sequence.
        """
        super().__init__()
        self.conv1 = torch.nn.Conv1d(
            in_channels=d_input,
            out_channels=256,
            kernel_size=3,
            padding='same',
        )
        self.conv2 = torch.nn.Conv1d(
            in_channels=256,
            out_channels=num_glosses,
            kernel_size=3,
            padding='same',
        )
        self.transformer = nn.Transformer(
            d_model=num_glosses,
            nhead=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            #dim_feedforward=64,
        )
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.d_input = d_input
        self.num_glosses = num_glosses
        
        pos_encoding = torch.zeros(max_len, num_glosses, requires_grad=False)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(torch.arange(0, num_glosses, 2).float() * (-math.log(10000.0)) / num_glosses)
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, X, y, tgt_mask=None, tgt_is_causal=False):
        X = torch.permute(X.unsqueeze(0), dims=(0, 2, 1)) # To N, C, L
        X = self.conv1(X)
        X = self.conv2(X)
        X = torch.permute(X.squeeze(), dims=(1, 0)) # Back to L, C
        X += self.pos_encoding[:X.shape[0], :X.shape[1]]
        transformer_out = self.transformer(X, y, tgt_mask=tgt_mask, tgt_is_causal=tgt_is_causal)
        out = self.log_softmax(transformer_out)
        return out
