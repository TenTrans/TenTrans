import torch
import torch.nn as nn
from src.utils.utility import get_embedding
import numpy as np


def positional_encoding(size, max_len=1024, learned=False):
    """
        Positional Encoding with maximum length max_len
        :param size:
        :param max_len:
        :param learned: if position embedding is learnedable.
    """
    if learned:
        return LearnedPositionalEncoding(size, max_len)
    return SinusoPositionalEncoding(size, max_len)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, size: int = 0, max_len: int = 1024):
        """
        Positional Encoding with maximum length max_len
        :param size:
        :param max_len:
        :param dropout:
        """
        super().__init__()
        self.pe = get_embedding(max_len, size)

    def forward(self, emb, positions=None):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(batch_size, seq_len, self.dim)``
        """
        length = emb.size(1)
        if positions is None:
            positions = (torch.arange(
                length, dtype=torch.long).unsqueeze(0).to(emb.device))
        return self.pe(positions) + emb


class SinusoPositionalEncoding(nn.Module):
    def __init__(self, size: int = 0, max_len: int = 1024):
        """
        Positional Encoding with maximum length max_len
        :param size:
        :param max_len:
        :param dropout:
        """
        super().__init__()
        self.pe = get_embedding(max_len, size)
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / size) for j in range(size)]
             for pos in range(max_len)])

        self.pe.weight[:,
                       0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        self.pe.weight[:,
                       1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        self.pe.weight.detach_()
        self.pe.weight.requires_grad = False

    def forward(self, emb, positions=None):

        length = emb.size(1)
        if positions is None:
            positions = (torch.arange(
                length, dtype=torch.long).unsqueeze(0).to(emb.device))
        return self.pe(positions) + emb
