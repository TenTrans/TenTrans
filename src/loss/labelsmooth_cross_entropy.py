import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, size, smoothing=0.0):
        super().__init__()
        self.smoothing = smoothing
        self.n_words = size

    def forward(self, x, target):
        assert x.size(1) == self.n_words
        scores = F.log_softmax(x, dim=-1)
        nll_loss = -scores.gather(dim=-1, index=target.unsqueeze(1))
        smooth_loss = -scores.sum(dim=-1, keepdim=True)

        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
        eps_i = self.smoothing / self.n_words
        loss = (1. - self.smoothing) * nll_loss + eps_i * smooth_loss
        loss = loss / scores.size(0)
        return loss