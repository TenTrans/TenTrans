from src.models.encoder.transformer_encoder import TransformerEncoder
import torch.nn as nn

support_model = {
    'transformer': TransformerEncoder,
    'xlm': TransformerEncoder,
}


class SentenceRepModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, **kwargs):
        return self.encoder(**kwargs)

    @classmethod
    def build_model(cls, model_config, vocab):
        encoder = support_model[model_config['type']](model_config, vocab)
        return cls(encoder)