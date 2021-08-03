import torch
import torch.nn as nn
from collections import OrderedDict


class ClassificationModel(nn.Module):
    def __init__(self, model_config, sentenceRepModel, vocab):

        super().__init__()
        self.sentenceRep = sentenceRepModel
        self.input_dim = model_config['sentence_rep_dim']
        self.vocab = vocab
        self.pad_index = vocab.pad_index
        self.target = nn.Sequential(
            OrderedDict([
                ('classification_dropout',
                 nn.Dropout(model_config['dropout'])),
                ('classification_output_layer',
                 nn.Linear(self.input_dim, model_config['num_label1'])),
            ]))

    def forward(self, src, lang_id=None):
        sentence_rep, _ = self.sentenceRep(src=src, lang_id=lang_id)
        sentence_rep = sentence_rep[:, 0]
        return self.target(sentence_rep)
