import torch.nn as nn


class MaskLanguageModel(nn.Module):
    def __init__(self, model_config, sentence_rep_model, vocab):

        super().__init__()
        self.sentence_rep = sentence_rep_model
        self.input_dim = model_config["sentence_rep_dim"]
        self.pad_index = vocab.pad_index
        self.target = nn.Linear(self.input_dim, len(vocab))

        if model_config["share_out_embedd"]:
            self.target.weight = self.sentence_rep.encoder.embedding.weight

    def forward(self, src, lang_id=None, pred_mask=None, positions=None):
        sentence_rep, _ = self.sentence_rep(
            src=src, lang_id=lang_id, positions=positions
        )
        if pred_mask is not None:
            tensor = sentence_rep[
                pred_mask.unsqueeze(-1).expand_as(sentence_rep).bool()
            ].view(-1, self.input_dim)
            return self.target(tensor)
        else:
            return self.target(sentence_rep)
