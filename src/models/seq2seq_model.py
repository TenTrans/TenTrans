import torch.nn as nn
from src.models.decoder import support_decoder
from collections import OrderedDict


class Seq2SeqModel(nn.Module):
    def __init__(self, model_config, sentence_rep_model, tgt_vocab):
        super().__init__()
        self.sentence_rep = sentence_rep_model
        self.pad_index = tgt_vocab.pad_index
        self.eos_index = tgt_vocab.eos_index
        self.bos_index = tgt_vocab.bos_index
        self.target = nn.Sequential(
            OrderedDict([
                (
                    "decoder",
                    support_decoder[model_config["type"]](model_config,
                                                          tgt_vocab),
                ),
            ]))

        if model_config.get("share_all_embedd", False):
            self.target[
                0].embedding.weight = self.sentence_rep.encoder.embedding.weight

        if model_config.get("share_out_embedd", False):
            self.target[0].output_layer.weight = self.target[
                0].embedding.weight

    def forward(self, mode, **kwargs):
        assert mode in ["fwd", "predict"]
        if mode == "fwd":
            return self.fwd(**kwargs)
        else:
            return self.predict(**kwargs)

    def fwd(self, src, tgt, lang1_id=None, lang2_id=None):
        encoder_out, _ = self.sentence_rep(src=src, lang_id=lang1_id)
        src_mask = (src != self.pad_index).unsqueeze(1)
        tensor = self.target[0](tgt, encoder_out, src_mask, lang2_id)
        return tensor

    def predict(self, tensor):
        return self.target[0].output_layer(tensor)
