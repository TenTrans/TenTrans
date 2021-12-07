import torch
from torch import nn
from src.layers.transformer_decoder_layer import TransformerDecoderLayer
from src.layers.positional_encoding import positional_encoding
from src.utils.utility import (
    get_embedding,
    subsequent_mask,
)


class TransformerDecoder(nn.Module):
    """
    A transformer decoder with N masked layers.
    Decoder layers are masked so that an attention head cannot see the future.
    """
    def __init__(self, model_config, vocab):

        super().__init__()

        self.pad_index = vocab.pad_index
        self.hidden_size = model_config["hidden_size"]
        self.ff_size = model_config["ff_size"]
        self.num_heads = model_config["num_heads"]
        self.num_layers = model_config["decoder_layers"]
        self.embedd_size = model_config["embedd_size"]
        self.embedding = get_embedding(len(vocab),
                                   self.embedd_size,
                                   padding_idx=vocab.pad_index)

        assert self.embedd_size == self.hidden_size
        dropout = model_config.get("dropout", 0)
        # create num_layers decoder layers and put them in a list
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                size=self.hidden_size,
                ff_size=self.ff_size,
                num_heads=self.num_heads,
                dropout=dropout,
                attention_dropout=model_config.get("attention_dropout", 0),
                normalize_before=model_config.get("pre_norm", False),
            ) for _ in range(self.num_layers)
        ])

        self.embed_norm = nn.LayerNorm(self.embedd_size, eps=1e-12)
        self.pe = positional_encoding(
            self.embedd_size,
            model_config.get("max_seq_length", 512),
            model_config.get("learned_pos", False),
        )

        self.dropout = nn.Dropout(p=dropout)
        self.use_langembed = model_config.get("use_langembed", False)
        self.output_layer = nn.Linear(self.hidden_size, len(vocab), bias=True)
        self.n_words = len(vocab)

    def forward(self,
                tgt,
                encoder_output,
                src_mask,
                lang_id,
                positions=None,
                cache=None):
        """
        src: Tensor. [bsz, length]
        lang_id: int or tensor. when use_langembed is true,
        especially for multilingual bert,
        positions: Tensor or None 
        the input embedding of bert  = word embedding + lang embedding + positions
        and the lang_id is int or tensor  the size of which is same 
        as src.
        positions: Tensor or None. If None given, the position order is sequential.
        Positions can also be customized for the 
        some tasks, likes TLM or XLNET.
        """
        bsz, seqlen = tgt.size()
        tgt_mask = (tgt != self.pad_index).unsqueeze(1)
        tgt_key_mask = tgt_mask & subsequent_mask(seqlen).type_as(tgt_mask)

        if positions is None:
            positions = torch.arange(seqlen).unsqueeze(0).repeat(bsz, 1).to(
                tgt.device)

        if cache is not None:
            positions = positions[:, -1:]
            tgt = tgt[:, -1:]
            tgt_mask = tgt_mask[:, -1:]  # (tgt != self.pad_index).unsqueeze(1)
            tgt_key_mask = tgt_key_mask[:, -1:]
            if self.use_langembed:
                if lang_id.size() == 1:
                    lang_id = lang_id[-1:]
                else:
                    lang_id = lang_id[:, -1:]

        x = self.pe(self.embedding(tgt), positions=positions)
        if self.use_langembed:
            assert (lang_id == self.unk_index).sum() == 0
            if lang_id.size() == 1:
                x = x + self.embedding(lang_id).unsqueeze(1).expand_as(x)
            else:
                x = x + self.embedding(lang_id)

        x = self.embed_norm(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(
                x=x,
                memory=encoder_output,
                src_mask=src_mask,
                trg_mask=tgt_key_mask,
                cache=cache,
            )
        return x
