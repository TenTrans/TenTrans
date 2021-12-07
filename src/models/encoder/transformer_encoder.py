from torch import nn
from src.layers.transformer_encoder_layer import TransformerEncoderLayer
from src.layers.positional_encoding import positional_encoding
from src.utils.utility import get_embedding


class TransformerEncoder(nn.Module):
    """
    A transformer encoder with N masked layers.
    """
    def __init__(self, model_config, vocab):
        """
        model_config: dict
        vocab: Vocab
        """
        super().__init__()

        self.pad_index = vocab.pad_index
        self.unk_index = vocab.unk_index

        self.hidden_size = model_config["hidden_size"]
        self.ff_size = model_config["ff_size"]
        self.num_heads = model_config["num_heads"]
        self.num_layers = model_config["encoder_layers"]
        self.embedd_size = model_config["embedd_size"]

        self.embedding = get_embedding(
            len(vocab), self.embedd_size, padding_idx=vocab.pad_index
        )

        dropout = model_config.get("dropout", 0)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    size=self.hidden_size,
                    ff_size=self.ff_size,
                    num_heads=self.num_heads,
                    dropout=dropout,
                    attention_dropout=model_config.get("attention_dropout", 0),
                    activation=model_config.get("activation", "relu"),
                    normalize_before=model_config.get("pre_norm", False),
                )
                for _ in range(self.num_layers)
            ]
        )

        self.embed_norm = nn.LayerNorm(self.embedd_size, eps=1e-12)
        self.pe = positional_encoding(
            self.embedd_size,
            model_config.get("max_seq_length", 512),
            model_config.get("learned_pos", False),
        )
        self.dropout = nn.Dropout(p=dropout)
        self.use_langembed = model_config.get("use_langembed", False)

    def forward(self, src, lang_id=None, positions=None):
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
        mask = (src != self.pad_index).unsqueeze(1)
        x = self.pe(self.embedding(src), positions=positions)

        # in next version remove lang embed
        if self.use_langembed:
            assert (lang_id == self.unk_index).sum() == 0
            if lang_id.dim() == 1:
                x = x + self.embedding(lang_id).unsqueeze(1).expand_as(x)
            else:
                x = x + self.embedding(lang_id)

        x = self.embed_norm(x)
        x = self.dropout(x)
        all_layer_hiddens = [x]
        for layer in self.layers:
            x = layer(x, mask)
            all_layer_hiddens.append(x)

        return x, all_layer_hiddens
