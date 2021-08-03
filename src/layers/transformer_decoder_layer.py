from torch import nn
from torch import Tensor
from src.layers.multihead_attention import MultiHeadedAttention
from src.layers.feedforward import FeedForward


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer.
    Consists of self-attention, source-attention, and feed-forward.
    """
    def __init__(self,
                 size: int = 0,
                 ff_size: int = 0,
                 num_heads: int = 0,
                 dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 normalize_before: bool = False):
        """
        Represents a single Transformer decoder layer.
        It attends to the source representation and the previous decoder states.
        :param size: model dimensionality
        :param ff_size: size of the feed-forward intermediate layer
        :param num_heads: number of heads
        :param dropout: dropout to apply to input
        """
        super().__init__()
        self.size = size

        self.tgt_tgt_att = MultiHeadedAttention(num_heads,
                                                size,
                                                dropout=attention_dropout)
        self.src_tgt_att = MultiHeadedAttention(num_heads,
                                                size,
                                                dropout=attention_dropout)

        self.feed_forward = FeedForward(size, ff_size=ff_size, dropout=dropout)

        self.tgt_att_layer_norm = nn.LayerNorm(size, eps=1e-12)
        self.src_tgt_att_layer_norm = nn.LayerNorm(size, eps=1e-12)
        self.ffn_layer_norm = nn.LayerNorm(size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        self.normalize_before = normalize_before

    # pylint: disable=arguments-differ
    def forward(self,
                x: Tensor = None,
                memory: Tensor = None,
                src_mask: Tensor = None,
                trg_mask: Tensor = None,
                cache=None) -> Tensor:
        """
        Forward pass of a single Transformer decoder layer.
        :param x: inputs
        :param memory: source representations
        :param src_mask: source mask
        :param trg_mask: target mask (so as to not condition on future steps)
        :return: output tensor
        """
        # decoder/target self-attention
        residual = x
        x = self.maybe_layer_norm(self.tgt_att_layer_norm, x, before=True)
        x = self.tgt_tgt_att(x, x, x, mask=trg_mask, cache=cache)
        x = self.dropout(x) + residual
        x = self.maybe_layer_norm(self.tgt_att_layer_norm, x, after=True)

        # source-target attention
        residual = x
        x = self.maybe_layer_norm(self.src_tgt_att_layer_norm, x, before=True)
        x = self.src_tgt_att(memory, memory, x, mask=src_mask, cache=cache)
        x = self.dropout(x) + residual
        x = self.maybe_layer_norm(self.src_tgt_att_layer_norm, x, after=True)

        # final feed-forward layer
        residual = x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, before=True)
        x = self.feed_forward(x)
        x = residual + x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, after=True)

        return x

    #@TODO
    def maybe_layer_norm(self,
                         layer_norm,
                         x,
                         before=False,
                         after=False) -> Tensor:
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x