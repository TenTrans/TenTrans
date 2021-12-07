from torch import nn
from torch import Tensor
from src.layers.multihead_attention import MultiHeadedAttention
from src.layers.feedforward import FeedForward


class TransformerEncoderLayer(nn.Module):
    """
    Encoder layer block.
    The main process is `dropout -> add residual -> layernorm`.
    You can also `dropout -> layernorm -> add residual` by
    using normalize_before=True
    Args:
       size: int
       ff_size: int
       num_heads: int
       dropout: float
       attention_dropout: float
       normalize_before: boolen
       activation: "gelu" or "relu",

       when using bert-style transformer, gelu activattion
       is suggestion.
       gelu(x) = 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*np.power(x,3))))
       when using transformer for machine tranlstion task,
       relu is better.
    """
    def __init__(
        self,
        size: int = 0,
        ff_size: int = 0,
        num_heads: int = 0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        normalize_before: bool = False,
        activation: str = "gelu",
    ):

        super().__init__()
        self.src_src_att = MultiHeadedAttention(num_heads,
                                                size,
                                                dropout=attention_dropout)
        self.att_layer_norm = nn.LayerNorm(size, eps=1e-12)
        self.feed_forward = FeedForward(size,
                                        ff_size=ff_size,
                                        dropout=dropout,
                                        activation=activation)
        self.ffn_layer_norm = nn.LayerNorm(size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        self.size = size
        self.normalize_before = normalize_before

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:

        residual = x
        x = self.maybe_layer_norm(self.att_layer_norm, x, before=True)
        x = self.src_src_att(x, x, x, mask)
        x = self.dropout(x) + residual
        x = self.maybe_layer_norm(self.att_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, before=True)
        x = self.feed_forward(x)
        x = x + residual
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, after=True)

        return x

    # @TODO
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
