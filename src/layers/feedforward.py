from torch import nn


class FeedForward(nn.Module):
    def __init__(self, input_size, ff_size, dropout=0.1, activation="relu"):
        """
        Initializes position-wise feed-forward layer.
        :param input_size: dimensionality of the input.
        :param ff_size: dimensionality of intermediate representation
        :param dropout:
        """
        super().__init__()
        assert activation in ["relu", "gelu"]
        # self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.ffn_layer = nn.Sequential(
            nn.Linear(input_size, ff_size),
            nn.ReLU(inplace=True) if activation == "relu" else nn.GELU(),
            nn.Linear(ff_size, input_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x_norm = self.layer_norm(x)
        return self.ffn_layer(x)  # + x
