from torch import nn


class Pivot(nn.Module):
    """Module to perform dropout at each node. """
    def __init__(self, dropout_rate=0.2):
        super(Pivot, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(x) * (1 - self.dropout_rate)


def pivot(**kwargs):
    model = Pivot(**kwargs)
    return model
