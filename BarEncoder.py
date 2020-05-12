import torch
import torch.nn as nn


class BarEncoder(nn.Module):
    """
    This module encodes a whole bar over all tracks as a 1-dim embedding to create short-term memory for
    the Bar Generator
    """

    def __init__(self, output_dim):
        super(BarEncoder, self).__init__()

        self.layer_1 = nn.Sequential(
            nn.Conv2d(5, 16, (3, 12), stride=(3, 4), padding=(0, 0)),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(16, 16, (2, 3), stride=(2, 2), padding=(0, 0)),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 1))
        )

        nn.init.kaiming_normal_(self.layer_1[0].weight)
        nn.init.kaiming_normal_(self.layer_2[0].weight)

        self.linear = nn.Linear(in_features=448, out_features=output_dim)

    def forward(self, bar):
        # The bar is of shape (batch,5,1,96,128) = (batch,n_tracks,n_bars,n_steps_per_bat,n_pitches)
        bar = bar.squeeze(2)
        # Now the shape is (batch,5,96,128)
        out = self.layer_1(bar)
        # Now the shape is (batch,16,16,15)
        out = self.layer_2(out)
        # Now the shape is (batch,16,4,7)
        out = nn.Flatten()(out)
        # Now the shape is (batch,448)
        out = self.linear(out)
        # Now the shape is (batch, output_shape)

        return out