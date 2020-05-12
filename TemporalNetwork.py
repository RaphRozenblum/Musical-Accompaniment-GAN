import torch
import torch.nn as nn

class TemporalNetwork(nn.Module):
    """
    This module extends the given noise in the time dimension
    """

    def __init__(self, z_dim, n_bars):
        super(TemporalNetwork, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 1024, (2, 1), stride=(1, 1)),
            nn.BatchNorm2d(1024, momentum=0.9),
            nn.ReLU()
        )
        self.layer_2 = nn.Sequential(
            nn.ConvTranspose2d(1024, z_dim, (n_bars - 1, 1), stride=(1, 1)),
            nn.BatchNorm2d(z_dim, momentum=0.9),
            nn.ReLU()
        )

        nn.init.kaiming_normal_(self.layer_1[0].weight)
        nn.init.kaiming_normal_(self.layer_2[0].weight)

    def forward(self, x):
        # Input if of size (batch, 1 bar, z_dim, 1)
        x = x.permute(0, 2, 1, 3)
        # Input if of size (batch, z_dim, 1, 1)
        x = self.layer_1(x)
        # Input if of size (batch, 1024, 2, 1)
        x = self.layer_2(x)
        # Input if of size (batch, z_dim, n_bars, 1)
        x.permute(0, 2, 1, 3)
        # Input if of size (batch, n_bars, z_dim, 1)

        return x