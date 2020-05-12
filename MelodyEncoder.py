import torch
import torch.nn as nn

class MelodyEncoder(nn.Module):
    """
    This module encodes the conditionnal melody given as input as a 1-dim embedding to feed
    to the Bar Generator
    """

    def __init__(self, output_dim):
        super(MelodyEncoder, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv3d(1, 32, (1, 4, 12), stride=(1, 4, 4)),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
        )
        self.layer_2 = nn.Sequential(
            nn.Conv3d(32, 64, (1, 3, 3), stride=(1, 3, 2), padding=(0,0,1)),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
        )
        self.layer_3 = nn.Linear(512, output_dim)

        nn.init.kaiming_normal_(self.layer_1[0].weight)
        nn.init.kaiming_normal_(self.layer_2[0].weight)

    def forward(self, melody):
        # Dimension of melody is (batch, 1, N_bars, 96, 128) = (batch,n_tracks,n_bars,n_steps_per_bat,n_pitches)
        x = self.layer_1(melody)
        # Output is of dimension (batch, 32, N_bars, 12, 15)
        x = self.layer_2(x)
        # Output is of dimension (batch, 64, N_bars, 2, 4)
        x = x.permute(0, 2, 1, 3, 4)
        # Output is of dimension (batch, N_bars, 64, 2, 4)
        x = x.reshape(melody.shape[0], melody.shape[2], -1)
        # Output is of dimension (batch, N_bars, 512)
        x = self.layer_3(x)
        # Output is of dimension (batch, N_bars, output_dim)

        return x