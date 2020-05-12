import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    The Discriminator of the model
    """

    def __init__(self, input_dim):
        super(Discriminator, self).__init__()

        n_bars = input_dim[1]

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=5,
                      out_channels=128,
                      kernel_size=(2, 1, 1),
                      stride=(1, 1, 1),
                      padding=0,
                      ),
            nn.LeakyReLU()
        )
        nn.init.kaiming_normal_(self.conv1[0].weight)

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=128,
                      out_channels=128,
                      kernel_size=(n_bars - 1, 1, 1),
                      stride=(1, 1, 1),
                      padding=0,
                      ),
            nn.LeakyReLU()
        )
        nn.init.kaiming_normal_(self.conv2[0].weight)

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=128,
                      out_channels=128,
                      kernel_size=(1, 1, 12),
                      stride=(1, 1, 12),
                      padding=(0, 0, 2),
                      ),
            nn.LeakyReLU()
        )
        nn.init.kaiming_normal_(self.conv3[0].weight)

        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels=128,
                      out_channels=128,
                      kernel_size=(1, 1, 7),
                      stride=(1, 1, 4),
                      padding=0,
                      ),
            nn.LeakyReLU()
        )
        nn.init.kaiming_normal_(self.conv4[0].weight)

        self.conv5 = nn.Sequential(
            nn.Conv3d(in_channels=128,
                      out_channels=128,
                      kernel_size=(1, 2, 1),
                      stride=(1, 2, 1),
                      padding=(0, 0, 0),
                      ),
            nn.LeakyReLU()
        )
        nn.init.kaiming_normal_(self.conv5[0].weight)

        self.conv6 = nn.Sequential(
            nn.Conv3d(in_channels=128,
                      out_channels=128,
                      kernel_size=(1, 2, 1),
                      stride=(1, 2, 1),
                      padding=(0, 0, 0),
                      ),
            nn.LeakyReLU()
        )
        nn.init.kaiming_normal_(self.conv6[0].weight)

        self.conv7 = nn.Sequential(
            nn.Conv3d(in_channels=128,
                      out_channels=256,
                      kernel_size=(1, 4, 1),
                      stride=(1, 2, 1),
                      padding=(0, 0, 0),
                      ),
            nn.LeakyReLU()
        )
        nn.init.kaiming_normal_(self.conv7[0].weight)

        self.conv8 = nn.Sequential(
            nn.Conv3d(in_channels=256,
                      out_channels=256,
                      kernel_size=(1, 3, 1),
                      stride=(1, 2, 1),
                      padding=(0, 0, 0),
                      ),
            nn.LeakyReLU()
        )
        nn.init.kaiming_normal_(self.conv8[0].weight)

        self.linear1 = nn.Sequential(
            nn.Linear(in_features=2560,
                      out_features=1024,
                      ),
            nn.LeakyReLU(),
        )
        nn.init.kaiming_normal_(self.linear1[0].weight)

        self.linear2 = nn.Sequential(
            nn.Linear(in_features=1024,
                      out_features=1,
                      ),
        )
        nn.init.kaiming_normal_(self.linear2[0].weight)

    def forward(self, song):
        # Dimension of song is (batch, 5, N_bars, 96, 128)
        out = self.conv1(song)
        # Dimension of output is (batch, 128, N_bars-1, 96, 128)
        out = self.conv2(out)
        # Dimension of output is (batch, 128, 1, 96, 128)
        out = self.conv3(out)
        # Dimension of output is (batch, 128, 1, 96, 11)
        out = self.conv4(out)
        # Dimension of output is (batch, 128, 1, 96, 2)
        out = self.conv5(out)
        # Dimension of output is (batch, 128, 1, 48, 2)
        out = self.conv6(out)
        # Dimension of output is (batch, 128, 1, 24, 2)
        out = self.conv7(out)
        # Dimension of output is (batch, 256, 1, 11, 2)
        out = self.conv8(out)
        # Dimension of output is (batch, 256, 1, 5, 2)
        out = nn.Flatten()(out)
        # Dimension of output is (batch, 2560)
        out = self.linear1(out)
        # Dimension of output is (batch, 1024)
        out = self.linear2(out)
        # Dimension of output is (batch, 1)

        return out



