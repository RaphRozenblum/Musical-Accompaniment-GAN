import torch
import torch.nn as nn

class BarGenerator(nn.Module):
    """
    This module uses 1-dim input, extends it along both the time and the pitch axis and creates the
    next bar for the given track.
    """

    def __init__(self, input_dim):
        super(BarGenerator, self).__init__()
        self.layer_1 = nn.Sequential(
            nn.Linear(input_dim, 1024),
            #nn.BatchNorm1d(1024, momentum=0.9),
            nn.ReLU()
        )
        self.layer_2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, (2, 1), stride=(2, 1)),
            #nn.BatchNorm2d(512, momentum=0.9),
            nn.ReLU()
        )
        self.layer_3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, (2, 1), stride=(2, 1)),
            #nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU()
        )
        self.layer_4 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, (2, 1), stride=(2, 1)),
            #nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU()
        )
        self.layer_5 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, (2, 1), stride=(2, 1)),
            #nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU()
        )
        self.layer_6 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, (2, 1), stride=(2, 1)),
            #nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU()
        )
        self.layer_7 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, (3, 1), stride=(3, 1)),
            #nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU()
        )
        self.layer_8 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, (1, 4), stride=(1, 4)),
            #nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU()
        )
        self.layer_9 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, (1, 4), stride=(1, 4)),
            #nn.BatchNorm2d(128, momentum=0.9),
            nn.ReLU()
        )
        self.layer_10 = nn.Sequential(
            nn.ConvTranspose2d(128, 1, (1, 12), stride=(1, 8), padding=(0, 2)),
            nn.Tanh()
        )

        nn.init.kaiming_normal_(self.layer_2[0].weight)
        nn.init.kaiming_normal_(self.layer_3[0].weight)
        nn.init.kaiming_normal_(self.layer_4[0].weight)
        nn.init.kaiming_normal_(self.layer_5[0].weight)
        nn.init.kaiming_normal_(self.layer_6[0].weight)
        nn.init.kaiming_normal_(self.layer_7[0].weight)
        nn.init.kaiming_normal_(self.layer_8[0].weight)
        nn.init.kaiming_normal_(self.layer_9[0].weight)
        nn.init.kaiming_normal_(self.layer_10[0].weight)

    def forward(self, x):
        # Dimension of input in (batch, 1 bar, input_dim, 1)
        x = x.squeeze(3).squeeze(1)
        # Dimension of x is (batch, input_dim)
        x = self.layer_1(x)
        # Dimension of x is (batch, 1024)
        x = x.unsqueeze(-1).unsqueeze(-1)
        # Dimension of x is (batch, 1024, 1, 1)
        x = self.layer_2(x)
        # Dimension of x is (batch, 512, 2, 1)
        x = self.layer_3(x)
        # Dimension of x is (batch, 256, 4, 1)
        x = self.layer_4(x)
        # Dimension of x is (batch, 256, 8, 1)
        x = self.layer_5(x)
        # Dimension of x is (batch, 256, 16, 1)
        x = self.layer_6(x)
        # Dimension of x is (batch, 256, 32, 1)
        x = self.layer_7(x)
        # Dimension of x is (batch, 256, 96, 1)
        x = self.layer_8(x)
        # Dimension of x is (batch, 256, 96, 4)
        x = self.layer_9(x)
        # Dimension of x is (batch, 128, 96, 16)
        x = self.layer_10(x)
        # Dimension of output is (batch, 1, 96, 128)

        return x