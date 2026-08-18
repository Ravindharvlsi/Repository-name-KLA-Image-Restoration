import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, channels=80):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )

    def forward(self, x):
        return x + 0.2 * self.block(x)


class V5RestorationNet(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=80,
        num_blocks=24
    ):
        super().__init__()

        self.head = nn.Conv2d(
            in_channels,
            features,
            3,
            1,
            1
        )

        self.blocks = nn.Sequential(
            *[
                ResidualBlock(features)
                for _ in range(num_blocks)
            ]
        )

        self.body = nn.Conv2d(
            features,
            features,
            3,
            1,
            1
        )

        self.up = nn.Sequential(
            nn.Conv2d(
                features,
                features * 4,
                3,
                1,
                1
            ),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                features,
                features,
                3,
                1,
                1
            ),
            nn.ReLU(inplace=True)
        )

        self.tail = nn.Sequential(
            nn.Conv2d(
                features,
                features,
                3,
                1,
                1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                features,
                out_channels,
                3,
                1,
                1
            )
        )

    def forward(self, x):

        shallow = self.head(x)

        deep = self.blocks(shallow)

        deep = self.body(deep)

        features = shallow + deep

        output = self.up(features)

        output = self.tail(output)

        return torch.clamp(
            output,
            0.0,
            1.0
        )