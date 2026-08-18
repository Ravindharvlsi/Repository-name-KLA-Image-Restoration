import torch
import torch.nn as nn


class ResidualDenseBlock(nn.Module):

    def __init__(self, channels=96, growth=32):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, growth, 3, 1, 1)
        self.conv2 = nn.Conv2d(
            channels + growth,
            growth,
            3, 1, 1
        )
        self.conv3 = nn.Conv2d(
            channels + growth * 2,
            growth,
            3, 1, 1
        )

        self.conv4 = nn.Conv2d(
            channels + growth * 3,
            channels,
            3, 1, 1
        )

        self.act = nn.LeakyReLU(
            0.1,
            inplace=True
        )

    def forward(self, x):

        x1 = self.act(
            self.conv1(x)
        )

        x2 = self.act(
            self.conv2(
                torch.cat([x, x1], dim=1)
            )
        )

        x3 = self.act(
            self.conv3(
                torch.cat([x, x1, x2], dim=1)
            )
        )

        x4 = self.conv4(
            torch.cat(
                [x, x1, x2, x3],
                dim=1
            )
        )

        return x + 0.2 * x4


class V3RestorationNet(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=96,
        num_blocks=12
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
                ResidualDenseBlock(
                    features,
                    32
                )
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

        self.upsample = nn.Sequential(

            nn.Conv2d(
                features,
                features * 4,
                3,
                1,
                1
            ),

            nn.PixelShuffle(2),

            nn.LeakyReLU(
                0.1,
                inplace=True
            ),

            nn.Conv2d(
                features,
                features,
                3,
                1,
                1
            ),

            nn.LeakyReLU(
                0.1,
                inplace=True
            )
        )

        self.tail = nn.Sequential(

            nn.Conv2d(
                features,
                features,
                3,
                1,
                1
            ),

            nn.LeakyReLU(
                0.1,
                inplace=True
            ),

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

        up = self.upsample(features)

        output = self.tail(up)

        return torch.clamp(
            output,
            0.0,
            1.0
        )