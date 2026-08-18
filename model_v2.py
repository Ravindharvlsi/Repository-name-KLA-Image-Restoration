import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, channels=64, res_scale=0.1):
        super().__init__()

        self.res_scale = res_scale

        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.res_scale * self.body(x)


class RestorationNetV2(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=64,
        num_blocks=16
    ):
        super().__init__()

        self.head = nn.Conv2d(
            in_channels,
            features,
            3,
            1,
            1
        )

        self.body = nn.Sequential(
            *[
                ResidualBlock(features)
                for _ in range(num_blocks)
            ]
        )

        self.body_conv = nn.Conv2d(
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

        body = self.body(shallow)

        body = self.body_conv(body)

        features = shallow + body

        up = self.upsample(features)

        output = self.tail(up)

        return torch.clamp(output, 0.0, 1.0)