import torch
import torch.nn as nn
import torch.nn.functional as F


class RCAB(nn.Module):

    def __init__(self, channels=64, reduction=16):

        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(
                channels,
                channels // reduction,
                1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels // reduction,
                channels,
                1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        residual = self.body(x)

        attention = self.attention(residual)

        return x + residual * attention


class V4RestorationNet(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=64,
        num_blocks=20
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
                RCAB(features)
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

        output = self.upsample(features)

        output = self.tail(output)

        return torch.clamp(
            output,
            0.0,
            1.0
        )