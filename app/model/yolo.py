import torch
import torch.nn as nn #contains classes and functions for building neural networks
from .architecture import architecture_config
from .CNNBlock import CNNBlock

# YOLOv1 model definition
class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers.append(CNNBlock(
                    in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3]
                ))
                in_channels = x[1]
            elif type(x) == str:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif type(x) == list:
                conv1, conv2, num_repeats = x
                for _ in range(num_repeats):
                    layers.append(CNNBlock(
                        in_channels,
                        conv1[1],
                        kernel_size=conv1[0],
                        stride=conv1[2],
                        padding=conv1[3]
                    ))
                    layers.append(CNNBlock(
                        conv1[1],
                        conv2[1],
                        kernel_size=conv2[0],
                        stride=conv2[2],
                        padding=conv2[3]
                    ))
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),  # Original paper states this should be 4096
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5))
        )

