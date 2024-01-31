import torch
from torch import nn
import numpy as np

class MarioNet(nn.Module):
    def __init__(self, input_dimensions, action_space, frozen=False):
        super(MarioNet, self).__init__()
        self.input_dimensions = input_dimensions
        self.action_space = action_space
        self.is_frozen = frozen

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_dimensions[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        output_size = self._calculate_conv_output(input_dimensions)

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(output_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_space)
        )

        if self.is_frozen:
            self._freeze_network()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        conv_out = self.conv_layers(x)
        return self.fc_layers(conv_out)

    def _calculate_conv_output(self, shape):
        dummy_input = torch.zeros(1, *shape)
        output = self.conv_layers(dummy_input)
        return int(np.prod(output.size()))

    def _freeze_network(self):
        for param in self.fc_layers.parameters():
            param.requires_grad = False
