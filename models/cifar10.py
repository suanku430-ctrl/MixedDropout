import torch
import torch.nn as nn
from regularization import MixDropout, RadDropout, DropConnect, AdaptiveDropConnect


class Cifar10Net(nn.Module):
    def __init__(self, dropout_type='baseline', dropout_params=None):
        super().__init__()
        self.dropout_type = dropout_type
        self.params = dropout_params if dropout_params else {}

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(3, 2, 1)
        )
        self.fc = self._make_fc(64 * 8 * 8, 1024)
        self.out = nn.Linear(1024, 10)

    def _make_fc(self, in_f, out_f):
        layers = []
        if self.dropout_type == 'mix':
            layers.append(MixDropout(in_f, out_f, **self.params))
        elif self.dropout_type == 'dropconnect':
            layers.append(DropConnect(in_f, out_f, **self.params))
        elif self.dropout_type == 'adaptive':
            layers.append(AdaptiveDropConnect(in_f, out_f))
        else:
            layers.append(nn.Linear(in_f, out_f))

        layers.append(nn.ReLU(inplace=True))

        if self.dropout_type == 'vanilla':
            layers.append(nn.Dropout(self.params.get('p', 0.5)))
        elif self.dropout_type == 'rad':
            layers.append(RadDropout())
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return self.out(x)