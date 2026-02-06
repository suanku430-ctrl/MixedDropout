import torch
import torch.nn as nn
from regularization import MixDropout, RadDropout, DropConnect, AdaptiveDropConnect


class MnistNet(nn.Module):
    def __init__(self, dropout_type='baseline', dropout_params=None):
        super().__init__()
        self.dropout_type = dropout_type
        self.params = dropout_params if dropout_params else {}

        self.fc1 = nn.Linear(784, 1024)
        self.fc2_block = self._make_layer(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 10)
        self.relu = nn.ReLU()

    def _make_layer(self, in_f, out_f):
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
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.fc2_block(x)
        x = self.relu(self.fc3(x))
        return self.fc4(x)