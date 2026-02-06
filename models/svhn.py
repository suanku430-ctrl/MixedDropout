import torch
import torch.nn as nn
from regularization import MixDropout, RadDropout, DropConnect, AdaptiveDropConnect
import torch.nn.functional as F

class SVHNNet(nn.Module):
    def __init__(self, dropout_type='baseline', dropout_params=None):
        super().__init__()
        self.dropout_type = dropout_type
        self.params = dropout_params if dropout_params else {}

        # 简化的VGG结构
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )

        self.fc1 = self._make_fc(4 * 4 * 128, 512)
        self.fc2 = nn.Linear(512, 128)
        self.out = nn.Linear(128, 10)

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
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(self.fc2(x))
        return self.out(x)