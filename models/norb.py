import os
import struct
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from regularization import MixDropout, RadDropout, DropConnect, AdaptiveDropConnect


class NORBDataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.transform = transform
        prefix = 'smallnorb-5x46789x9x18x6x2x96x96-training' if train else 'smallnorb-5x01235x9x18x6x2x96x96-testing'
        self.data = self._load_data(os.path.join(data_dir, f'{prefix}-dat.mat'))
        self.labels = self._load_labels(os.path.join(data_dir, f'{prefix}-cat.mat'))

    def _load_data(self, file_path):
        with open(file_path, 'rb') as f:
            struct.unpack('<i', f.read(4))
            num_dims = struct.unpack('<i', f.read(4))[0]
            dims = struct.unpack('<' + 'i' * num_dims, f.read(4 * num_dims))
            data = np.fromfile(f, dtype=np.uint8, count=np.prod(dims))
            data = data.reshape(dims).transpose(0, 2, 3, 1)  # (N, H, W, C)
        return data

    def _load_labels(self, file_path):
        with open(file_path, 'rb') as f:
            struct.unpack('<i', f.read(4))
            num_dims = struct.unpack('<i', f.read(4))[0]
            dims = struct.unpack('<' + 'i' * num_dims, f.read(4 * num_dims))
            labels = np.fromfile(f, dtype=np.int32, count=dims[0])
        return labels

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx].astype(np.float32)
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        if self.transform: img = self.transform(img)
        return img, label


class NORBNet(nn.Module):
    def __init__(self, dropout_type='baseline', dropout_params=None):
        super().__init__()
        self.dropout_type = dropout_type
        self.params = dropout_params if dropout_params else {}

        # NORB Input: 96x96, 2 channels (stereo)
        self.features = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )  # Output: 512 x 3 x 3

        self.fc1 = self._make_fc(512 * 3 * 3, 4096)
        self.fc2 = self._make_fc(4096, 4096)
        self.out = nn.Linear(4096, 5)

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
        x = self.fc2(x)
        return self.out(x)