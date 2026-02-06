# -*- coding: utf-8 -*-
"""
Modified NORB Rate-In Experiment Code (Fixed)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import struct
import time
import random
import json
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from dataclasses import dataclass, asdict
from typing import Callable, Optional
from sklearn.metrics import f1_score, precision_score, recall_score
from thop import profile

@dataclass
class OptimizerConfig:
    max_iterations: int = 100
    learning_rate: float = 0.10
    decay_rate: float = 0.9
    stopping_error: float = 0.01

class AdaptiveInformationDropout(nn.Module):
    def __init__(
        self,
        initial_p: float = 0.5,
        calc_information_loss: Optional[Callable] = None,
        information_loss_threshold: float = 0.10,
        optimizer_config: Optional[OptimizerConfig] = None,
        name: str = "",
        verbose: int = 0,
        **kwargs
    ):
        super().__init__()
        if not 0 <= initial_p <= 1:
            raise ValueError("Initial dropout probability must be between 0 and 1")
        
        if calc_information_loss is None:
            calc_information_loss = self._default_calc_information_loss

        self.p = torch.nn.Parameter(torch.tensor(initial_p), requires_grad=False)
        self.calc_information_loss = calc_information_loss
        self.information_loss_threshold = information_loss_threshold
        self.optimizer_config = optimizer_config or OptimizerConfig()
        self.name = name
        self.verbose = verbose
        self.additional_properties = kwargs.get('properties', {})
        
        self.current_rate = initial_p
        self.rate_history = []

    def _default_calc_information_loss(self, pre_dropout: torch.Tensor, post_dropout: torch.Tensor, properties: dict) -> torch.Tensor:
        cov_pre = pre_dropout.std() / (torch.abs(pre_dropout).mean() + 1e-8)
        cov_post = post_dropout.std() / (torch.abs(post_dropout).mean() + 1e-8)
        return torch.abs(cov_pre - cov_post)

    def _apply_dropout(self, x: torch.Tensor, rate: float) -> torch.Tensor:
        return F.dropout(x, p=rate, training=self.training)

    def _optimize_dropout_rate(self, x: torch.Tensor) -> float:
        pre_dropout = x.detach()
        config = self.optimizer_config

        for iteration in range(config.max_iterations):
            current_rate = np.clip(self.p.item(), 0, 1)
            post_dropout = self._apply_dropout(pre_dropout, current_rate)

            info_loss = self.calc_information_loss(
                pre_dropout=pre_dropout,
                post_dropout=post_dropout,
                properties=self.additional_properties
            )

            error = info_loss.item() - self.information_loss_threshold
            current_lr = (config.learning_rate * config.decay_rate
                          if iteration % 10 == 0 else config.learning_rate)

            updated_rate = current_rate - current_lr / (1 + abs(error)) * error
            self.p.data = torch.tensor(np.clip(updated_rate, 0, 1))

            if abs(error) < config.stopping_error:
                break

        optimized_rate = np.clip(self.p.item(), 0, 1)
        self.current_rate = optimized_rate
        self.rate_history.append(optimized_rate)
        
        return optimized_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            optimized_rate = self._optimize_dropout_rate(x)
            return self._apply_dropout(x, optimized_rate)
        return x

    def get_current_rate(self):
        return self.current_rate

    def get_rate_history(self):
        return self.rate_history


# ================== Dataset and Model ==================

class NORBDataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.transform = transform
        if train:
            self.data_file = os.path.join(data_dir, 'smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat')
            self.label_file = os.path.join(data_dir, 'smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat')
        else:
            self.data_file = os.path.join(data_dir, 'smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat')
            self.label_file = os.path.join(data_dir, 'smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat')

        self.data = self._load_data(self.data_file)
        self.labels = self._load_labels(self.label_file)

    def _load_data(self, file_path):
        with open(file_path, 'rb') as f:
            struct.unpack('<i', f.read(4))
            num_dims = struct.unpack('<i', f.read(4))[0]
            dims = struct.unpack('<' + 'i' * num_dims, f.read(4 * num_dims))
            total_num = np.prod(dims)
            data = np.fromfile(f, dtype=np.uint8, count=total_num)
            data = data.reshape(dims)
            data = np.transpose(data, (0, 2, 3, 1))
        return data

    def _load_labels(self, file_path):
        with open(file_path, 'rb') as f:
            struct.unpack('<i', f.read(4))
            num_dims = struct.unpack('<i', f.read(4))[0]
            dims = struct.unpack('<' + 'i' * num_dims, f.read(4 * num_dims))
            labels = np.fromfile(f, dtype=np.int32, count=dims[0])
        return labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx].astype(np.float32)
        label = int(self.labels[idx])
        label = torch.tensor(label, dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label


class NORBVGG(nn.Module):
    def __init__(self, dropout_params=None):
        super(NORBVGG, self).__init__()
        self.dropout_params = dropout_params if dropout_params is not None else {}

        self.features = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = self._build_fc_layers()
        self.apply(self.init_weights)

    def _build_fc_layers(self):
        modules = []
        in_features = 512 * 3 * 3

        initial_p = self.dropout_params.get('initial_p', 0.5)
        information_loss_threshold = self.dropout_params.get('information_loss_threshold', 0.10)
        optimizer_config = self.dropout_params.get('optimizer_config', OptimizerConfig())

        modules.append(nn.Linear(in_features, 4096))
        modules.append(nn.ReLU(inplace=True))
        
        modules.append(AdaptiveInformationDropout(
            initial_p=initial_p,
            information_loss_threshold=information_loss_threshold,
            optimizer_config=optimizer_config,
            verbose=0
        ))

        modules.append(nn.Linear(4096, 4096))
        modules.append(nn.ReLU(inplace=True))
        
        modules.append(AdaptiveInformationDropout(
            initial_p=initial_p,
            information_loss_threshold=information_loss_threshold,
            optimizer_config=optimizer_config,
            verbose=0
        ))

        modules.append(nn.Linear(4096, 5))
        return nn.Sequential(*modules)

    def init_weights(self, layer):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                layer.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x

    def get_dropout_rates(self):
        rates = []
        for module in self.fc_layers:
            if isinstance(module, AdaptiveInformationDropout):
                rates.append(module.get_current_rate())
        return rates

# ================== Utilities ==================

def calculate_flops(model, input_size=(1, 2, 96, 96), device='cpu'):
    dummy_input = torch.randn(input_size).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    return flops
def calculate_metrics(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    return f1, precision, recall

def setup_dataloaders(data_dir, batch_size=128):
    nor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x if x.shape[0] == 2 else x[:2]),
        transforms.Normalize((0.5, 0.5), (0.5, 0.5))
    ])

    train_dataset = NORBDataset(data_dir, train=True, transform=nor_transform)
    test_dataset = NORBDataset(data_dir, train=False, transform=nor_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


def setup_experiment(exp_id, dropout_params=None, lr=0.00001):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    param_str = f"lr{lr}"
    if dropout_params:
        init_p = dropout_params.get('initial_p', 0.5)
        thresh = dropout_params.get('information_loss_threshold', 0.1)
        param_str += f"_p{init_p}_th{thresh}"

    base_dir = "experiments/norb/rate-in"
    folder_name = f"run_{timestamp}_exp{exp_id}_{param_str}"
    exp_dir = os.path.join(base_dir, folder_name)
    
    os.makedirs(exp_dir, exist_ok=True)

    return {
        'exp_dir': exp_dir,
        'log_file': os.path.join(exp_dir, "training_log.csv"),
        'config_file': os.path.join(exp_dir, "config.json"),
        'model_file': os.path.join(exp_dir, "best_model.pth"),
        'results_file': os.path.join(exp_dir, "results.json")
    }

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    start_time = time.time()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    train_time = time.time() - start_time
    avg_loss = epoch_loss / len(train_loader)
    train_acc = 100. * correct / total
    return avg_loss, train_acc, train_time

def evaluate(model, test_loader, criterion, device):
    model.eval()
    start_time = time.time()
    correct = 0
    total = 0
    epoch_loss = 0.0
    
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            epoch_loss += loss.item()

            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    test_time = time.time() - start_time
    avg_loss = epoch_loss / len(test_loader)
    test_acc = 100. * correct / total
    
    f1, prec, rec = calculate_metrics(all_targets, all_preds)
    
    return test_acc, avg_loss, test_time, f1, prec, rec

def run_experiment(exp_id, data_dir, lr, dropout_params=None, num_epochs=200, batch_size=128):
    # Setup
    paths = setup_experiment(exp_id, dropout_params, lr)
    train_loader, test_loader = setup_dataloaders(data_dir, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config
    serializable_params = {}
    if dropout_params:
        for key, value in dropout_params.items():
            if isinstance(value, OptimizerConfig):
                serializable_params[key] = asdict(value)
            else:
                serializable_params[key] = value

    config = {
        'dropout_type': 'ratein',
        'exp_id': exp_id,
        'dropout_params': serializable_params,
        'learning_rate': lr,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'device': str(device),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(paths['config_file'], 'w') as f:
        json.dump(config, f, indent=2)


    model = NORBVGG(dropout_params=dropout_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    

    flops = calculate_flops(model, device=device)
    print(f"Model FLOPs: {flops:,.0f}")

    with open(paths['log_file'], 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 
            'train_loss', 'train_acc', 'train_time',
            'test_loss', 'test_acc', 'test_time',
            'f1_score', 'precision', 'recall', 'flops',
            'best_test_acc', 'dropout_rate_1', 'dropout_rate_2', 'timestamp'
        ])

    best_acc = 0.0

    print(f"\n{'=' * 50}")
    print(f"Starting Experiment {exp_id}. Logs: {paths['exp_dir']}")

    for epoch in range(1, num_epochs + 1):

        train_loss, train_acc, train_time = train_epoch(model, train_loader, criterion, optimizer, device)
        

        test_acc, test_loss, test_time, f1, prec, rec = evaluate(model, test_loader, criterion, device)
        

        current_rates = model.get_dropout_rates()
        rate1 = current_rates[0] if len(current_rates) > 0 else 0
        rate2 = current_rates[1] if len(current_rates) > 1 else 0

        # Checkpoint
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': config
            }, paths['model_file'])

        # Log
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(paths['log_file'], 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, 
                f"{train_loss:.6f}", f"{train_acc:.4f}", f"{train_time:.4f}",
                f"{test_loss:.6f}", f"{test_acc:.4f}", f"{test_time:.4f}",
                f"{f1:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{flops:.0f}",
                f"{best_acc:.4f}", f"{rate1:.4f}", f"{rate2:.4f}", current_time
            ])

        print(f"Ep {epoch:3d} | Train: {train_loss:.4f}/{train_acc:.2f}% ({train_time:.1f}s) | "
              f"Test: {test_loss:.4f}/{test_acc:.2f}% ({test_time:.1f}s) | "
              f"Best: {best_acc:.2f}% | Rates: {rate1:.3f}, {rate2:.3f}")

    # Final Summary
    with open(os.path.join(paths['exp_dir'], "results_summary.txt"), 'w') as f:
        f.write(f"Best Test Accuracy: {best_acc:.2f}%\n")
        f.write(f"Final Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Total Epochs: {num_epochs}\n")
        f.write(f"Model FLOPs: {flops:,.0f}\n")
        f.write(f"Params: {dropout_params}\n")

    return best_acc

def main():
    parser = argparse.ArgumentParser(description='Rate-In Dropout NORB (Modified)')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to NORB dataset')
    parser.add_argument('--num_experiments', type=int, default=1, help='Repeats')
    parser.add_argument('--num_epochs', type=int, default=200, help='Epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--initial_p', type=float, default=0.5, help='Init Rate')
    parser.add_argument('--threshold', type=float, default=0.10, help='Info Loss Threshold')

    args = parser.parse_args()

    dropout_params = {
        'initial_p': args.initial_p,
        'information_loss_threshold': args.threshold,
        'optimizer_config': OptimizerConfig()
    }

    print(f"Starting {args.num_experiments} experiments on NORB...")
    
    best_accs = []
    for i in range(1, args.num_experiments + 1):
        print(f"\n>>> Running Experiment {i}/{args.num_experiments}")
        acc = run_experiment(
            exp_id=i, 
            data_dir=args.data_dir,
            lr=args.lr,
            dropout_params=dropout_params, 
            num_epochs=args.num_epochs, 
            batch_size=args.batch_size
        )
        best_accs.append(acc)

    print("\n" + "="*50)
    print("FINAL SUMMARY ALL RUNS")
    print("="*50)
    print(f"Results saved in: experiments/norb/rate-in/")
    print(f"Avg Best Accuracy: {np.mean(best_accs):.2f}% +/- {np.std(best_accs):.2f}")
    print(f"Best Run: {max(best_accs):.2f}%")

if __name__ == "__main__":
    main()