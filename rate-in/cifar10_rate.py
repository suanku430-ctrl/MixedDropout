# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 20:01:04 2026

@author: Admin
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import time
import argparse
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Callable, Optional
from sklearn.metrics import f1_score, precision_score, recall_score
from datetime import datetime  # <--- 新增引入

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class OptimizerConfig:
    max_iterations: int = 100
    learning_rate: float = 0.10
    decay_rate: float = 0.9
    stopping_error: float = 0.01
def calc_information_loss_fn(pre_dropout: torch.Tensor, post_dropout: torch.Tensor, properties: dict) -> torch.Tensor:
    cov_pre = pre_dropout.std() / (torch.abs(pre_dropout).mean() + 1e-8)
    cov_post = post_dropout.std() / (torch.abs(post_dropout).mean() + 1e-8)
    return torch.abs(cov_pre - cov_post)

class AdaptiveInformationDropout(nn.Module):
    def __init__(
            self,
            initial_p: float,
            calc_information_loss: Callable,
            information_loss_threshold: float = 0.10,
            optimizer_config: Optional[OptimizerConfig] = None,
            name: str = "",
            **kwargs
    ):
        super().__init__()
        self.p = torch.nn.Parameter(torch.tensor(initial_p), requires_grad=False)
        self.calc_information_loss = calc_information_loss
        self.information_loss_threshold = information_loss_threshold
        self.optimizer_config = optimizer_config or OptimizerConfig()
        self.name = name
        self.additional_properties = kwargs.get('properties', {})

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

        return np.clip(self.p.item(), 0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            optimized_rate = self._optimize_dropout_rate(x)
            return self._apply_dropout(x, optimized_rate)
        return x

# =====================================================
# =====================================================
class CIFAR10Net(nn.Module):
    def __init__(self, use_rate_in=False, initial_p=0.5):
        super(CIFAR10Net, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.fc1 = nn.Linear(64 * 8 * 8, 1024)
        self.relu1 = nn.ReLU()

        self.use_rate_in = use_rate_in
        if use_rate_in:
            self.rate_in = AdaptiveInformationDropout(
                initial_p=initial_p,
                calc_information_loss=calc_information_loss_fn,
                information_loss_threshold=0.10,
                name="Rate-In"
            )
            
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        
        if self.use_rate_in and self.training:
            x = self.rate_in(x)
            
        x = self.fc2(x)
        return x

# =====================================================

# =====================================================
def setup_dataloaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader

# =====================================================

# =====================================================
def run_training(args, save_dir):
    train_loader, test_loader = setup_dataloaders(args.batch_size)
    model = CIFAR10Net(use_rate_in=True, initial_p=args.rate_in_initial_p).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    history = []
    
    print(f"Model initialized on {DEVICE}. Starting training...")

    for epoch in range(1, args.epochs + 1):
        # --- Train Loop ---
        model.train()
        train_start = time.time()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", leave=False)
        for data, target in pbar:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            pbar.set_postfix({'loss': loss.item()})
            
        train_time = time.time() - train_start
        train_acc = 100. * correct / total
        train_avg_loss = train_loss / len(train_loader)

        # --- Test Loop ---
        model.eval()
        test_start = time.time()
        test_loss = 0.0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc=f"Epoch {epoch}/{args.epochs} [Test]", leave=False):
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                
                _, predicted = output.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        test_time = time.time() - test_start
        test_avg_loss = test_loss / len(test_loader)
        
        test_acc = 100. * sum(np.array(all_preds) == np.array(all_targets)) / len(all_targets)
        f1 = f1_score(all_targets, all_preds, average='macro')
        precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)

        # --- Save Best Model ---
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            is_best_str = "(*)"
        else:
            is_best_str = ""

        print(f"Epoch {epoch:03d} | "
              f"Train: Loss={train_avg_loss:.4f}, Acc={train_acc:.2f}%, Time={train_time:.1f}s | "
              f"Test: Loss={test_avg_loss:.4f}, Acc={test_acc:.2f}%, Time={test_time:.1f}s | "
              f"F1: {f1:.4f} {is_best_str}")

        history.append({
            'epoch': epoch,
            'train_loss': train_avg_loss,
            'train_acc': train_acc,
            'train_time': train_time,
            'test_loss': test_avg_loss,
            'test_acc': test_acc,
            'test_time': test_time,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        })

    df = pd.DataFrame(history)
    df.to_csv(os.path.join(save_dir, "training_history.csv"), index=False)
    
    print(f"\nTraining Finished. Best Accuracy: {best_acc:.2f}%")
    return df

def plot_results(df, save_dir):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['test_loss'], label='Test Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(df['epoch'], df['train_acc'], label='Train Acc')
    plt.plot(df['epoch'], df['test_acc'], label='Test Acc')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(df['epoch'], df['f1_score'], label='F1')
    plt.plot(df['epoch'], df['precision'], label='Precision')
    plt.plot(df['epoch'], df['recall'], label='Recall')
    plt.title('Metrics (Macro)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(df['epoch'], df['train_time'], label='Train Time')
    plt.plot(df['epoch'], df['test_time'], label='Test Time')
    plt.title('Time Consumption (s)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics_plot.png"))
    print(f"Plots saved to {os.path.join(save_dir, 'metrics_plot.png')}")

# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rate-In CIFAR10 Experiment")
    
    parser.add_argument('--exp_id', type=str, default='1', help='Repeats')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs per run')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch Size')
    parser.add_argument('--rate_in_initial_p', type=float, default=0.5, help='Init Rate')
    
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    

    base_dir = "experiments/cifar10/rate_in"
    folder_name = f"exp_{args.exp_id}_{timestamp}"
    save_dir = os.path.join(base_dir, folder_name)
    
    os.makedirs(save_dir, exist_ok=True)
    

    with open(os.path.join(save_dir, "config.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write(f"timestamp: {timestamp}\n")
            
    print("="*60)
    print(f"Starting Experiment: {args.exp_id}")
    print(f"Time: {timestamp}")
    print(f"Save Directory: {save_dir}")
    print(f"Config: LR={args.lr}, Epochs={args.epochs}, Batch={args.batch_size}")
    print("="*60)
    
    # 运行
    history_df = run_training(args, save_dir)
    

    plot_results(history_df, save_dir)