import os
import json
import csv
import time
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from regularization import MixDropout,DropConnect, AdaptiveDropConnect
from thop import profile

class ExpLogger:
    def __init__(self, dataset, dropout_type, exp_id, params=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        param_str = ""
        if params:
            for k, v in params.items():
                param_str += f"_{k}_{v}"

        self.exp_dir = f"experiments/{dataset}/{dropout_type}/exp_{exp_id}_{timestamp}{param_str}"
        os.makedirs(self.exp_dir, exist_ok=True)

        self.log_file = os.path.join(self.exp_dir, "training_log.csv")
        self.model_file = os.path.join(self.exp_dir, "best_model.pth")
        self.curve_file = os.path.join(self.exp_dir, "curves.png")
        self.results_file = os.path.join(self.exp_dir, "results.json")

        # Init CSV
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'train_time',
                             'test_loss', 'test_acc', 'test_time',
                             'f1', 'precision', 'recall'])

    def log_epoch(self, epoch_data):
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch_data['epoch'],
                f"{epoch_data['train_loss']:.4f}", f"{epoch_data['train_acc']:.2f}", f"{epoch_data['train_time']:.2f}",
                f"{epoch_data['test_loss']:.4f}", f"{epoch_data['test_acc']:.2f}", f"{epoch_data['test_time']:.2f}",
                f"{epoch_data['f1']:.4f}", f"{epoch_data['precision']:.4f}", f"{epoch_data['recall']:.4f}"
            ])

    def save_results(self, results):
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=4)

    def plot_curves(self, history):
        epochs = range(1, len(history['train_loss']) + 1)
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        axs[0, 0].plot(epochs, history['train_loss'], label='Train Loss')
        axs[0, 0].plot(epochs, history['test_loss'], label='Test Loss')
        axs[0, 0].set_title('Loss')
        axs[0, 0].legend()

        # Accuracy
        axs[0, 1].plot(epochs, history['train_acc'], label='Train Acc')
        axs[0, 1].plot(epochs, history['test_acc'], label='Test Acc')
        axs[0, 1].set_title('Accuracy')
        axs[0, 1].legend()

        # F1 & Metrics
        axs[1, 0].plot(epochs, history['f1'], label='F1-Score')
        axs[1, 0].set_title('F1 Score')
        axs[1, 0].legend()

        axs[1, 1].axis('off')  # Placeholder

        plt.tight_layout()
        plt.savefig(self.curve_file)
        plt.close()


def calculate_flops(model, input_size, device):

    dummy_input = torch.randn(1, *input_size).to(device)
    model.eval()


    def count_custom_linear(m, x, y):

        total_ops = m.in_features * m.out_features
        m.total_ops += torch.DoubleTensor([total_ops])


    custom_ops = {
        MixDropout: count_custom_linear,
        DropConnect: count_custom_linear,
        AdaptiveDropConnect: count_custom_linear,
    }

    flops, params = profile(model, inputs=(dummy_input,), custom_ops=custom_ops, verbose=False)

    return flops, params

def evaluate_metrics(model, loader, criterion, device):
    model.eval()
    test_loss = 0
    all_preds = []
    all_targets = []
    start_time = time.time()

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = output.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    end_time = time.time()
    test_time = end_time - start_time

    test_loss /= len(loader)
    acc = accuracy_score(all_targets, all_preds) * 100
    f1 = f1_score(all_targets, all_preds, average='macro')
    prec = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_targets, all_preds, average='macro')

    return {
        'loss': test_loss, 'acc': acc, 'time': test_time,
        'f1': f1, 'precision': prec, 'recall': rec
    }