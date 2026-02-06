# -*- coding: utf-8 -*-
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
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, precision_score, recall_score
from thop import profile



class DDCModule(nn.Module):
    def __init__(self, in_features, out_features, I_P=0.3, GD_P=0.5, gd_small=True, name="DDC"):
        super().__init__()
        self.eps = 1e-9
        self.I_P = I_P
        self.GD_P = GD_P
        self.gd_small = gd_small
        self.drop_name = name
        self.THR = 0.5
        
        self.linear = nn.Linear(in_features, out_features)
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.current_input = input[0].detach().clone()
        self.linear.register_forward_hook(forward_hook)
        
        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0].clone()
        self.linear.register_full_backward_hook(backward_hook)

    def apply_ddc(self, weight):
        if self.gradients is None or not self.training:
            return weight
            
        final_drop_p = torch.full_like(weight, self.I_P)

        if self.gd_small is not None:
            grad_abs = torch.abs(self.gradients)
            grad_abs_mean = torch.mean(grad_abs, dim=0)

            gd_mean = torch.mean(grad_abs_mean)
            gd_std = torch.std(grad_abs_mean)

            standard_gd = (grad_abs_mean - gd_mean) / (gd_std + self.eps)
            gd_sigmoid = torch.sigmoid(standard_gd)

            if self.gd_small:
                gd_sigmoid = 1 - gd_sigmoid

            thr_gd_sigmoid = (gd_sigmoid > self.THR)
            gd_sigmoid = thr_gd_sigmoid * gd_sigmoid
            gd_sigmoid_expanded = gd_sigmoid.unsqueeze(1).expand_as(weight)

            final_drop_p += self.GD_P * gd_sigmoid_expanded

        uniform = torch.rand_like(weight)
        final_mask = (final_drop_p >= uniform).float()
        left_rate = torch.sum(final_mask) / final_mask.numel()

        modified_weight = (weight * final_mask) / (left_rate + self.eps)
        return modified_weight

    def forward(self, x):
        if self.training and self.gradients is not None:
            modified_weight = self.apply_ddc(self.linear.weight)
            x = F.linear(x, modified_weight, self.linear.bias)
        else:
            x = self.linear(x)
        return x


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
            nn.Conv2d(2, 64, kernel_size=3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self._build_fc_layers()
        self.apply(self.init_weights)

    def _build_fc_layers(self):
        in_features = 512 * 3 * 3
        I_P = self.dropout_params.get('I_P', 0.3)
        GD_P = self.dropout_params.get('GD_P', 0.5)
        gd_small = self.dropout_params.get('gd_small', True)

        self.ddc1 = DDCModule(in_features, 4096, I_P=I_P, GD_P=GD_P, gd_small=gd_small, name="DDC_FC1")
        self.relu1 = nn.ReLU(inplace=False)
        self.ddc2 = DDCModule(4096, 4096, I_P=I_P, GD_P=GD_P, gd_small=gd_small, name="DDC_FC2")
        self.relu2 = nn.ReLU(inplace=False)
        self.fc3 = nn.Linear(4096, 5)

    def init_weights(self, layer):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                layer.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.ddc1(x)
        x = self.relu1(x)
        x = self.ddc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


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

# ================== Training and Evaluation ==================

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
    
    return avg_loss, test_acc, test_time, f1, prec, rec

# ================== Experiment Flow ==================

def run_experiment(exp_id, args, dropout_params):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"norb_ddc_{timestamp}_lr{args.lr}_bs{args.batch_size}_IP{args.I_P}_GDP{args.GD_P}"
    
    base_dir = "experiments/norb/ddc"
    save_dir = os.path.join(base_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n>>> Starting Experiment {exp_id}")
    print(f">>> Results will be saved to: {save_dir}")


    config = vars(args)
    config['dropout_params'] = dropout_params
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = setup_dataloaders(args.data_dir, args.batch_size)
    model = NORBVGG(dropout_params=dropout_params).to(device)
    

    flops = calculate_flops(model, device=device)
    print(f">>> Model FLOPs: {flops:,.0f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    csv_file = os.path.join(save_dir, 'results.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 
            'train_loss', 'train_acc', 'train_time',
            'test_loss', 'test_acc', 'test_time',
            'f1_score', 'precision', 'recall', 'flops',
            'best_test_acc'
        ])

    best_acc = 0.0
    model_save_path = os.path.join(save_dir, 'best_model.pth')


    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in range(1, args.num_epochs + 1):

        train_loss, train_acc, train_time = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, test_time, f1, prec, rec = evaluate(model, test_loader, criterion, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), model_save_path)
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, 
                f"{train_loss:.6f}", f"{train_acc:.4f}", f"{train_time:.4f}",
                f"{test_loss:.6f}", f"{test_acc:.4f}", f"{test_time:.4f}",
                f"{f1:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{flops:.0f}",
                f"{best_acc:.4f}"
            ])

        print(f"Epoch {epoch}/{args.num_epochs} | "
              f"Train: {train_loss:.4f}/{train_acc:.2f}% ({train_time:.1f}s) | "
              f"Test: {test_loss:.4f}/{test_acc:.2f}% ({test_time:.1f}s) | "
              f"Best: {best_acc:.2f}% | F1: {f1:.4f}")

    try:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.title('Loss Curve')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Acc')
        plt.plot(test_accs, label='Test Acc')
        plt.title('Accuracy Curve')
        plt.legend()
        
        plt.savefig(os.path.join(save_dir, 'training_curves.png'))
        plt.close()
    except Exception as e:
        print(f"Error plotting curves: {e}")

    with open(os.path.join(save_dir, 'summary.txt'), 'w') as f:
        f.write(f"Best Accuracy: {best_acc:.2f}%\n")
        f.write(f"Final Accuracy: {test_acc:.2f}%\n")
        f.write(f"Total Epochs: {args.num_epochs}\n")
        f.write(f"FLOPs: {flops}\n")
        f.write(f"Parameters: {dropout_params}\n")

    return best_acc

def main():
    parser = argparse.ArgumentParser(description='Run DDC experiments on NORB (Modified)')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to NORB dataset directory')
    parser.add_argument('--num_experiments', type=int, default=1, help='Number of experiments')
    parser.add_argument('--num_epochs', type=int, default=200, help='Epochs per experiment')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--I_P', type=float, default=0.3, help='Initial dropout rate')
    parser.add_argument('--GD_P', type=float, default=0.5, help='Gradient dropout rate')
    parser.add_argument('--no_gd_small', action='store_false', dest='gd_small', help='Drop large gradients instead')
    parser.set_defaults(gd_small=True)
    args = parser.parse_args()

    dropout_params = {
        'I_P': args.I_P,
        'GD_P': args.GD_P,
        'gd_small': args.gd_small
    }

    best_accs = []
    
    for i in range(1, args.num_experiments + 1):
        acc = run_experiment(i, args, dropout_params)
        best_accs.append(acc)

    print("\n" + "="*50)
    print("FINAL SUMMARY ALL EXPERIMENTS")
    print("="*50)
    print(f"Results saved in: experiments/norb/ddc")
    print(f"Average Best Acc: {np.mean(best_accs):.2f}% +/- {np.std(best_accs):.2f}")
    print(f"Max Best Acc: {np.max(best_accs):.2f}%")

if __name__ == "__main__":
    main()