
import torch
import torch.nn as nn
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
import csv
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score
from thop import profile, clever_format

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
class GradWeightDrop(nn.Module):
    def __init__(self, device, I_P=0.5, GD_P=0, gd_small=True, name="DropConnectModel"):
        super().__init__()
        self.eps = 1e-9
        self.I_P = I_P
        self.GD_P = GD_P
        self.device = device
        self.gd_small = gd_small
        self.cur_time = 0
        self.final_drop_rate = 0
        self.final_left_rate = 1
        self.drop_name = name
        self.THR = 0.5

    def forward(self, grad, weight, training=True):
        if grad is None or self.cur_time == 0:
            self.cur_time += 1
            return weight
        else:
            if training:
                final_drop_p = torch.full_like(weight, self.I_P)
                if self.gd_small is not None:
                    grad_abs = torch.abs(grad)
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

                final_mask = self._mask(final_drop_p)
                left_rate = torch.sum(final_mask) / final_mask.numel()
                self.final_drop_rate = final_drop_p.mean().item()
                self.final_left_rate = left_rate.item()
                
                self.cur_time += 1
                return (weight * final_mask) / (self.final_left_rate + self.eps)
            else:
                self.cur_time += 1
                return weight

    def _mask(self, p):
        uniform = torch.rand_like(p).to(self.device)
        mask = p >= uniform
        return mask.float()

    def _reset_time(self):
        self.cur_time = 0
        self.final_drop_rate = 0
        self.final_left_rate = 1
class CIFAR10Net(nn.Module):
    def __init__(self, drop_model=None):
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
        self.fc2 = nn.Linear(1024, 10)
        self.drop_model = drop_model
        self.gradients = {}
        self.hook_handles = []
        self.register_hooks()
    def register_hooks(self):
        def hook_fn(module, grad_input, grad_output):
            self.gradients['fc1'] = grad_output[0]
        handle = self.fc1.register_full_backward_hook(hook_fn)
        self.hook_handles.append(handle)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        raw_weight = self.fc1.weight.data.clone()
        if self.drop_model and self.training:
            grad = self.gradients.get('fc1', None)
            modified_weight = self.drop_model(grad, self.fc1.weight)
            self.fc1.weight.data = modified_weight
            x = self.fc1(x)
            self.fc1.weight.data = raw_weight
        else:
            x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

    def __del__(self):
        for handle in self.hook_handles:
            handle.remove()

def setup_dataloaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader

def calculate_flops(model):
    input_tensor = torch.randn(1, 3, 32, 32).to(DEVICE)
    macs, params = profile(model, inputs=(input_tensor, ), verbose=False)
    flops, _ = clever_format([macs, params], "%.3f")
    return flops
def train_model(model, train_loader, test_loader, optimizer, criterion, args, experiment_id, csv_writer, flops_str, save_dir):
    best_acc = 0.0
    results_history = {
        'train_losses': [], 'train_accs': [], 'train_times': [],
        'test_losses': [], 'test_accs': [], 'test_times': [],
        'f1_scores': [], 'precisions': [], 'recalls': []
    }
    for epoch in range(1, args.epochs + 1):
        # --- Training ---
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        if model.drop_model:
            model.drop_model._reset_time()

        train_start_time = time.time()
        for data, target in tqdm(train_loader, desc=f"Exp {experiment_id} | Epoch {epoch}/{args.epochs} [Train]"):
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

        epoch_train_time = time.time() - train_start_time
        train_acc = 100. * correct / total
        avg_train_loss = train_loss / len(train_loader)

        # --- Testing ---
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_preds = []
        all_targets = []

        test_start_time = time.time()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                _, pred = output.max(1)
                test_correct += pred.eq(target).sum().item()
                test_total += target.size(0)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        epoch_test_time = time.time() - test_start_time
        test_acc = 100. * test_correct / test_total
        avg_test_loss = test_loss / len(test_loader)

        # --- Metrics ---
        epoch_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        epoch_precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        epoch_recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        if test_acc > best_acc:
            best_acc = test_acc
            model_save_path = os.path.join(save_dir, f'best_model_exp_{experiment_id}.pth')
            torch.save(model.state_dict(), model_save_path)

        results_history['train_losses'].append(avg_train_loss)
        results_history['train_accs'].append(train_acc)
        results_history['train_times'].append(epoch_train_time)
        results_history['test_losses'].append(avg_test_loss)
        results_history['test_accs'].append(test_acc)
        results_history['test_times'].append(epoch_test_time)
        results_history['f1_scores'].append(epoch_f1)
        results_history['precisions'].append(epoch_precision)
        results_history['recalls'].append(epoch_recall)

        csv_writer.writerow([
            experiment_id, epoch, 
            f"{avg_train_loss:.4f}", f"{train_acc:.2f}", f"{epoch_train_time:.2f}",
            f"{avg_test_loss:.4f}", f"{test_acc:.2f}", f"{epoch_test_time:.2f}",
            f"{epoch_f1:.4f}", f"{epoch_precision:.4f}", f"{epoch_recall:.4f}",
            flops_str, f"{best_acc:.2f}"
        ])

        print(f"Epoch {epoch}/{args.epochs} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Best: {best_acc:.2f}%")

    return best_acc, results_history


def create_visualizations(experiment_id, results, save_dir):
    epochs = range(1, len(results['train_losses']) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plotting code (simplified for brevity, logic same as before)
    axes[0, 0].plot(epochs, results['train_losses'], 'b-', label='Train'); axes[0, 0].plot(epochs, results['test_losses'], 'r-', label='Test')
    axes[0, 0].set_title('Loss'); axes[0, 0].legend(); axes[0, 0].grid(True)
    
    axes[0, 1].plot(epochs, results['train_accs'], 'b-', label='Train'); axes[0, 1].plot(epochs, results['test_accs'], 'r-', label='Test')
    axes[0, 1].set_title('Accuracy'); axes[0, 1].legend(); axes[0, 1].grid(True)
    
    axes[0, 2].plot(epochs, results['train_times'], 'b-', label='Train'); axes[0, 2].plot(epochs, results['test_times'], 'r-', label='Test')
    axes[0, 2].set_title('Time (s)'); axes[0, 2].legend(); axes[0, 2].grid(True)

    axes[1, 0].plot(epochs, results['f1_scores'], 'g-'); axes[1, 0].set_title('F1 Score'); axes[1, 0].grid(True)
    
    axes[1, 1].plot(epochs, results['precisions'], 'c-', label='Prec'); axes[1, 1].plot(epochs, results['recalls'], 'm-', label='Recall')
    axes[1, 1].set_title('Precision & Recall'); axes[1, 1].legend(); axes[1, 1].grid(True)

    axes[1, 2].axis('off')
    axes[1, 2].text(0.1, 0.5, f"Best Acc: {results['best_acc']:.2f}%", fontsize=14)

    plt.tight_layout()

    plot_filename = os.path.join(save_dir, f'plot_exp_{experiment_id}.png')
    plt.savefig(plot_filename)
    plt.close()


def run_experiment(experiment_id, args, csv_writer, save_dir):
    train_loader, test_loader = setup_dataloaders(args.batch_size)
    
    drop_small_gd = GradWeightDrop(
        device=DEVICE,
        I_P=args.gd_init_droprate,
        GD_P=args.gd_droprate,
        gd_small=True,
        name="DropSmallGd"
    )

    model = CIFAR10Net(drop_model=drop_small_gd).to(DEVICE)
    flops_str = calculate_flops(model)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    best_acc, results_history = train_model(
        model, train_loader, test_loader, optimizer, criterion, args, experiment_id, csv_writer, flops_str, save_dir
    )
    total_time = time.time() - start_time

    results_history['best_acc'] = best_acc
    create_visualizations(experiment_id, results_history, save_dir)

    return best_acc, total_time, results_history

def main():
    parser = argparse.ArgumentParser(description='DropSmallGd on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gd_init_droprate', type=float, default=0.3)
    parser.add_argument('--gd_droprate', type=float, default=0.5)
    parser.add_argument('--exp_id', type=int, default=1)
    args = parser.parse_args()


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_name = f"lr_{args.lr}_bs_{args.batch_size}_init_{args.gd_init_droprate}_drop_{args.gd_droprate}_{timestamp}"
    

    base_dir = "experiments/cifar10/ddc"
    save_dir = os.path.join(base_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting Run. Results will be saved to: {save_dir}")


    params_filename = os.path.join(save_dir, 'parameters.json')
    with open(params_filename, 'w') as f:
        json.dump(vars(args), f, indent=4)

    csv_filename = os.path.join(save_dir, 'results_detailed.csv')
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            'experiment_id', 'epoch', 
            'train_loss', 'train_acc', 'train_time',
            'test_loss', 'test_acc', 'test_time',
            'f1_score', 'precision', 'recall', 'flops', 'best_test_acc'
        ])

        best_accs = []
        

        for i in range(args.exp_id):
            print(f"\n>>> Experiment {i + 1}/{args.exp_id}")
            acc, _, _ = run_experiment(i + 1, args, csv_writer, save_dir)
            best_accs.append(acc)


    avg_acc = np.mean(best_accs)
    std_acc = np.std(best_accs)
    summary_path = os.path.join(save_dir, 'summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("Experiment Summary\n==================\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Parameters: {vars(args)}\n\n")
        f.write(f"Results per run: {best_accs}\n")
        f.write(f"Average Best Accuracy: {avg_acc:.2f}%\n")
        f.write(f"Std Dev: {std_acc:.4f}\n")

    print(f"\nAll Done! Saved to: {save_dir}")
    print(f"Avg Acc: {avg_acc:.2f}%")

if __name__ == "__main__":
    main()