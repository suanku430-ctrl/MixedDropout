import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import time
import random
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from thop import profile


class DDCModule(nn.Module):
    def __init__(self, device, I_P=0.5, GD_P=0, gd_small=True, name="DDCModule"):
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

                if self.cur_time == 1:
                     pass 


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


class DDCLinear(nn.Module):
    def __init__(self, in_features, out_features, device, ddc_params=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device

        ddc_params = ddc_params if ddc_params is not None else {}
        I_P = ddc_params.get('I_P', 0.5)
        GD_P = ddc_params.get('GD_P', 0)
        gd_small = ddc_params.get('gd_small', True)

        self.linear = nn.Linear(in_features, out_features)
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear.bias)

        self.ddc = DDCModule(
            device=device,
            I_P=I_P,
            GD_P=GD_P,
            gd_small=gd_small,
            name=f"DDC_Linear_{in_features}_{out_features}"
        )

        self.gradients = {}
        self.hook_handles = []
        self.register_hooks()

    def register_hooks(self):
        def hook_fn(module, grad_input, grad_output):
            self.gradients['linear'] = grad_output[0]

        handle = self.linear.register_full_backward_hook(hook_fn)
        self.hook_handles.append(handle)

    def forward(self, x):
        raw_weight = self.linear.weight.data.clone()

        if self.training:
            grad = self.gradients.get('linear', None)
            modified_weight = self.ddc(grad, self.linear.weight)
            self.linear.weight.data = modified_weight
            output = self.linear(x)
            self.linear.weight.data = raw_weight
        else:
            output = self.linear(x)

        return output

    def __del__(self):
        for handle in self.hook_handles:
            handle.remove()


class FCNet(nn.Module):
    def __init__(self, ddc_params=None):
        super().__init__()
        self.ddc_params = ddc_params if ddc_params is not None else {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = DDCLinear(1024, 1024, self.device, self.ddc_params)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 10)
        self.relu = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.relu(self.fc3(x))
        return self.fc4(x)


# ================== Utility Functions ==================

def calculate_flops(model, input_shape=(1, 1, 28, 28), device='cpu'):
    dummy_input = torch.randn(input_shape).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    return flops


def calculate_metrics(y_true, y_pred):

    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    return f1, precision, recall

def setup_experiment(ddc_type, exp_id, ddc_params=None, lr=None, batch_size=128):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


    param_str = ""
    if ddc_params:

        ip = ddc_params.get('I_P', 0)
        gp = ddc_params.get('GD_P', 0)
        param_str += f"_IP{ip}_GP{gp}"

    if lr is not None:
        param_str += f"_lr{lr}"
    
    param_str += f"_bs{batch_size}"


    exp_dir = f"experiments/mnist/ddc/exp_{timestamp}_{exp_id}{param_str}"
    os.makedirs(exp_dir, exist_ok=True)

    log_file = f"{exp_dir}/training.log"
    config_file = f"{exp_dir}/config.json"
    model_file = f"{exp_dir}/best_model.pth"
    results_file = f"{exp_dir}/results.json"
    metrics_csv = f"{exp_dir}/training_metrics.csv"

    return {
        'exp_dir': exp_dir,
        'log_file': log_file,
        'config_file': config_file,
        'model_file': model_file,
        'results_file': results_file,
        'metrics_csv': metrics_csv
    }


def setup_dataloaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1000,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, test_loader




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

    end_time = time.time()
    train_time = end_time - start_time
    
    train_acc = 100. * correct / total
    avg_loss = epoch_loss / len(train_loader)
    
    return avg_loss, train_acc, train_time


def evaluate(model, test_loader, criterion, device):
    model.eval()
    start_time = time.time()
    
    correct = 0
    total = 0
    test_loss = 0.0
    
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Collect for metrics
            all_targets.extend(target.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    end_time = time.time()
    test_time = end_time - start_time

    accuracy = 100. * correct / total
    test_loss /= len(test_loader)
    

    f1, precision, recall = calculate_metrics(all_targets, all_preds)
    
    return test_loss, accuracy, test_time, f1, precision, recall


def run_experiment(ddc_type, exp_id, ddc_params=None, num_epochs=200, lr=0.0001, batch_size=128):

    paths = setup_experiment(ddc_type, exp_id, ddc_params, lr, batch_size)
    train_loader, test_loader = setup_dataloaders(batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    config = {
        'ddc_type': ddc_type,
        'exp_id': exp_id,
        'ddc_params': ddc_params if ddc_params else {},
        'num_epochs': num_epochs,
        'learning_rate': lr,
        'batch_size': batch_size,
        'device': str(device),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(paths['config_file'], 'w') as f:
        json.dump(config, f, indent=2)

    # Initialize model
    model = FCNet(ddc_params=ddc_params).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()


    flops = calculate_flops(model, input_shape=(1, 1, 28, 28), device=device)
    print(f"Model FLOPs: {flops}")


    headers = ['epoch', 'test_loss', 'test_acc', 'test_time', 
               'train_loss', 'train_acc', 'train_time', 
               'f1_score', 'precision', 'recall', 'flops']
    
    df_metrics = pd.DataFrame(columns=headers)
    df_metrics.to_csv(paths['metrics_csv'], index=False)

    best_acc = 0.0

    print(f"\n{'=' * 60}")
    print(f"Starting Experiment {exp_id} | Path: {paths['exp_dir']}")
    print(f"LR: {lr} | Batch Size: {batch_size}")
    if ddc_params:
        print(f"DDC Params: {ddc_params}")
    print(f"{'=' * 60}")

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_acc, train_time = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        test_loss, test_acc, test_time, f1, precision, recall = evaluate(model, test_loader, criterion, device)

        # Save Best Checkpoint
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), paths['model_file'])
            # print(f"  --> New Best Accuracy: {best_acc:.2f}%")

        # Save metrics to CSV row
        new_row = {
            'epoch': epoch,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_time': test_time,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_time': train_time,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'flops': flops
        }
        
        # Append to CSV efficiently
        pd.DataFrame([new_row]).to_csv(paths['metrics_csv'], mode='a', header=False, index=False)

        # Print progress
        print(f"Ep {epoch:3d} | Train: {train_loss:.4f}/{train_acc:.2f}% ({train_time:.2f}s) | "
              f"Test: {test_loss:.4f}/{test_acc:.2f}% ({test_time:.2f}s) | "
              f"F1: {f1:.4f} | Best: {best_acc:.2f}%")

    # Generate Graphs
    try:
        data = pd.read_csv(paths['metrics_csv'])
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(data['epoch'], data['train_loss'], label='Train Loss')
        plt.plot(data['epoch'], data['test_loss'], label='Test Loss')
        plt.title('Loss Curve')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(data['epoch'], data['train_acc'], label='Train Acc')
        plt.plot(data['epoch'], data['test_acc'], label='Test Acc')
        plt.title('Accuracy Curve')
        plt.legend()
        
        plt.savefig(os.path.join(paths['exp_dir'], 'curves.png'))
        plt.close()
    except Exception as e:
        print(f"Error plotting curves: {e}")

    # Save final results summary
    results = {
        'best_test_accuracy': best_acc,
        'final_test_accuracy': test_acc,
        'final_f1_score': f1,
        'final_precision': precision,
        'final_recall': recall,
        'flops': flops,
        'total_params': sum(p.numel() for p in model.parameters())
    }

    with open(paths['results_file'], 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nExperiment {exp_id} Finished. Best Accuracy: {best_acc:.2f}%")
    return best_acc


def run_experiments(args):
    """Run multiple experiments"""
    # Create base directory
    os.makedirs("experiments/mnist/ddc", exist_ok=True)

    # Global seed
    global_seed = int(time.time() * 1000000) % 2 ** 31
    random.seed(global_seed)
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(global_seed)

    ddc_params = {
        'I_P': args.I_P,
        'GD_P': args.GD_P,
        'gd_small': args.gd_small
    }

    best_accuracies = []

    for i in range(1, args.num_experiments + 1):
        print(f"\n>>> Running Experiment {i} / {args.num_experiments}")
        acc = run_experiment(
            ddc_type=args.ddc_type, 
            exp_id=i, 
            ddc_params=ddc_params, 
            num_epochs=args.num_epochs, 
            lr=args.lr,
            batch_size=args.batch_size
        )
        best_accuracies.append(acc)

    # Final Summary across all experiments
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED")
    print(f"Average Best Accuracy: {np.mean(best_accuracies):.2f}% +/- {np.std(best_accuracies):.2f}")
    print(f"Max Best Accuracy: {np.max(best_accuracies):.2f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Run DDC experiments on MNIST (Modified)')
    parser.add_argument('--ddc_type', type=str, default='ddc', help='Type of DDC')
    parser.add_argument('--num_experiments', type=int, default=1, help='Number of experiments')
    parser.add_argument('--num_epochs', type=int, default=50, help='Epochs per experiment')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

    # DDC params
    parser.add_argument('--I_P', type=float, default=0.3, help='Initial drop rate')
    parser.add_argument('--GD_P', type=float, default=0.5, help='Gradient drop rate')
    parser.add_argument('--gd_small', type=bool, default=True, help='Drop small gradients')

    args = parser.parse_args()
    
    run_experiments(args)

if __name__ == "__main__":
    main()