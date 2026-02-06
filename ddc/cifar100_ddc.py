import torch
import torch.nn as nn
import torch.optim as optim
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
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score
from thop import profile

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# -------------------------------------------------------------------------
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
# -------------------------------------------------------------------------
class CIFAR100Net(nn.Module):
    def __init__(self, drop_model=None):
        super(CIFAR100Net, self).__init__()
        # 卷积部分
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 16x16
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 8x8
        )

        # 全连接部分
        self.fc1 = nn.Linear(64 * 8 * 8, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 100)  # CIFAR-100 有 100 个类

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

# -------------------------------------------------------------------------
def setup_dataloaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    train_set = datasets.CIFAR100('./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR100('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def calculate_flops(model, input_size=(1, 3, 32, 32)):
    dummy_input = torch.randn(input_size).to(DEVICE)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    return flops



def calculate_metrics(y_true, y_pred):

    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    return f1, precision, recall



# -------------------------------------------------------------------------
def train_model(model, train_loader, test_loader, optimizer, criterion, args, exp_id, csv_writer, save_dir, flops):
    best_acc = 0.0
    checkpoint_path = os.path.join(save_dir, 'best_model.pth')

    for epoch in range(1, args.epochs + 1):

        model.train()
        if model.drop_model:
            model.drop_model._reset_time()

        train_loss = 0.0
        correct = 0
        total = 0

        start_train = time.time()
        for data, target in tqdm(train_loader, desc=f"Exp {exp_id} | Epoch {epoch}/{args.epochs}", leave=False):
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

        train_time = time.time() - start_train
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total

        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_preds = []
        all_targets = []

        start_test = time.time()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()

                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
                test_total += target.size(0)


                all_preds.extend(pred.view(-1).cpu().numpy())
                all_targets.extend(target.view(-1).cpu().numpy())

        test_time = time.time() - start_test
        avg_test_loss = test_loss / len(test_loader)
        test_acc = 100. * test_correct / test_total

        f1, prec, rec = calculate_metrics(all_targets, all_preds)


        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'args': vars(args)
            }, checkpoint_path)


        csv_writer.writerow([
            exp_id, epoch,
            f"{avg_test_loss:.6f}", f"{test_acc:.4f}", f"{test_time:.4f}",
            f"{avg_train_loss:.6f}", f"{train_acc:.4f}", f"{train_time:.4f}",
            f"{f1:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{flops:.0f}", f"{best_acc:.4f}"
        ])

        print(f"Epoch {epoch}: Train Acc {train_acc:.2f}% ({train_time:.1f}s) | "
              f"Test Acc {test_acc:.2f}% ({test_time:.1f}s) | "
              f"Best: {best_acc:.2f}% | F1: {f1:.4f}")

    return best_acc


def run_experiment(args):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_name = f"cifar100_lr{args.lr}_bs{args.batch_size}_init{args.gd_init_droprate}_drop{args.gd_droprate}_{timestamp}"


    base_dir = "experiments/cifar100/ddc"
    save_dir = os.path.join(base_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Starting Run. Results will be saved to: {save_dir}")


    with open(os.path.join(save_dir, 'parameters.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    csv_filename = os.path.join(save_dir, 'results_detailed.csv')
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        csv_writer.writerow([
            'experiment_id', 'epoch',
            'test_loss', 'test_acc', 'test_time',
            'train_loss', 'train_acc', 'train_time',
            'f1_score', 'precision', 'recall', 'flops', 'best_test_acc'
        ])


        train_loader, test_loader = setup_dataloaders(args.batch_size)
        best_accs = []

        for i in range(args.num_experiments):
            print(f"\n>>> Experiment {i + 1}/{args.num_experiments}")

            drop_small_gd = GradWeightDrop(
                device=DEVICE, I_P=args.gd_init_droprate, GD_P=args.gd_droprate, gd_small=True
            )
            model = CIFAR100Net(drop_model=drop_small_gd).to(DEVICE)

            flops_val = 0
            if i == 0:
                flops_val = calculate_flops(model)
                print(f"Model FLOPs: {flops_val:,.0f}")

            optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
            criterion = nn.CrossEntropyLoss()

            # 训练
            acc = train_model(
                model, train_loader, test_loader, optimizer, criterion,
                args, i + 1, csv_writer, save_dir, flops_val
            )
            best_accs.append(acc)
            csvfile.flush()


    avg_acc = np.mean(best_accs)
    std_acc = np.std(best_accs)

    summary_content = (
        f"Experiment Summary\n==================\n"
        f"Timestamp: {timestamp}\n"
        f"Dataset: CIFAR-100\n"
        f"Parameters: {vars(args)}\n\n"
        f"Results per run: {best_accs}\n"
        f"Average Best Accuracy: {avg_acc:.2f}%\n"
        f"Std Dev: {std_acc:.4f}\n"
    )

    with open(os.path.join(save_dir, 'summary.txt'), 'w') as f:
        f.write(summary_content)

    print(f"\nAll Done! Saved to: {save_dir}")
    print(f"Avg Best Acc: {avg_acc:.2f}%")



def main():
    parser = argparse.ArgumentParser(description='DropSmallGd on CIFAR-100 (Modified)')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--batch_size', type=int, default=128, help='批大小')
    parser.add_argument('--gd_init_droprate', type=float, default=0.3, help='初始丢弃率')
    parser.add_argument('--gd_droprate', type=float, default=0.5, help='梯度丢弃率')
    parser.add_argument('--num_experiments', type=int, default=1, help='实验次数')
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()