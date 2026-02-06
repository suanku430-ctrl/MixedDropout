import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from thop import profile
import time
from datetime import datetime

@dataclass
class OptimizerConfig:
    max_iterations: int = 100
    learning_rate: float = 0.10
    decay_rate: float = 0.9
    stopping_error: float = 0.01


class AdaptiveInformationDropout(nn.Module):
    def __init__(
            self,
            initial_p: float,
            calc_information_loss: Callable,
            information_loss_threshold: float = 0.10,
            optimizer_config: Optional[OptimizerConfig] = None,
            name: str = "",
            verbose: int = 0,
            **kwargs
    ):
        super().__init__()
        if not 0 <= initial_p <= 1:
            raise ValueError("Initial dropout probability must be between 0 and 1")
        if not callable(calc_information_loss):
            raise ValueError("calc_information_loss must be callable")

        self.p = torch.nn.Parameter(torch.tensor(initial_p), requires_grad=False)
        self.calc_information_loss = calc_information_loss
        self.information_loss_threshold = information_loss_threshold
        self.optimizer_config = optimizer_config or OptimizerConfig()
        self.name = name
        self.verbose = verbose
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
def calc_information_loss(pre_dropout: torch.Tensor, post_dropout: torch.Tensor, properties: dict) -> torch.Tensor:
    cov_pre = pre_dropout.std() / (torch.abs(pre_dropout).mean() + 1e-8)
    cov_post = post_dropout.std() / (torch.abs(post_dropout).mean() + 1e-8)
    return torch.abs(cov_pre - cov_post)


# =====================================================
class RateInMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout2 = AdaptiveInformationDropout(
            initial_p=0.5,
            calc_information_loss=calc_information_loss,
            information_loss_threshold=0.10
        )
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x



# =====================================================
def calculate_flops_params(model, input_size=(1, 1, 28, 28)):
    try:
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_size).to(device)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        return flops, params
    except Exception as e:
        print(f"Warning: FLOPs calculation failed: {e}")
        return 0, 0


def plot_history(history, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)


    plots = [
        ('loss', 'train_loss', 'test_loss', 'Loss'),
        ('acc', 'train_acc', 'test_acc', 'Accuracy (%)'),
        ('time', 'train_time', 'test_time', 'Time (s)')  # 新增时间图表
    ]

    for name, train_key, test_key, ylabel in plots:
        plt.figure()
        if train_key in history:
            plt.plot(epochs, history[train_key], label=f'Train {name.capitalize()}')
        if test_key in history:
            plt.plot(epochs, history[test_key], label=f'Test {name.capitalize()}')
        plt.title(f'{ylabel} Curve')
        plt.xlabel('Epochs')
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'{name}_curve.png'))
        plt.close()

    # F1 score separately
    plt.figure()
    plt.plot(epochs, history['test_f1'], label='Test F1 Score')
    plt.title('F1 Score Curve')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'f1_curve.png'))
    plt.close()



# =====================================================
def train(model, device, train_loader, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    start_time = time.time()  # [Start Timer]

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    end_time = time.time()  # [End Timer]
    epoch_duration = end_time - start_time

    train_loss = total_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc, epoch_duration


def test(model, device, test_loader):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    start_time = time.time()  # [Start Timer]

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            total_loss += loss.item()

            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    end_time = time.time()  # [End Timer]
    epoch_duration = end_time - start_time

    test_loss = total_loss / len(test_loader)

    test_acc = 100. * np.sum(np.array(all_preds) == np.array(all_targets)) / len(all_targets)


    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    return test_loss, test_acc, precision, recall, f1, epoch_duration


# =====================================================
# Main
# =====================================================
def main():

    parser = argparse.ArgumentParser(description='Rate-In Dropout MNIST Experiment')
    parser.add_argument('--exp_id', type=str, default='1', help='Experiment ID')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')  # 修改这个值可以改变结果
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    np.random.seed(args.seed)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = f"experiments/mnist/Rate_In/exp_{args.exp_id}_{timestamp}"
    os.makedirs(exp_dir, exist_ok=True)
    log_file = os.path.join(exp_dir, "training_log.txt")
    model_save_path = os.path.join(exp_dir, "best_model.pth")

    print(f"Starting Experiment {args.exp_id} on {device}")
    print(f"Logs and models will be saved to: {exp_dir}")


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(args.data_dir, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False,pin_memory=True)

    model = RateInMLP().to(device)
    flops, params = calculate_flops_params(model)
    print(f"Model Params: {params / 1e6:.2f}M | FLOPs: {flops / 1e6:.2f}M")

    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)


    best_test_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [], 'train_time': [],
        'test_loss': [], 'test_acc': [], 'test_f1': [], 'test_time': []
    }

    with open(log_file, "w") as f:
        f.write(f"Config: {vars(args)}\n")
        f.write(f"Params: {params}, FLOPs: {flops}\n")
        f.write("-" * 120 + "\n")

        f.write(
            "Epoch\tTrainLoss\tTrainAcc\tTestLoss\tTestAcc\tPrecision\tRecall\tF1Score\tTrTime(s)\tTeTime(s)\tBestAcc\n")

    total_start_time = time.time()

    for epoch in range(1, args.epochs + 1):

        train_loss, train_acc, train_duration = train(model, device, train_loader, optimizer)
        test_loss, test_acc, precision, recall, f1, test_duration = test(model, device, test_loader)


        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_time'].append(train_duration)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['test_f1'].append(f1)
        history['test_time'].append(test_duration)


        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), model_save_path)
            saved_msg = "[Saved]"
        else:
            saved_msg = ""


        log_line = (f"{epoch}\t{train_loss:.4f}\t{train_acc:.2f}%\t"
                    f"{test_loss:.4f}\t{test_acc:.2f}%\t"
                    f"{precision:.4f}\t{recall:.4f}\t{f1:.4f}\t"
                    f"{train_duration:.2f}\t{test_duration:.2f}\t"  
                    f"{best_test_acc:.2f}%\t{saved_msg}")

        print(log_line)

        with open(log_file, "a") as f:
            f.write(log_line + "\n")

    total_end_time = time.time()


    print("-" * 50)
    print(f"Experiment {args.exp_id} Finished.")
    print(f"Total Duration: {(total_end_time - total_start_time) / 60:.2f} minutes")
    print(f"Best Test Accuracy: {best_test_acc:.2f}%")
    print(f"Total Params: {params}")
    print(f"Total FLOPs: {flops}")
    print(f"Plots saved to {exp_dir}")

    plot_history(history, exp_dir)


if __name__ == '__main__':
    main()