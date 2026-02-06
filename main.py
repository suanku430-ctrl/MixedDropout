import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils import ExpLogger, calculate_flops, evaluate_metrics
from models.mnist import MnistNet
from models.cifar10 import Cifar10Net
from models.cifar100 import Cifar100Net
from models.svhn import SVHNNet
from models.norb import NORBNet, NORBDataset


def get_data(dataset_name, data_dir, batch_size):
    if dataset_name == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(data_dir, train=False, transform=transform)

    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(data_dir, train=False, transform=transform)

    elif dataset_name == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        train_set = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR100(data_dir, train=False, transform=transform)

    elif dataset_name == 'svhn':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = datasets.SVHN(data_dir, split='train', download=True, transform=transform)
        test_set = datasets.SVHN(data_dir, split='test', download=True, transform=transform)

    elif dataset_name == 'norb':

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:2]),  # Keep 2 channels
            
            transforms.Normalize((0.5, 0.5), (0.5, 0.5))
        ])
        train_set = NORBDataset(data_dir, train=True, transform=transform)
        test_set = NORBDataset(data_dir, train=False, transform=transform)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


    if dataset_name == 'mnist':
        input_size = (1, 28, 28)
    elif dataset_name == 'norb':
        input_size = (2, 96, 96)
    else:
        input_size = (3, 32, 32)

    return train_loader, test_loader, input_size


def get_optimizer(model, dataset_name, lr):

    if dataset_name == 'mnist' or dataset_name == 'cifar100':
        return optim.RMSprop(model.parameters(), lr=lr)
    elif dataset_name in ['cifar10', 'svhn', 'norb']:
        return optim.Adam(model.parameters(), lr=lr)
    else:
        return optim.Adam(model.parameters(), lr=lr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'cifar10', 'cifar100', 'svhn', 'norb'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dropout', type=str, default='baseline',
                        choices=['baseline', 'vanilla', 'rad', 'mix', 'dropconnect', 'adaptive'])
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--exp_id', type=int, default=1)
    # Dropout Params
    parser.add_argument('--p', type=float, default=0.5)
    parser.add_argument('--p_init', type=float, default=0.5)
    parser.add_argument('--rho_init', type=float, default=0.5)
    parser.add_argument('--use_rho_init', action='store_true')

    args = parser.parse_args()


    dropout_params = {}
    if args.dropout in ['vanilla', 'dropconnect']:
        dropout_params['p'] = args.p
    elif args.dropout == 'mix':
        dropout_params = {'p_init': args.p_init, 'rho_init': args.rho_init, 'use_rho_init': args.use_rho_init}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device} | Dataset: {args.dataset} | Dropout: {args.dropout}")


    train_loader, test_loader, input_size = get_data(args.dataset, args.data_dir, args.batch_size)

    model_map = {
        'mnist': MnistNet,
        'cifar10': Cifar10Net,
        'cifar100': Cifar100Net,
        'svhn': SVHNNet,
        'norb': NORBNet
    }
    model = model_map[args.dataset](dropout_type=args.dropout, dropout_params=dropout_params).to(device)


    flops, params = calculate_flops(model, input_size, device)
    print(f"Model FLOPs: {flops:,}")
    print(f"Model params: {params:,}")


    optimizer = get_optimizer(model, args.dataset, args.lr)
    criterion = nn.CrossEntropyLoss()


    logger = ExpLogger(args.dataset, args.dropout, args.exp_id, dropout_params)

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'f1': []}
    best_acc = 0.0


    for epoch in range(1, args.epochs + 1):

        model.train()
        train_loss = 0
        correct = 0
        total = 0
        start_time = time.time()

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        train_time = time.time() - start_time
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = 100. * correct / total


        metrics = evaluate_metrics(model, test_loader, criterion, device)


        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.2f}% | "
              f"Test Loss: {metrics['loss']:.4f} Acc: {metrics['acc']:.2f}% F1: {metrics['f1']:.4f} | "
              f"Time: {train_time + metrics['time']:.1f}s")

        epoch_data = {
            'epoch': epoch,
            'train_loss': avg_train_loss, 'train_acc': avg_train_acc, 'train_time': train_time,
            'test_loss': metrics['loss'], 'test_acc': metrics['acc'], 'test_time': metrics['time'],
            'f1': metrics['f1'], 'precision': metrics['precision'], 'recall': metrics['recall']
        }
        logger.log_epoch(epoch_data)


        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['test_loss'].append(metrics['loss'])
        history['test_acc'].append(metrics['acc'])
        history['f1'].append(metrics['f1'])


        if metrics['acc'] > best_acc:
            best_acc = metrics['acc']
            torch.save(model.state_dict(), logger.model_file)


    logger.save_results({
        'best_acc': best_acc,
        'flops': flops,
        'final_f1': history['f1'][-1],
        'params': dropout_params
    })
    logger.plot_curves(history)
    print(f"Experiment Finished. Best Acc: {best_acc:.2f}%. Results saved to {logger.exp_dir}")


if __name__ == '__main__':
    main()