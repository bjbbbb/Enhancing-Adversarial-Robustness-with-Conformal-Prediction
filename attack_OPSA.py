import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import argparse
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import logging
import sys
import gc

# ===============================
# Log Configuration
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('maximize_set_attack.log'),
        logging.StreamHandler(sys.stdout)
    ]
)


# ===============================
# Definition of Attack Methods
# ===============================
def maximize_prediction_set_opsa(model, device, x, y, eps, atkIter=10, lr=0.01, T=1.0, norm='l_inf'):
    """Implementation of optimized maximum prediction set attack"""
    model.eval()
    batch_size = x.shape[0]
 
    num_attempts = 5  

    x_adv = x.unsqueeze(1).repeat(1, num_attempts, 1, 1, 1).detach()
    x_adv = x_adv + (torch.rand_like(x_adv) * 2 * eps - eps)
    x_adv = torch.clamp(x_adv, 0, 1).requires_grad_(True)

    optimizer = optim.Adam([x_adv], lr=lr)

    best_perturbed = x.clone()
    best_margins = torch.zeros(batch_size, device=device) - float('inf')

    for itr in range(atkIter):
        optimizer.zero_grad()

        sub_batch_size = 2
        for j in range(0, num_attempts, sub_batch_size):
            current_x_adv = x_adv[:, j:min(j + sub_batch_size, num_attempts)].reshape(-1, *x.shape[1:])

            logits = model(current_x_adv)
            logits = logits.view(batch_size, -1, logits.shape[-1])

            true_logits = torch.gather(logits, 2, y.unsqueeze(1).unsqueeze(2).expand(-1, logits.shape[1], 1))
            margin = logits - true_logits
            sigma = torch.sigmoid(margin / T)
            sum_sigma = sigma.sum(dim=2)

            loss = -sum_sigma.mean()
            loss.backward()

        optimizer.step()

        with torch.no_grad():
            if norm == 'l_inf':
                x_adv.data = torch.clamp(x_adv.data, x.unsqueeze(1) - eps, x.unsqueeze(1) + eps)
            x_adv.data = torch.clamp(x_adv.data, 0, 1)

        del logits, margin, sigma, sum_sigma
        torch.cuda.empty_cache()

    with torch.no_grad():
        for j in range(num_attempts):
            current_x_adv = x_adv[:, j]
            logits = model(current_x_adv)

            true_logits = torch.gather(logits, 1, y.unsqueeze(1)).squeeze(1)
            margin = logits - true_logits.unsqueeze(1)
            sigma = torch.sigmoid(margin / T)
            sum_sigma = sigma.sum(dim=1)

            update_mask = sum_sigma > best_margins
            best_margins[update_mask] = sum_sigma[update_mask]
            best_perturbed[update_mask] = current_x_adv[update_mask]

            del logits, margin, sigma, sum_sigma
            torch.cuda.empty_cache()

    return best_perturbed, best_margins


# ===============================
# Model Definition
# ===============================
def get_model(dataset_name, num_classes, input_channels=3):
    if dataset_name == 'CIFAR100':
        model = models.resnet34(pretrained=False, num_classes=num_classes)
        model_type = 'resnet34'
    elif dataset_name == 'CIFAR10':
        model = models.resnet34(pretrained=False, num_classes=num_classes)
        model_type = 'resnet34'
    elif dataset_name == 'ImageNetMini':
        model = models.resnet50(pretrained=False, num_classes=num_classes)
        model_type = 'resnet50'
    if input_channels != 3 and dataset_name in ['CIFAR10', 'CIFAR100']:
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    return model, model_type


def get_data_loaders(dataset_name, batch_size=32, calibrate_indices_path=None, test_indices_path=None):
    if dataset_name == 'MNIST':
        data_class = datasets.MNIST
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        input_channels = 1
    else:
        data_class = datasets.CIFAR10 if dataset_name == 'CIFAR10' else datasets.CIFAR100
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465) if dataset_name == 'CIFAR10' else (0.5071, 0.4867, 0.4408),
                (0.2023, 0.1994, 0.2010) if dataset_name == 'CIFAR10' else (0.2675, 0.2565, 0.2761)
            )
        ])
        input_channels = 3

    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': 2,
        'pin_memory': True,
        'persistent_workers': True
    }

    full_test_dataset = data_class('data', train=False, download=True, transform=transform)

    calibrate_loader = None
    test_loader = None

    if calibrate_indices_path:
        calibrate_indices = np.load(calibrate_indices_path)
        calibrate_subset = Subset(full_test_dataset, calibrate_indices)
        calibrate_loader = DataLoader(calibrate_subset, shuffle=False, **loader_kwargs)

    if test_indices_path and dataset_name != 'MNIST':
        test_indices = np.load(test_indices_path)
        test_subset = Subset(full_test_dataset, test_indices)
        test_loader = DataLoader(test_subset, shuffle=False, **loader_kwargs)

    return calibrate_loader, test_loader

def generate_and_save_attacked_data(model, device, loader, eps, atkIter, lr, T, norm):
    all_perturbed = []
    all_labels = []

    for data, target in tqdm(loader):
        data, target = data.to(device), target.to(device)

        perturbed, _ = maximize_prediction_set_opsa(
            model, device, data, target,
            eps, atkIter, lr, T, norm
        )

        all_perturbed.append(perturbed.cpu())
        all_labels.append(target.cpu())

        del perturbed, data, target
        torch.cuda.empty_cache()

    attacked_data = torch.cat(all_perturbed, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return {
        'data': attacked_data,
        'labels': labels
    }

def main():
    parser = argparse.ArgumentParser(description='OPSA')
    parser.add_argument('--save_dir', type=str, default='E:/adversarial/attack/attacked_data_OPSA')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epsilon', type=float, default=0.03)
    parser.add_argument('--atkIter', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--norm', type=str, default='l_inf', choices=['l_inf', 'l2'])
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()


    datasets_config = [
        # {
        #     'name': 'CIFAR10',
        #     'num_classes': 10,
        #     'input_channels': 3,
        #     'calibrate_indices_path': r'/calibrate_indices.npy',
        #     'test_indices_path': r'/test_indices.npy',
        #     'model_path': r'\cifar10_scheme1_resnet34.pth',
        #     'T': 1  
        # },
    ]

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    for dataset in datasets_config:
        dataset_name = dataset['name']

        calibrate_loader, test_loader = get_data_loaders(
            dataset_name=dataset_name,
            batch_size=args.batch_size,
            calibrate_indices_path=dataset['calibrate_indices_path'],
            test_indices_path=dataset['test_indices_path']
        )

        model = get_model(dataset_name, dataset['num_classes'], dataset['input_channels']).to(device)
        if os.path.exists(dataset['model_path']):
            model.load_state_dict(torch.load(dataset['model_path'], map_location=device, weights_only=True))
        model.eval()

        T = dataset.get('T', 1.0)  # 或者使用 T = dataset['T']

        if calibrate_loader is not None:
            calibrate_save_dir = os.path.join(
                args.save_dir,
                dataset_name,
                'calibrate',
                'maximize_set'
            )
            os.makedirs(calibrate_save_dir, exist_ok=True)

            calibrate_results = generate_and_save_attacked_data(
                model, device, calibrate_loader,
                args.epsilon, args.atkIter,
                args.lr, T, args.norm
            )
            save_path = os.path.join(calibrate_save_dir,
                                     f'eps_{args.epsilon}_iter_{args.atkIter}_maximize_set_{args.lr}.pth')
            torch.save(calibrate_results, save_path)

        if test_loader is not None:
            test_save_dir = os.path.join(
                args.save_dir,
                dataset_name,
                'test',
                'maximize_set'
            )
            os.makedirs(test_save_dir, exist_ok=True)

            test_results = generate_and_save_attacked_data(
                model, device, test_loader,
                args.epsilon, args.atkIter,
                args.lr, T, args.norm
            )
            save_path = os.path.join(test_save_dir,
                                     f'eps_{args.epsilon}_iter_{args.atkIter}_maximize_set_{args.lr}.pth')
            torch.save(test_results, save_path)

        torch.cuda.empty_cache()
        gc.collect()



if __name__ == '__main__':
    main()
