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
# 日志配置
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
# 攻击方法定义
# ===============================
def maximize_prediction_set_opsa(model, device, x, y, eps, atkIter=10, lr=0.01, T=1.0, norm='l_inf'):
    """优化后的最大化预测集攻击实现"""
    model.eval()
    batch_size = x.shape[0]

    # 减少重复的扰动副本数量
    num_attempts = 5  # 每次只尝试5个不同的初始扰动

    x_adv = x.unsqueeze(1).repeat(1, num_attempts, 1, 1, 1).detach()
    x_adv = x_adv + (torch.rand_like(x_adv) * 2 * eps - eps)
    x_adv = torch.clamp(x_adv, 0, 1).requires_grad_(True)

    optimizer = optim.Adam([x_adv], lr=lr)

    best_perturbed = x.clone()
    best_margins = torch.zeros(batch_size, device=device) - float('inf')

    for itr in range(atkIter):
        optimizer.zero_grad()

        # 分批处理扰动以减少内存使用
        sub_batch_size = 2
        for j in range(0, num_attempts, sub_batch_size):
            current_x_adv = x_adv[:, j:min(j + sub_batch_size, num_attempts)].reshape(-1, *x.shape[1:])

            logits = model(current_x_adv)
            logits = logits.view(batch_size, -1, logits.shape[-1])

            # 计算损失
            true_logits = torch.gather(logits, 2, y.unsqueeze(1).unsqueeze(2).expand(-1, logits.shape[1], 1))
            margin = logits - true_logits
            sigma = torch.sigmoid(margin / T)
            sum_sigma = sigma.sum(dim=2)

            loss = -sum_sigma.mean()
            loss.backward()

        # 更新和投影
        optimizer.step()

        with torch.no_grad():
            if norm == 'l_inf':
                x_adv.data = torch.clamp(x_adv.data, x.unsqueeze(1) - eps, x.unsqueeze(1) + eps)
            x_adv.data = torch.clamp(x_adv.data, 0, 1)

        # 清除中间变量
        del logits, margin, sigma, sum_sigma
        torch.cuda.empty_cache()

    # 最终评估和选择最佳扰动
    with torch.no_grad():
        for j in range(num_attempts):
            current_x_adv = x_adv[:, j]
            logits = model(current_x_adv)

            true_logits = torch.gather(logits, 1, y.unsqueeze(1)).squeeze(1)
            margin = logits - true_logits.unsqueeze(1)
            sigma = torch.sigmoid(margin / T)
            sum_sigma = sigma.sum(dim=1)

            # 更新最佳结果
            update_mask = sum_sigma > best_margins
            best_margins[update_mask] = sum_sigma[update_mask]
            best_perturbed[update_mask] = current_x_adv[update_mask]

            del logits, margin, sigma, sum_sigma
            torch.cuda.empty_cache()

    return best_perturbed, best_margins


# ===============================
# 模型定义
# ===============================
def get_model(dataset_name, num_classes, input_channels=3):
    """获取模型"""
    if dataset_name == 'MNIST':
        model = models.resnet18(weights=None, num_classes=num_classes)
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    else:
        model = models.resnet34(weights=None, num_classes=num_classes)
        if input_channels != 3:
            model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ===============================
# 数据加载函数
# ===============================
def get_data_loaders(dataset_name, batch_size=32, calibrate_indices_path=None, test_indices_path=None):
    """优化后的数据加载器"""
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
        logging.info(f'加载了{dataset_name}的校准子集，共有{len(calibrate_subset)}个样本。')

    # 删除 MNIST 的测试集加载
    if test_indices_path and dataset_name != 'MNIST':
        test_indices = np.load(test_indices_path)
        test_subset = Subset(full_test_dataset, test_indices)
        test_loader = DataLoader(test_subset, shuffle=False, **loader_kwargs)
        logging.info(f'加载了{dataset_name}的测试子集，共有{len(test_subset)}个样本。')

    return calibrate_loader, test_loader


# ===============================
# 攻击数据生成和保存函数
# ===============================
def generate_and_save_attacked_data(model, device, loader, eps, atkIter, lr, T, norm):
    """生成攻击数据并返回完整结果"""
    all_perturbed = []
    all_labels = []

    for data, target in tqdm(loader):
        data, target = data.to(device), target.to(device)

        # 执行攻击
        perturbed, _ = maximize_prediction_set_opsa(
            model, device, data, target,
            eps, atkIter, lr, T, norm
        )

        all_perturbed.append(perturbed.cpu())
        all_labels.append(target.cpu())

        # 清理内存
        del perturbed, data, target
        torch.cuda.empty_cache()

    # 合并所有结果
    attacked_data = torch.cat(all_perturbed, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return {
        'data': attacked_data,
        'labels': labels
    }


# ===============================
# 主函数
# ===============================
# ===============================
# 主函数
# ===============================
def main():
    parser = argparse.ArgumentParser(description='优化后的最大化预测集对抗性攻击生成')
    parser.add_argument('--save_dir', type=str, default='E:/adversarial/attack/attacked_data_OPSA')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epsilon', type=float, default=0.03)
    parser.add_argument('--atkIter', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--norm', type=str, default='l_inf', choices=['l_inf', 'l2'])
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # 数据集配置
    datasets_config = [
        # {
        #     'name': 'CIFAR10',
        #     'num_classes': 10,
        #     'input_channels': 3,
        #     'calibrate_indices_path': r'E:/adversarial/data_splits/CIFAR10/split1/calibrate_indices.npy',
        #     'test_indices_path': r'E:/adversarial/data_splits/CIFAR10/split1/test_indices.npy',
        #     'model_path': r'E:\adversarial\saved_models\CIFAR10\scheme1\cifar10_scheme1_resnet34.pth',
        #     'T': 1  # 设置 CIFAR10 的 T
        # },
        {

            'name': 'CIFAR100',
            'num_classes': 100,
            'input_channels': 3,
            'calibrate_indices_path': r'E:/adversarial/data_splits/CIFAR100/split1/calibrate_indices.npy',
            'test_indices_path': r'E:/adversarial/data_splits/CIFAR100/split1/test_indices.npy',
            'model_path': r'E:/adversarial/saved_models\CIFAR100\phase1_and_phase2\cifar100_resnet34_phase1_and_phase2.pth',
            'T': 1  # 设置 CIFAR100 的 T
        }
    ]

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f'使用设备: {device}')

    for dataset in datasets_config:
        dataset_name = dataset['name']
        logging.info(f'\n=== 处理数据集: {dataset_name} ===')

        # 加载数据
        calibrate_loader, test_loader = get_data_loaders(
            dataset_name=dataset_name,
            batch_size=args.batch_size,
            calibrate_indices_path=dataset['calibrate_indices_path'],
            test_indices_path=dataset['test_indices_path']
        )

        if calibrate_loader is None and test_loader is None:
            logging.warning(f'{dataset_name}没有可用的数据加载器。跳过。')
            continue

        # 加载模型
        model = get_model(dataset_name, dataset['num_classes'], dataset['input_channels']).to(device)
        if os.path.exists(dataset['model_path']):
            # 设置 weights_only=True 以增强安全性
            model.load_state_dict(torch.load(dataset['model_path'], map_location=device, weights_only=True))
            logging.info(f'已加载{dataset_name}的模型')
        else:
            logging.warning(f'模型文件未找到: {dataset["model_path"]}')
        model.eval()

        # 根据数据集设置 T
        T = dataset.get('T', 1.0)  # 或者使用 T = dataset['T']

        # 处理校准集
        if calibrate_loader is not None:
            logging.info(f'\n处理{dataset_name}的校准集...')
            # 修改保存路径
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

        # 处理测试集（仅针对 CIFAR10 和 CIFAR100）
        if test_loader is not None:
            logging.info(f'\n处理{dataset_name}的测试集...')
            # 修改保存路径
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

        # 清理内存
        torch.cuda.empty_cache()
        gc.collect()

    logging.info('\n=== 攻击生成完成 ===')


if __name__ == '__main__':
    main()