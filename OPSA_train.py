import os
import sys
import argparse
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# ===============================
# 日志配置
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# ===============================
# 攻击方法定义
# ===============================
def maximize_prediction_set_opsa(model, device, x, y, eps, atkIter=10, lr=0.1, T=1.0, norm='l_inf'):
    """优化后的最大化预测集攻击实现"""
    model.eval()
    batch_size = x.shape[0]

    num_attempts = 2  # 每次尝试2个不同的初始扰动

    # 扩展批量以包含尝试次数
    x_adv = x.unsqueeze(1).repeat(1, num_attempts, 1, 1, 1).detach()
    # 初始化扰动：在原始图像附近随机扰动
    x_adv = x_adv + (torch.rand_like(x_adv) * 2 * eps - eps)
    # 确保扰动后的图像在 [0,1] 范围内
    x_adv = torch.clamp(x_adv, 0, 1).requires_grad_(True)

    optimizer = optim.Adam([x_adv], lr=lr)

    best_perturbed = x.clone()
    best_margins = torch.full((batch_size,), float('-inf'), device=device)

    for itr in range(atkIter):
        optimizer.zero_grad()

        # 分批处理扰动以减少内存使用
        sub_batch_size = 2
        for j in range(0, num_attempts, sub_batch_size):
            current_x_adv = x_adv[:, j:min(j + sub_batch_size, num_attempts)].reshape(-1, *x.shape[1:])

            logits = model(current_x_adv)
            logits = torch.clamp(logits, min=-50, max=50)  # 限制logits范围
            logits = logits.view(batch_size, -1, logits.shape[-1])

            # 获取真实类别的logits
            true_logits = torch.gather(logits, 2, y.view(batch_size, 1, 1).expand(-1, logits.shape[1], 1))
            margin = logits - true_logits

            # 计算损失
            sigma = torch.sigmoid(margin / T)
            sum_sigma = sigma.sum(dim=2)

            loss = -sum_sigma.mean()
            loss.backward()

        # 更新
        optimizer.step()

        # 投影到允许的扰动范围内
        with torch.no_grad():
            if norm == 'l_inf':
                delta = torch.clamp(x_adv.data - x.unsqueeze(1), min=-eps, max=eps)
                x_adv.data = x.unsqueeze(1) + delta
            x_adv.data = torch.clamp(x_adv.data, 0, 1)

        torch.cuda.empty_cache()

    # 最终评估和选择最佳扰动
    with torch.no_grad():
        for j in range(num_attempts):
            current_x_adv = x_adv[:, j]

            logits = model(current_x_adv)
            logits = torch.clamp(logits, min=-1e3, max=1e3)

            true_logits = torch.gather(logits, 1, y.unsqueeze(1)).squeeze(1)
            margin = logits - true_logits.unsqueeze(1)

            sigma = torch.sigmoid(margin / T)
            sum_sigma = sigma.sum(dim=1)

            # 更新最佳结果
            update_mask = sum_sigma > best_margins
            best_margins[update_mask] = sum_sigma[update_mask]
            best_perturbed[update_mask] = current_x_adv[update_mask]

            torch.cuda.empty_cache()

    return best_perturbed, best_margins

# ===============================
# 模型定义
# ===============================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, input_channels=3):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(input_channels, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, lin=0, lout=5):
        out = x
        if lin < 1 and lout > -1:
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)
        if lin < 2 and lout > 0:
            out = self.layer1(out)
        if lin < 3 and lout > 1:
            out = self.layer2(out)
        if lin < 4 and lout > 2:
            out = self.layer3(out)
        if lin < 5 and lout > 3:
            out = self.layer4(out)
        if lout > 4:
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        return out


def PreActResNet18(num_classes=10, input_channels=3):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes, input_channels=input_channels)


def get_model(dataset_name, num_classes, input_channels=3):
    """
    根据数据集名称返回适当的模型。
    对于 CIFAR10 数据集，使用 ResNet34。
    对于 ImageNetMini 数据集，使用 ResNet50。
    对于 CIFAR100 数据集，使用 PreAct ResNet。

    参数:
        dataset_name (str): 数据集名称，支持 'CIFAR10', 'CIFAR100', 'ImageNetMini'。
        num_classes (int): 类别数量。
        input_channels (int): 输入通道数（对于RGB图像为3）。

    返回:
        model (nn.Module): 配置好的模型。
        model_type (str): 模型类型，如 'resnet34', 'resnet50' 或 'preact_resnet'。
    """
    if dataset_name == 'CIFAR100':
        model = models.resnet34(pretrained=False, num_classes=num_classes)
        model_type = 'resnet34'
    elif dataset_name == 'CIFAR10':
        model = models.resnet34(pretrained=False, num_classes=num_classes)
        model_type = 'resnet34'
    elif dataset_name == 'ImageNetMini':
        # 使用自定义的 PreActResNet18
        model = models.resnet50(pretrained=False, num_classes=num_classes)
        model_type = 'resnet50'
    else:
        model = PreActResNet18(num_classes=num_classes, input_channels=input_channels)
        model_type = 'preact_resnet18'

    if input_channels != 3 and dataset_name in ['CIFAR10', 'CIFAR100']:
        # 修改标准 ResNet 的第一层
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    return model, model_type

# ===============================
# 模型权重初始化（从头开始）
# ===============================
def initialize_model(model, device):
    """
    初始化模型权重并移动到指定设备。

    参数:
        model (nn.Module): 要初始化的模型。
        device (torch.device): 设备。

    返回:
        model (nn.Module): 初始化后的模型。
    """
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    model.apply(init_weights)
    model.to(device)
    return model

# ===============================
# 加载数据划分索引
# ===============================
def load_split_indices(dataset_name, split='split1', root_dir='/home/kaiyuan/PycharmProjects/icml_adversarial/data_splits'):
    """
    加载指定划分的数据索引。

    参数:
        dataset_name (str): 数据集名称。
        split (str): 划分名称，如 'split1', 'split2'。
        root_dir (str): 划分索引文件的根目录。

    返回:
        np.ndarray: 索引数组。
    """
    if 'split1' in split or 'scheme1' in split:
        split_num = '1'
    elif 'split2' in split or 'scheme2' in split:
        split_num = '2'
    else:
        raise ValueError("split 参数必须包含 'split1' 或 'split2'")

    # 如果是 ImageNetMini，则更改文件夹名称
    if dataset_name.lower() == 'imagenetmini':
        folder_name = 'mini-imagenet'
    else:
        folder_name = dataset_name

    split_path = os.path.join(root_dir, folder_name, f'split{split_num}', 'train_adv_indices.npy')
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"{split_path} 不存在。请先运行数据划分脚本.")
    indices = np.load(split_path)
    return indices

# ===============================
# 数据加载器定义
# ===============================
def get_dataloader(dataset_name, split, batch_size=128, root_dir='/home/kaiyuan/PycharmProjects/icml_adversarial/data_splits'):
    """
    获取数据加载器，仅处理单一数据划分。

    参数:
        dataset_name (str): 数据集名称，支持 'CIFAR10', 'CIFAR100', 'ImageNetMini'。
        split (str): 划分名称，包含 'split1' 或 'split2'。
        batch_size (int): 批大小。
        root_dir (str): 划分文件的根目录。

    返回:
        DataLoader: 数据加载器。
        mean (torch.Tensor): 数据集均值（形状为 (1, 3, 1, 1)）。
        std (torch.Tensor): 数据集标准差（形状为 (1, 3, 1, 1)）。
    """
    if dataset_name == 'CIFAR10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        dataset = datasets.CIFAR10('/home/kaiyuan/PycharmProjects/icml_adversarial/data', train=True, download=True, transform=transform)
    elif dataset_name == 'CIFAR100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        dataset = datasets.CIFAR100('/home/kaiyuan/PycharmProjects/icml_adversarial/data', train=True, download=True, transform=transform)
    elif dataset_name.lower() == 'imagenetmini':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        # 注意：请确保此处路径与实际数据存放路径一致
        data_dir = os.path.join('/home/kaiyuan/PycharmProjects/icml_adversarial/data', 'mini-imagenet', 'train')
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    else:
        raise ValueError("不支持的数据集。请选择 'CIFAR10', 'CIFAR100' 或 'ImageNetMini'。")

    indices = load_split_indices(dataset_name, split=split, root_dir=root_dir)
    subset = Subset(dataset, indices)
    shuffle = True  # 训练时打乱数据
    num_workers = 2 if os.name != 'nt' else 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return loader, torch.tensor(mean).view(1, 3, 1, 1).to(device), torch.tensor(std).view(1, 3, 1, 1).to(device)

# ===============================
# 训练辅助函数
# ===============================
def compute_size_loss(probs, tau, T_train):
    """
    计算大小损失。

    参数:
        probs (torch.Tensor): 预测概率，形状为 (batch_size, num_classes)。
        tau (float): 阈值。
        T_train (float): 温度参数。

    返回:
        torch.Tensor: 大小损失。
    """
    clamped_input = torch.clamp((probs - tau) / T_train, min=-50.0, max=50.0)
    C_pred_sigmoid = torch.sigmoid(clamped_input)
    probs = torch.clamp(probs, min=1e-7, max=1-1e-7)
    size_loss = C_pred_sigmoid.sum(dim=1).mean()
    return size_loss

def compute_accuracy(outputs, labels):
    """
    计算预测准确度。

    参数:
        outputs (torch.Tensor): 模型输出的 logits，形状为 (batch_size, num_classes)。
        labels (torch.Tensor): 真实标签，形状为 (batch_size,)。

    返回:
        float: 准确度（0到1之间）。
    """
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total

# ===============================
# 标准训练函数
# ===============================
def train_standard(model, train_loader, device, num_epochs, optimizer, scheduler=None, scaler=None):
    """
    标准训练循环，使用交叉熵损失。

    参数:
        model (nn.Module): 要训练的模型。
        train_loader (DataLoader): 训练数据加载器。
        device (torch.device): 设备。
        num_epochs (int): 训练轮数。
        optimizer (torch.optim.Optimizer): 优化器。
        scheduler (torch.optim.lr_scheduler, optional): 学习率调度器。
        scaler (torch.cuda.amp.GradScaler, optional): 混合精度梯度缩放器。

    返回:
        model (nn.Module): 训练后的模型。
    """
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        num_batches = 0

        batch_iterator = tqdm(train_loader, desc=f"标准训练 Epoch {epoch}/{num_epochs}", leave=False)
        for batch in batch_iterator:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with autocast(enabled=(scaler is not None)):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            preds = torch.argmax(outputs, dim=1)
            correct = (preds == labels).sum().item()
            total = labels.size(0)

            epoch_correct += correct
            epoch_total += total
            epoch_loss += loss.item()
            num_batches += 1

            if num_batches % 10 == 0:
                current_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0
                batch_iterator.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{current_accuracy:.4f}"
                })

        if num_batches == 0:
            logger.warning("本轮训练未处理任何批次。")
            continue

        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0

        logger.info(f"** 标准训练 Epoch {epoch}: 平均损失={avg_loss:.4f}, 准确度={avg_accuracy:.4f} **")
    return model

# ===============================
# 训练函数（对抗训练）
# ===============================
def save_model(model, save_dir, dataset_name, scheme, model_type, epoch=None):
    """
    保存模型到指定路径。

    参数:
        model (nn.Module): 要保存的模型。
        save_dir (str): 保存目录。
        dataset_name (str): 数据集名称。
        scheme (str): 划分方案，如 'scheme1' 或 'phase1_and_phase2'。
        model_type (str): 模型类型，如 'resnet34' 或 'resnet50'。
        epoch (int, optional): 当前训练轮数。如果提供，将添加到文件名中。
    """
    os.makedirs(save_dir, exist_ok=True)

    if epoch is not None:
        model_save_path = os.path.join(save_dir, f"{dataset_name.lower()}_{scheme}_{model_type}_epoch{epoch}.pth")
    else:
        model_save_path = os.path.join(save_dir, f"{dataset_name.lower()}_{scheme}_{model_type}.pth")

    torch.save(model.state_dict(), model_save_path)
    logger.info(f"模型已保存到 {model_save_path}")


def train_conftr(model, conformal_train_loader, alpha, device, num_epochs, T_train,
                 optimizer, scheduler=None,
                 lambda_weight=1.0,
                 scaler=None,
                 atk_epsilon=0.03,
                 atk_atkIter=10,
                 atk_lr=0.1,
                 atk_T=1.0,
                 atk_norm='l_inf',
                 dataset_name='CIFAR10',
                 save_dir=None,
                 model_type=None,
                 save_interval=10):
    """
    对抗训练循环，使用 conformal 机制。

    参数:
        model (nn.Module): 要训练的模型。
        conformal_train_loader (DataLoader): 校准训练数据加载器。
        alpha (float): 置信水平参数。
        device (torch.device): 设备。
        num_epochs (int): 训练轮数。
        T_train (float): 训练温度参数。
        optimizer (torch.optim.Optimizer): 优化器。
        scheduler (torch.optim.lr_scheduler, optional): 学习率调度器。
        lambda_weight (float): 大小损失的权重。
        scaler (torch.cuda.amp.GradScaler, optional): 混合精度梯度缩放器。
        atk_epsilon (float): 对抗攻击的扰动大小。
        atk_atkIter (int): 对抗攻击的迭代次数。
        atk_lr (float): 对抗攻击的学习率。
        atk_T (float): 对抗攻击的温度参数。
        atk_norm (str): 对抗攻击的范数类型。
        save_dir (str, optional): 模型保存目录。
        model_type (str, optional): 模型类型。
        save_interval (int, optional): 保存模型的间隔轮数。

    返回:
        model (nn.Module): 训练后的模型。
        best_loss (float): 训练过程中达到的最佳损失值。
    """
    best_loss = float('-inf')
    best_model_state = None

    for epoch in range(1, num_epochs + 1):
        logger.info(f"Epoch {epoch}/{num_epochs}: 启用对抗训练。")
        model.train()

        epoch_size_loss = 0.0
        epoch_class_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        num_batches = 0

        batch_iterator = tqdm(conformal_train_loader, desc=f"对抗训练 Epoch {epoch}/{num_epochs}", leave=False)
        for batch in batch_iterator:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 生成对抗样本
            try:
                inputs_adv, _ = maximize_prediction_set_opsa(
                    model=model,
                    device=device,
                    x=inputs,
                    y=labels,
                    eps=atk_epsilon,
                    atkIter=atk_atkIter,
                    lr=atk_lr,
                    T=atk_T,
                    norm=atk_norm
                )
            except Exception as e:
                logger.error(f"生成对抗样本时发生错误: {e}")
                continue

            combined_inputs = torch.cat([inputs, inputs_adv], dim=0)
            combined_labels = torch.cat([labels, labels], dim=0)

            split_ratio = 0.5
            mask = torch.rand(combined_inputs.size(0), device=device) < split_ratio
            Bcal_inputs, Bcal_labels = combined_inputs[mask], combined_labels[mask]
            Btrain_inputs, Btrain_labels = combined_inputs[~mask], combined_labels[~mask]

            with torch.no_grad():
                if Bcal_inputs.size(0) > 0:
                    outputs_calib = model(Bcal_inputs)
                    probs_calib = nn.functional.softmax(outputs_calib, dim=1)
                    scores_calib = probs_calib.gather(1, Bcal_labels.view(-1, 1)).squeeze().cpu().numpy()
                    if len(scores_calib) == 0:
                        logger.warning("scores_calib 为空，跳过此次批次优化。")
                        continue
                    tau = np.quantile(scores_calib, 1 - alpha * (1 + 1 / len(scores_calib)))
                    tau = np.clip(tau, 1e-5, 1 - 1e-5)
                    logger.debug(f"  重新计算 tau: {tau:.4f}")
                else:
                    logger.warning("校准集为空，跳过此次批次优化。")
                    continue

            with autocast(enabled=(scaler is not None)):
                outputs_train = model(Btrain_inputs)
                probs_train = nn.functional.softmax(outputs_train, dim=1)
                if torch.any(torch.isnan(probs_train)) or torch.any(torch.isinf(probs_train)):
                    logger.error("probs_train 包含 NaN 或 Inf，跳过此次优化。")
                    continue
                probs_train = torch.clamp(probs_train, min=1e-7, max=1 - 1e-7)
                clamped_input = torch.clamp((probs_train - tau) / T_train, min=-50.0, max=50.0)
                sigma = torch.sigmoid(clamped_input)
                if torch.any(torch.isnan(sigma)) or torch.any(torch.isinf(sigma)):
                    logger.error("sigma 包含 NaN 或 Inf，跳过此次优化。")
                    continue
                one_hot_labels = torch.nn.functional.one_hot(Btrain_labels, num_classes=probs_train.size(1)).float().to(
                    device)
                L_class = (sigma * (1 - one_hot_labels)).sum(dim=1).mean() + ((1 - sigma) * one_hot_labels).sum(
                    dim=1).mean()
                L_size = sigma.sum(dim=1).mean()
                loss = L_class + lambda_weight * L_size
                logger.debug(
                    f"  计算损失: L_class={L_class.item():.4f}, L_size={L_size.item():.4f}, Loss={loss.item():.4f}")

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"发现NaN或Inf损失，跳过此次优化。loss={loss.item()}")
                continue

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                skip_step = False
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.any(torch.isnan(param.grad)) or torch.any(torch.isinf(param.grad)):
                            logger.error(f"参数 {name} 的梯度包含 NaN 或 Inf，跳过此次优化。")
                            skip_step = True
                            break
                if skip_step:
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            with torch.no_grad():
                preds = torch.argmax(outputs_train, dim=1)
                correct = (preds == Btrain_labels).sum().item()
                total = Btrain_labels.size(0)
                epoch_correct += correct
                epoch_total += total

            epoch_class_loss += L_class.item()
            epoch_size_loss += L_size.item()
            num_batches += 1

            if num_batches % 10 == 0:
                current_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0
                batch_iterator.set_postfix({
                    'L_class': f"{L_class.item():.4f}",
                    'L_size': f"{L_size.item():.4f}",
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{current_accuracy:.4f}"
                })

        if num_batches == 0:
            logger.warning("本轮训练未处理任何批次。")
            continue

        avg_L_class = epoch_class_loss / num_batches
        avg_L_size = epoch_size_loss / num_batches
        total_avg_loss = avg_L_class + lambda_weight * avg_L_size
        avg_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0

        logger.info(
            f"  平均损失: L_class={avg_L_class:.4f}, L_size={avg_L_size:.4f}, 总损失={total_avg_loss:.4f}, 准确度={avg_accuracy:.4f}")

        if total_avg_loss > best_loss:
            best_loss = total_avg_loss
            best_model_state = model.state_dict().copy()
            logger.info(f"  新的最佳损失: {best_loss:.4f}")

        # 每隔 save_interval 轮保存一次模型
        if save_dir is not None and model_type is not None and epoch % save_interval == 0:
            current_model_state = model.state_dict().copy()
            # 临时加载最佳模型状态用于保存
            if best_model_state is not None:
                model.load_state_dict(best_model_state)

            save_model(
                model=model,
                save_dir=save_dir,
                dataset_name=dataset_name,
                scheme='phase1_and_phase2',
                model_type=model_type,
                epoch=epoch
            )
            logger.info(f"  在第 {epoch} 轮保存了当前最佳模型")

            # 恢复当前模型状态继续训练
            model.load_state_dict(current_model_state)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"恢复最佳模型状态，最佳损失: {best_loss:.4f}")

    return model, best_loss


# ===============================
# 主函数
# ===============================
def main():
    parser = argparse.ArgumentParser(description="基于覆盖保证的对抗训练（Adversarial ConfTr）")

    # 训练参数
    parser.add_argument('--standard_epochs', type=int, default=10, help='标准训练的训练轮数')
    parser.add_argument('--conftr_epochs', type=int, default=40, help='对抗训练的训练轮数')  # 修改为40轮
    parser.add_argument('--save_interval', type=int, default=10, help='保存模型的间隔轮数')
    parser.add_argument('--batch-size', type=int, default=64, help='批大小')
    parser.add_argument('--model_lr', type=float, default=0.005, help='模型优化器学习率')

    # 数据划分参数
    parser.add_argument('--root_dir', type=str, default='/home/kaiyuan/PycharmProjects/icml_adversarial/data_splits',
                        help='数据集划分文件的根目录')

    # 保存参数
    parser.add_argument('--save_dir', type=str, default='/home/kaiyuan/PycharmProjects/icml_adversarial/saved_models',
                        help='训练好的模型保存目录')

    # 训练配置
    parser.add_argument('--alpha', type=float, default=0.10, help='置信水平参数')
    parser.add_argument('--T_train', type=float, default=1, help='训练温度参数')
    parser.add_argument('--lambda_weight', type=float, default=5, help='大小损失的权重')

    # 对抗训练配置
    parser.add_argument('--atk_epsilon', type=float, default=0.03, help='对抗攻击的扰动大小')
    parser.add_argument('--atk_atkIter', type=int, default=10, help='对抗攻击的迭代次数')
    parser.add_argument('--atk_lr', type=float, default=0.1, help='对抗攻击的学习率')
    parser.add_argument('--atk_T', type=float, default=1.0, help='对抗攻击的温度参数')
    parser.add_argument('--atk_norm', type=str, default='l_inf', choices=['l_inf'], help='对抗攻击的范数类型')

    # 设备参数
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='使用的设备')

    # 优化器和调度器
    parser.add_argument('--use_scheduler', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='是否使用学习率调度器')

    # 混合精度训练
    parser.add_argument('--disable_amp', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='是否禁用混合精度训练')

    args = parser.parse_args()

    if args.T_train <= 0:
        raise ValueError("T_train 必须为正数。")
    if args.alpha <= 0 or args.alpha >= 1:
        raise ValueError("alpha 必须在 (0, 1) 区间内。")
    if args.lambda_weight < 0:
        raise ValueError("lambda_weight 必须为非负数。")
    if args.model_lr <= 0:
        raise ValueError("model_lr 必须为正数。")
    if args.atk_epsilon < 0:
        raise ValueError("atk_epsilon 必须为非负数。")
    if args.atk_atkIter <= 0:
        raise ValueError("atk_atkIter 必须为正整数。")
    if args.atk_lr <= 0:
        raise ValueError("atk_lr 必须为正数。")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')

    # 数据集列表，可同时包含 CIFAR100 和 ImageNetMini 实验（根据需要选择）
    datasets_list = ['ImageNetMini']
    model_save_root = args.save_dir

    if not args.disable_amp and device.type == 'cuda':
        scaler = GradScaler()
        logger.info("启用混合精度训练 (AMP)。")
    else:
        scaler = None
        logger.info("禁用混合精度训练 (AMP)。")

    for dataset_name in datasets_list:
        logger.info(f"\n=== 开始处理数据集：{dataset_name} ===")

        if dataset_name == 'CIFAR10':
            num_classes = 10
            input_channels = 3
        elif dataset_name == 'CIFAR100':
            num_classes = 100
            input_channels = 3
        elif dataset_name.lower() == 'imagenetmini':
            num_classes = 64  # 请根据实际类别数调整
            input_channels = 3
        else:
            raise ValueError("不支持的数据集。")

        # 初始化模型并从头训练
        model, model_type = get_model(dataset_name, num_classes=num_classes, input_channels=input_channels)
        model = initialize_model(model, device)
        optimizer = optim.Adam(model.parameters(), lr=args.model_lr)

        scheduler = None
        if args.use_scheduler:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
            logger.info("使用学习率调度器: StepLR")

        # ========== 方案1：使用 split1 数据训练 ==========
        logger.info(f"\n--- 方案1：使用 split1 数据训练模型 ---")
        try:
            train_loader_scheme1, mean, std = get_dataloader(dataset_name, split='split1',
                                                             batch_size=args.batch_size,
                                                             root_dir=args.root_dir)
        except Exception as e:
            logger.error(f"加载数据集 {dataset_name} 的 split1 失败: {e}")
            continue

        if len(train_loader_scheme1.dataset) == 0:
            logger.error("训练数据加载失败，数据集为空。请检查数据划分文件。")
            continue

        # Phase 1: 标准训练
        logger.info(f"\n--- Phase 1: 标准训练 {args.standard_epochs} 轮 ---")
        try:
            model = train_standard(
                model=model,
                train_loader=train_loader_scheme1,
                device=device,
                num_epochs=args.standard_epochs,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler
            )
        except Exception as e:
            logger.error(f"标准训练过程中发生错误: {e}")
            continue

        # Phase 2: 对抗训练
        logger.info(f"\n--- Phase 2: 对抗训练 {args.conftr_epochs} 轮 ---")
        try:
            # 创建保存目录
            save_scheme = 'phase1_and_phase2'
            save_dir = os.path.join(model_save_root, dataset_name, save_scheme)
            os.makedirs(save_dir, exist_ok=True)

            model, best_loss = train_conftr(
                model=model,
                conformal_train_loader=train_loader_scheme1,
                alpha=args.alpha,
                device=device,
                num_epochs=args.conftr_epochs,
                T_train=args.T_train,
                optimizer=optimizer,
                scheduler=scheduler,
                lambda_weight=args.lambda_weight,
                scaler=scaler,
                atk_epsilon=args.atk_epsilon,
                atk_atkIter=args.atk_atkIter,
                atk_lr=args.atk_lr,
                atk_T=args.atk_T,
                atk_norm=args.atk_norm,
                dataset_name=dataset_name,
                save_dir=save_dir,
                model_type=model_type,
                save_interval=args.save_interval
            )
        except Exception as e:
            logger.error(f"对抗训练过程中发生错误: {e}")
            continue

        # 保存最终模型
        try:
            save_model(
                model=model,
                save_dir=save_dir,
                dataset_name=dataset_name,
                scheme=save_scheme,
                model_type=model_type
            )
            logger.info(f"最终模型已保存")
        except Exception as e:
            logger.error(f"保存最终模型失败: {e}")

        logger.info(f"\n=== 数据集 {dataset_name} 训练已完成 ===")

    logger.info("\n====== 所有训练已完成 ======")


if __name__ == '__main__':
    main()