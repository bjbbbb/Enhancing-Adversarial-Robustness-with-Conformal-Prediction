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
# Log Configuration
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
# Definition of Attack Methods
# ===============================
def maximize_prediction_set_opsa(model, device, x, y, eps, atkIter=10, lr=0.1, T=1.0, norm='l_inf'):
    """Implementation of optimized maximum prediction set attack"""
    model.eval()
    batch_size = x.shape[0]

    num_attempts = 2 

    x_adv = x.unsqueeze(1).repeat(1, num_attempts, 1, 1, 1).detach()
    x_adv = x_adv + (torch.rand_like(x_adv) * 2 * eps - eps)
    x_adv = torch.clamp(x_adv, 0, 1).requires_grad_(True)

    optimizer = optim.Adam([x_adv], lr=lr)

    best_perturbed = x.clone()
    best_margins = torch.full((batch_size,), float('-inf'), device=device)

    for itr in range(atkIter):
        optimizer.zero_grad()

        sub_batch_size = 2
        for j in range(0, num_attempts, sub_batch_size):
            current_x_adv = x_adv[:, j:min(j + sub_batch_size, num_attempts)].reshape(-1, *x.shape[1:])

            logits = model(current_x_adv)
            logits = torch.clamp(logits, min=-50, max=50) 
            logits = logits.view(batch_size, -1, logits.shape[-1])

            true_logits = torch.gather(logits, 2, y.view(batch_size, 1, 1).expand(-1, logits.shape[1], 1))
            margin = logits - true_logits

            sigma = torch.sigmoid(margin / T)
            sum_sigma = sigma.sum(dim=2)

            loss = -sum_sigma.mean()
            loss.backward()

        optimizer.step()
        
        with torch.no_grad():
            if norm == 'l_inf':
                delta = torch.clamp(x_adv.data - x.unsqueeze(1), min=-eps, max=eps)
                x_adv.data = x.unsqueeze(1) + delta
            x_adv.data = torch.clamp(x_adv.data, 0, 1)

        torch.cuda.empty_cache()

    with torch.no_grad():
        for j in range(num_attempts):
            current_x_adv = x_adv[:, j]

            logits = model(current_x_adv)
            logits = torch.clamp(logits, min=-1e3, max=1e3)

            true_logits = torch.gather(logits, 1, y.unsqueeze(1)).squeeze(1)
            margin = logits - true_logits.unsqueeze(1)

            sigma = torch.sigmoid(margin / T)
            sum_sigma = sigma.sum(dim=1)

            update_mask = sum_sigma > best_margins
            best_margins[update_mask] = sum_sigma[update_mask]
            best_perturbed[update_mask] = current_x_adv[update_mask]

            torch.cuda.empty_cache()

    return best_perturbed, best_margins

# ===============================
# Model Definition
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
Parameters:
    dataset_name (str): Name of the dataset. Supported datasets include 'CIFAR10', 'CIFAR100', 'ImageNetMini'.
    num_classes (int): Number of classes in the dataset.
    input_channels (int): Number of input channels (e.g., 3 for RGB images).

Returns:
    model (nn.Module): The configured model.
    model_type (str): Type of the model, such as 'resnet34', 'resnet50', or 'preact_resnet'.
    """
    if dataset_name == 'CIFAR100':
        model = models.resnet34(pretrained=False, num_classes=num_classes)
        model_type = 'resnet34'
    elif dataset_name == 'CIFAR10':
        model = models.resnet34(pretrained=False, num_classes=num_classes)
        model_type = 'resnet34'
    elif dataset_name == 'ImageNetMini':
        model = models.resnet50(pretrained=False, num_classes=num_classes)
        model_type = 'resnet50'
    else:
        model = PreActResNet18(num_classes=num_classes, input_channels=input_channels)
        model_type = 'preact_resnet18'

    if input_channels != 3 and dataset_name in ['CIFAR10', 'CIFAR100']:
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    return model, model_type

# ===============================
# Model weight initialization
# ===============================
def initialize_model(model, device):
    """
Parameters:
    model (nn.Module): The model to be initialized.
    device (torch.device): The device (e.g., CPU or GPU) on which the model will be initialized.

Returns:
    model (nn.Module): The initialized model.
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
# Load data partitioning index
# ===============================
def load_split_indices(dataset_name, split='split1', root_dir='/home/kaiyuan/PycharmProjects/icml_adversarial/data_splits'):
    if 'split1' in split or 'scheme1' in split:
        split_num = '1'
    elif 'split2' in split or 'scheme2' in split:
        split_num = '2'
    else:
        raise ValueError("The split parameter must contain either 'split 1' or 'split 2'")

    if dataset_name.lower() == 'imagenetmini':
        folder_name = 'mini-imagenet'
    else:
        folder_name = dataset_name

    split_path = os.path.join(root_dir, folder_name, f'split{split_num}', 'train_adv_indices.npy')
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"{split_path} non-existent. Please run the data partitioning script first.")
    indices = np.load(split_path)
    return indices

# ===============================
# Definition of Data Loader
# ===============================
def get_dataloader(dataset_name, split, batch_size=128, root_dir='/home/kaiyuan/PycharmProjects/icml_adversarial/data_splits'):
    if dataset_name == 'CIFAR10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        dataset = datasets.CIFAR10('/data', train=True, download=True, transform=transform)
    elif dataset_name == 'CIFAR100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        dataset = datasets.CIFAR100('/data', train=True, download=True, transform=transform)
    elif dataset_name.lower() == 'imagenetmini':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        data_dir = os.path.join('/data', 'mini-imagenet', 'train')
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    else:
        raise ValueError("Unsupported dataset. Please select 'CIFAR10', 'CIFAR100' or 'ImageNetMini'.")

    indices = load_split_indices(dataset_name, split=split, root_dir=root_dir)
    subset = Subset(dataset, indices)
    shuffle = True  
    num_workers = 2 if os.name != 'nt' else 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return loader, torch.tensor(mean).view(1, 3, 1, 1).to(device), torch.tensor(std).view(1, 3, 1, 1).to(device)

# ===============================
# conformal training
# ===============================
def compute_size_loss(probs, tau, T_train):
    clamped_input = torch.clamp((probs - tau) / T_train, min=-50.0, max=50.0)
    C_pred_sigmoid = torch.sigmoid(clamped_input)
    probs = torch.clamp(probs, min=1e-7, max=1-1e-7)
    size_loss = C_pred_sigmoid.sum(dim=1).mean()
    return size_loss

def compute_accuracy(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total

# ===============================
# Standard training function
# ===============================
def train_standard(model, train_loader, device, num_epochs, optimizer, scheduler=None, scaler=None):
    """
Standard training loop using cross-entropy loss.

Parameters:
    model (nn.Module): The model to be trained.
    train_loader (DataLoader): The training data loader.
    device (torch.device): The device (e.g., CPU or GPU) on which the training will be performed.
    num_epochs (int): The number of training epochs.
    optimizer (torch.optim.Optimizer): The optimizer used for training.
    scheduler (torch.optim.lr_scheduler, optional): The learning rate scheduler (optional).
    scaler (torch.cuda.amp.GradScaler, optional): The gradient scaler for mixed-precision training (optional).

Returns:
    model (nn.Module): The trained model.
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
            logger.warning("This round of training did not process any batches.")
            continue

        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0

        logger.info(f"** Standard Training Epoch {epoch}: Average loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f} **")
    return model

# ===============================
# Adversarial training
# ===============================
def save_model(model, save_dir, dataset_name, scheme, model_type, epoch=None):
    """
    Parameters:
    model (nn.Module): The model to be saved.
    save_dir (str): The directory where the model will be saved.
    dataset_name (str): The name of the dataset used for training.
    scheme (str): The partitioning scheme (e.g., 'scheme1' or 'phase1_and_phase2').
    model_type (str): The type of the model (e.g., 'resnet34' or 'resnet50').
    epoch (int, optional): The current training epoch. If provided, it will be appended to the filename.
    """
    os.makedirs(save_dir, exist_ok=True)

    if epoch is not None:
        model_save_path = os.path.join(save_dir, f"{dataset_name.lower()}_{scheme}_{model_type}_epoch{epoch}.pth")
    else:
        model_save_path = os.path.join(save_dir, f"{dataset_name.lower()}_{scheme}_{model_type}.pth")

    torch.save(model.state_dict(), model_save_path)
    logger.info(f"The model has been saved to {model_save_path}")


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
    Adversarial training loop with a conformal mechanism.

Parameters:
    model (nn.Module): The model to be trained.
    conformal_train_loader (DataLoader): The calibration training data loader for the conformal mechanism.
    alpha (float): The confidence level parameter for the conformal mechanism.
    device (torch.device): The device (e.g., CPU or GPU) on which the training will be performed.
    num_epochs (int): The number of training epochs.
    T_train (float): The temperature parameter for training.
    optimizer (torch.optim.Optimizer): The optimizer used for training.
    scheduler (torch.optim.lr_scheduler, optional): The learning rate scheduler (optional).
    lambda_weight (float): The weight for the size loss term.
    scaler (torch.cuda.amp.GradScaler, optional): The gradient scaler for mixed-precision training (optional).
    atk_epsilon (float): The perturbation size for adversarial attacks.
    atk_atkIter (int): The number of iterations for adversarial attacks.
    atk_lr (float): The learning rate for adversarial attacks.
    atk_T (float): The temperature parameter for adversarial attacks.
    atk_norm (str): The norm type for adversarial attacks (e.g., 'Linf' or 'L2').
    save_dir (str, optional): The directory where the model will be saved (optional).
    model_type (str, optional): The type of the model (optional).
    save_interval (int, optional): The interval (in epochs) at which the model will be saved (optional).

Returns:
    model (nn.Module): The trained model.
    best_loss (float): The best loss value achieved during the training process.
    """
    best_loss = float('-inf')
    best_model_state = None

    for epoch in range(1, num_epochs + 1):
        logger.info(f"Epoch {epoch}/{num_epochs}: Enable adversarial training.")
        model.train()

        epoch_size_loss = 0.0
        epoch_class_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        num_batches = 0

        batch_iterator = tqdm(conformal_train_loader, desc=f"Adversarial training Epoch {epoch}/{num_epochs}", leave=False)
        for batch in batch_iterator:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

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
                logger.error(f"An error occurred while generating adversarial samples: {e}")
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
                        logger.warning("scores_calib Empty, skip this batch optimization.")
                        continue
                    tau = np.quantile(scores_calib, 1 - alpha * (1 + 1 / len(scores_calib)))
                    tau = np.clip(tau, 1e-5, 1 - 1e-5)
                    logger.debug(f"  recalculate tau: {tau:.4f}")
                else:
                    logger.warning("The calibration set is empty, skip this batch optimization.")
                    continue

            with autocast(enabled=(scaler is not None)):
                outputs_train = model(Btrain_inputs)
                probs_train = nn.functional.softmax(outputs_train, dim=1)
                if torch.any(torch.isnan(probs_train)) or torch.any(torch.isinf(probs_train)):
                    logger.error("Probs_train contains NaN or Inf, skip this optimization.")
                    continue
                probs_train = torch.clamp(probs_train, min=1e-7, max=1 - 1e-7)
                clamped_input = torch.clamp((probs_train - tau) / T_train, min=-50.0, max=50.0)
                sigma = torch.sigmoid(clamped_input)
                if torch.any(torch.isnan(sigma)) or torch.any(torch.isinf(sigma)):
                    logger.error("Sigma contains NaN or Inf, skip this optimization.")
                    continue
                one_hot_labels = torch.nn.functional.one_hot(Btrain_labels, num_classes=probs_train.size(1)).float().to(
                    device)
                L_class = (sigma * (1 - one_hot_labels)).sum(dim=1).mean() + ((1 - sigma) * one_hot_labels).sum(
                    dim=1).mean()
                L_size = sigma.sum(dim=1).mean()
                loss = L_class + lambda_weight * L_size
                logger.debug(
                    f"  computer losses: L_class={L_class.item():.4f}, L_size={L_size.item():.4f}, Loss={loss.item():.4f}")

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Found NaN or Inf loss, skip this optimization. loss={loss.item()}")
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
                            logger.error(f"Skipping optimization step due to NaN or Inf gradients in parameter {name}.")
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
            logger.warning("This round of training did not process any batches.")
            continue

        avg_L_class = epoch_class_loss / num_batches
        avg_L_size = epoch_size_loss / num_batches
        total_avg_loss = avg_L_class + lambda_weight * avg_L_size
        avg_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0

        logger.info(
            f"  Average loss: L_class={avg_L_class:.4f}, L_size={avg_L_size:.4f}, total loss={total_avg_loss:.4f}, Accuracy={avg_accuracy:.4f}")

        if total_avg_loss > best_loss:
            best_loss = total_avg_loss
            best_model_state = model.state_dict().copy()
            logger.info(f"  New Best Loss: {best_loss:.4f}")

        if save_dir is not None and model_type is not None and epoch % save_interval == 0:
            current_model_state = model.state_dict().copy()
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
            logger.info(f"Saved the current best model at epoch {epoch}.")

            model.load_state_dict(current_model_state)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"{best_loss:.4f}")

    return model, best_loss


# ===============================
# main
# ===============================
def main():
    parser = argparse.ArgumentParser(description="Adversarial ConfTr: Adversarial Training with Coverage Guarantees")

    # Training parameters
    parser.add_argument('--standard_epochs', type=int, default=5, help='The number of training rounds for standard training')
    parser.add_argument('--conftr_epochs', type=int, default=40, help='The number of training rounds for adversarial training')  
    parser.add_argument('--save_interval', type=int, default=10, help='Save the number of interval rounds for the model')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--model_lr', type=float, default=0.005, help='Model optimizer learning rate')

    # Data partitioning parameters
    parser.add_argument('--root_dir', type=str, default='/data_splits')

    # Save parameters
    parser.add_argument('--save_dir', type=str, default='saved_models')

    # Training configuration
    parser.add_argument('--alpha', type=float, default=0.10, help='Confidence level parameter')
    parser.add_argument('--T_train', type=float, default=1, help='Training temperature parameters')
    parser.add_argument('--lambda_weight', type=float, default=5, help='Weight of size loss')

    # Adversarial training configuration
    parser.add_argument('--atk_epsilon', type=float, default=0.03, help='The disturbance size of adversarial attacks')
    parser.add_argument('--atk_atkIter', type=int, default=10, help='The number of iterations for adversarial attacks')
    parser.add_argument('--atk_lr', type=float, default=0.1, help='Learning rate of adversarial attacks')
    parser.add_argument('--atk_T', type=float, default=1.0, help='Temperature parameters for countering attacks')
    parser.add_argument('--atk_norm', type=str, default='l_inf', choices=['l_inf'], help='Norm types for adversarial attacks')

    # Equipment parameters
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])

    # Optimizer and Scheduler
    parser.add_argument('--use_scheduler', type=lambda x: (str(x).lower() == 'true'), default=False)

    # Mixed precision training
    parser.add_argument('--disable_amp', type=lambda x: (str(x).lower() == 'true'), default=False)

    args = parser.parse_args()

    if args.T_train <= 0:
        raise ValueError("T_train must be a positive number.")
    if args.alpha <= 0 or args.alpha >= 1:
        raise ValueError("Alpha must be within the interval (0, 1).")
    if args.lambda_weight < 0:
        raise ValueError("lambda_weight must be a non-negative number.")
    if args.model_lr <= 0:
        raise ValueError("model_lr must be a positive number.")
    if args.atk_epsilon < 0:
        raise ValueError("Atk_epsilon must be non negative.")
    if args.atk_atkIter <= 0:
        raise ValueError("Atk_atkIter must be a positive integer.")
    if args.atk_lr <= 0:
        raise ValueError("Atk_lr must be a positive number.")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using equipment: {device}')

    datasets_list = ['CIFAR10', 'CIFAR100', 'ImageNetMini']
    model_save_root = args.save_dir

    if not args.disable_amp and device.type == 'cuda':
        scaler = GradScaler()
        logger.info("Enable mixed precision training (AMP)")
    else:
        scaler = None
        logger.info("Disable mixed precision training (AMP)")

    for dataset_name in datasets_list:
        logger.info(f"\n=== Start processing dataset: {dataset_name} ===")

        if dataset_name == 'CIFAR10':
            num_classes = 10
            input_channels = 3
        elif dataset_name == 'CIFAR100':
            num_classes = 100
            input_channels = 3
        elif dataset_name.lower() == 'imagenetmini':
            num_classes = 64 
            input_channels = 3
        else:
            raise ValueError("Unsupported dataset.")

        model, model_type = get_model(dataset_name, num_classes=num_classes, input_channels=input_channels)
        model = initialize_model(model, device)
        optimizer = optim.Adam(model.parameters(), lr=args.model_lr)

        scheduler = None
        if args.use_scheduler:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
            logger.info("Using a learning rate scheduler: StepLR")

        # ========== Option 1: Train with split1 data ==========
        logger.info(f"\n--- Option 1: Train the model using split1 data ---")
        try:
            train_loader_scheme1, mean, std = get_dataloader(dataset_name, split='split1',
                                                             batch_size=args.batch_size,
                                                             root_dir=args.root_dir)
        except Exception as e:
            logger.error(f"Failed to load split1 of dataset {dataset_name}: {e}")
            continue

        if len(train_loader_scheme1.dataset) == 0:
            logger.error("Training data loading failed, dataset is empty. Please check the data partitioning file.")
            continue

        # Phase 1: Standard Training
        logger.info(f"\n--- Phase 1: Standard Training {args.standard_epochs} epoch ---")
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
            logger.error(f"An error occurred during the standard training process: {e}")
            continue

        # Phase 2: Adversarial training
        logger.info(f"\n--- Phase 2: Adversarial training {args.conftr_epochs} epoch ---")
        try:
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
            logger.error(f"An error occurred during the adversarial training process: {e}")
            continue

        # Save the final model
        try:
            save_model(
                model=model,
                save_dir=save_dir,
                dataset_name=dataset_name,
                scheme=save_scheme,
                model_type=model_type
            )
            logger.info(f"The final model has been saved")
        except Exception as e:
            logger.error(f"Failed to save the final model: {e}")

        logger.info(f"\n=== Training for dataset {dataset_name} has been completed. ===")

    logger.info("\n====== All training has been completed ======")


if __name__ == '__main__':
    main()
