import copy
import math
import multiprocessing
import os
# import pickle
import time

import torch
import torch.nn as nn
import torch.optim as optim
from pycls.core.config import cfg
import pycls.core.builders as model_builder

from sklearn.utils import class_weight
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler
from torchvision import datasets, models, transforms
from torchvision.datasets.folder import default_loader
from torchvision.models import EfficientNet_V2_L_Weights
import torch.multiprocessing
# from collections import Counter
from PIL import ImageFile

if torch.cuda.is_available():
    from torch.cuda.amp import autocast as autocast, GradScaler

    scaler = GradScaler()

ImageFile.LOAD_TRUNCATED_IMAGES = True
# Assuming you want to sample 10% of the dataset, the ratio should be 0.1
sampling_ratio = 1.0

data_dir = './images'
models_dir = './models'
model_name = 'efv2l_bird'
num_class = 12000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

freeze = False


def get_sampler(dataset, is_train):
    # sampling ratio is defined out scope for usage in train.
    num_samples = int(math.ceil(len(dataset) * sampling_ratio))

    # 创建可调整权重的采样器
    if is_train:
        targets = dataset.targets  # 获取样本标签列表
        weights = class_weight.compute_sample_weight("balanced", targets)
        class_weights = torch.from_numpy(weights)

        _sampler = WeightedRandomSampler(weights=class_weights, num_samples=num_samples, replacement=True)
    else:
        _sampler = RandomSampler(data_source=dataset, num_samples=num_samples, replacement=True)

    return _sampler


def freeze_model(_model: nn.Module) -> nn.Module:
    ######################################################################
    # Freeze all the network except the final layer. We need
    # to set ``requires_grad = False`` to freeze the parameters so that the
    # gradients are not computed in ``backward()``.
    #
    # You can read more about this in the documentation
    # `here <https://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward>`__.
    #

    for param in _model.parameters():
        param.requires_grad = False

    return _model


def max_index_file(directory, prefix, suffix):
    max_index = -1
    max_file = None

    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith(suffix):
            # 提取索引部分
            index_str = filename[len(prefix) + 1: -len(suffix) - 1]
            try:
                index = int(index_str)
                if index > max_index:
                    max_index = index
                    max_file = filename
            except ValueError:
                continue

    return max_index, max_file


class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader):
        super(CustomImageFolder, self).__init__(root, transform, target_transform, loader)

        # 重新生成类别到索引的映射
        self.classes, self.class_to_idx = self._find_classes(self.root)

    def _find_classes(self, dir):
        class_to_idx = {d.split('.', 1)[1]: int(d.split('.')[0]) for d in os.listdir(dir) if
                        os.path.isdir(os.path.join(dir, d))}
        classes = [d.split('.', 1)[1] for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()

        return classes, class_to_idx


def get_model(_models_dir: str, name: str, _num_class: int, _freeze: bool) -> nn.Module:
    _, max_file = max_index_file(_models_dir, name, 'pth')

    if max_file is None:
        pretrained = EfficientNet_V2_L_Weights.DEFAULT
    else:
        pretrained = None
        print(f'Loading model {max_file}.')

    _model = models.efficientnet_v2_l(weights=pretrained)
    if _freeze:
        _model = freeze_model(_model)

    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    _model.classifier[1] = nn.Linear(_model.classifier[1].in_features, _num_class)

    if max_file is not None:
        _model.load_state_dict(torch.load(os.path.join(_models_dir, max_file), map_location=device))

    _model = _model.to(device)
    return _model


def train_model(_model, _dataloaders, _criterion, _optimizer, _scheduler, _num_epochs=25):
    since = time.time()

    best_acc = 0.0
    best_model_wts = copy.deepcopy(_model.state_dict())

    for epoch in range(_num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, _num_epochs - 1))

        # load best model weights
        # _model.load_state_dict(best_model_wts)

        for phase in ['train', 'val']:
            if phase == 'train':
                _model.train()  # Set model to training mode
            else:
                _model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in _dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                _optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # compute output
                    if torch.cuda.is_available():
                        with autocast():
                            outputs = _model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = _criterion(outputs, labels)
                        if phase == 'train':
                            # Scales loss. 为了梯度放大.
                            scaler.scale(loss).backward()

                            # scaler.step() 首先把梯度的值unscale回来.
                            # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
                            # 否则，忽略step调用，从而保证权重不更新（不被破坏）
                            scaler.step(_optimizer)

                            # 准备着，看是否要增大scaler
                            scaler.update()
                    else:
                        outputs = _model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = _criterion(outputs, labels)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            _optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = (running_loss / dataset_sizes[phase]) / sampling_ratio
            epoch_acc = (running_corrects.double() / dataset_sizes[phase]) / sampling_ratio

            if phase == 'train':
                _scheduler.step(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(_model.state_dict())
                torch.save(_model.state_dict(), os.path.join(models_dir, f'{model_name}_temp.pth'))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    _model.load_state_dict(best_model_wts)
    return _model


def save_model(_model: nn.Module, _models_dir: str, name: str):
    max_index, _ = max_index_file(_models_dir, name, 'pth')
    torch.save(_model.state_dict(), os.path.join(_models_dir, f'{name}_{max_index + 1}.pth'))


if __name__ == '__main__':
    # 数据增强和预处理
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    torch.multiprocessing.set_sharing_strategy('file_system')

    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    image_datasets = {x: CustomImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    samplers = {x: get_sampler(image_datasets[x], is_train=x == 'train')
                for x in ['train', 'val']}
    num_cpus = multiprocessing.cpu_count()

    # 创建数据加载器
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=196, sampler=samplers[x], num_workers=num_cpus)
                   for x in ['train', 'val']}

    # 使用模型
    model = get_model(models_dir, model_name, num_class or len(class_names), _freeze=freeze)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # 优化器
    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    params = model.classifier[1].parameters() if freeze else model.parameters()
    optimizer = optim.AdamW(params, lr=0.001)

    # 学习率调整策略
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", factor=0.1, patience=3, verbose=True, threshold=5e-3, threshold_mode="abs")

    # for i in range(100):  # uncomment本行时下面两行都应该缩进，否则会连训100轮不保存。
    # 训练模型
    model = train_model(model, dataloaders, criterion, optimizer, scheduler, _num_epochs=25)
    save_model(model, models_dir, model_name)
