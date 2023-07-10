import copy
import os
import pickle
import time
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms


def get_sampler():
    if os.path.exists('sampler.pkl'):
        # 从文件中加载 sampler
        with open('sampler.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        # 计算每个类别的权重
        class_counts = Counter(img[1] for img in image_datasets['train'])
        class_weights = {c: float(min(class_counts.values())) / class_counts[c] for c in class_counts}
        class_weights = [class_weights[c] for c in range(len(class_counts))]
        class_weights = torch.tensor(class_weights, dtype=torch.float)

        # 创建可调整权重的采样器
        sampler = WeightedRandomSampler(weights=class_weights, num_samples=len(class_weights), replacement=True)

        # 将 sampler 保存到文件中
        with open('sampler.pkl', 'wb') as f:
            pickle.dump(sampler, f)

        return sampler


def get_model(_models_dir: str, name: str, nclass: int) -> nn.Module:
    idx = 0
    while os.path.exists(os.path.join(_models_dir, '%s_%d.pth' % (name, idx))):
        idx = idx + 1
    else:
        if idx > 0:
            _model = models.resnet50()
            _model.fc = nn.Linear(_model.fc.in_features, len(class_names))

            _model.load_state_dict(torch.load(os.path.join(_models_dir, '%s_%d.pth' % (name, (idx - 1)))))
            _model.to(device)
            _model.eval()
            print('Loading model %d.' % (idx - 1))
        else:
            url = "https://download.pytorch.org/models/resnet50-0676ba61.pth"

            _model = models.resnet50()
            state_dict = torch.hub.load_state_dict_from_url(url=url)
            _model.load_state_dict(state_dict)

            _model.fc = nn.Linear(_model.fc.in_features, nclass)
            # Here the size of each output sample is set to 2.
            # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
            _model = _model.to(device)
    return _model


def train_model(_model, _criterion, _optimizer, _scheduler, _num_epochs=25):
    since = time.time()

    best_acc = 0.0
    best_model_wts = copy.deepcopy(_model.state_dict())

    for epoch in range(_num_epochs):
        print('Epoch {}/{}'.format(epoch, _num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                _model.train()  # Set model to training mode
            else:
                _model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                _optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = _model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = _criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        _optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(input=labels.data)
            if phase == 'train':
                _scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(_model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    _model.load_state_dict(best_model_wts)
    return _model


def save_model(_model: nn.Module, _models_dir: str, name: str):
    # Save model
    idx = 0
    while os.path.exists(os.path.join(_models_dir, '%s_%d.pth' % (name, idx))):
        idx = idx + 1
    else:
        torch.save(_model.state_dict(), os.path.join(_models_dir, '%s_%d.pth' % (name, idx)))


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

data_dir = './data'
models_dir = 'models'
if not os.path.exists(models_dir):
    os.mkdir(models_dir)

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

sampler: WeightedRandomSampler = get_sampler()

# 创建数据加载器
dataloaders['train'] = DataLoader(image_datasets['train'], batch_size=32, sampler=sampler, num_workers=4)
dataloaders['val'] = DataLoader(image_datasets['val'], batch_size=32, sampler=sampler, num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 使用预训练的ResNet-50模型
model = get_model(models_dir, 'model50', len(class_names))

model = model.to(device)

criterion = nn.CrossEntropyLoss()

# 优化器
optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.9)

# 学习率调整策略
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.9)

# 训练模型
model = train_model(model, criterion, optimizer, exp_lr_scheduler, _num_epochs=1)

save_model(model, models_dir, 'model50')
