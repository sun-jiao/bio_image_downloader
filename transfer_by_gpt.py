import os
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights

# 数据增强和预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = './data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# 计算每个类别的权重
class_counts = Counter(img[1] for img in image_datasets['train'])
class_weights = {c: float(min(class_counts.values())) / class_counts[c] for c in class_counts}
class_weights = [class_weights[c] for c in range(len(class_counts))]
class_weights = torch.tensor(class_weights, dtype=torch.float)

# 创建可调整权重的采样器
sampler = WeightedRandomSampler(weights=class_weights, num_samples=len(class_weights), replacement=True)

# 创建数据加载器
dataloaders['train'] = DataLoader(image_datasets['train'], batch_size=32, sampler=sampler, num_workers=4)
dataloaders['val'] = DataLoader(image_datasets['val'], batch_size=32, shuffle=True, num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 使用预训练的ResNet-50模型
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features

# 修改最后一层，使输出为100个类别
model.fc = nn.Linear(num_ftrs, 100)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

# 优化器
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 学习率调整策略
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(input=labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

    return model


# 训练模型
model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=1)

torch.save(model.state_dict(), 'checkpoint.pth')
