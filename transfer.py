# -*- coding: utf-8 -*-
"""transfer_learning_tutorial.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/62840b1eece760d5e42593187847261f/transfer_learning_tutorial.ipynb
"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
from __future__ import print_function, division

import copy
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms


# License: BSD
# Author: Sasank Chilamkurthy


def train_model(_model, _criterion, _optimizer, scheduler, num_epochs: int = 25):
    since = time.time()

    best_model_wts = copy.deepcopy(_model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                _model.train()  # Set model to training mode
            else:
                _model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for _inputs, labels in dataloaders[phase]:
                _inputs = _inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                _optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = _model(_inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = _criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        _optimizer.step()

                # statistics
                running_loss += loss.item() * _inputs.size(0)
                running_corrects += torch.sum(input=labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(_model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    _model.load_state_dict(best_model_wts)
    return _model


def get_model(_models_dir: str, name: str, nclass: int) -> nn.Module:
    idx = 0
    while os.path.exists(os.path.join(_models_dir, '%s_%d.pth' % (name, idx))):
        idx = idx + 1
    else:
        if idx > 0:
            _model = models.resnet50()
            # model.fc = nn.Linear(model.fc.in_features, len(class_names))

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


def save_model(_model: nn.Module, _models_dir: str, name: str):
    # Save model
    idx = 0
    while os.path.exists(os.path.join(_models_dir, '%s_%d.pth' % (name, idx))):
        idx = idx + 1
    else:
        torch.save(_model.state_dict(), os.path.join(_models_dir, '%s_%d.pth' % (name, idx)))


if __name__ == '__main__':

    data_dir = 'data'

    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    # Data augmentation and normalization for training
    # Just normalization for validation
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

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_model(models_dir, 'model50', len(class_names))

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.Adadelta(model.parameters(), lr=0.01)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Train and evaluate
    # It should take around 15-25 min on CPU. On GPU though, it takes less than a minute.

    model = train_model(model, criterion, optimizer, exp_lr_scheduler,
                        num_epochs=100)

    save_model(model, models_dir, 'model50')
