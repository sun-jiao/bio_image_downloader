# -*- coding: utf-8 -*-
"""
Transfer Learning for Computer Vision Tutorial
==============================================
**Author**: `Sasank Chilamkurthy <https://chsasank.github.io>`_

In this tutorial, you will learn how to train a convolutional neural network for
image classification using transfer learning. You can read more about the transfer
learning at `cs231n notes <https://cs231n.github.io/transfer-learning/>`__

Quoting these notes,

    In practice, very few people train an entire Convolutional Network
    from scratch (with random initialization), because it is relatively
    rare to have a dataset of sufficient size. Instead, it is common to
    pretrain a ConvNet on a very large dataset (e.g. ImageNet, which
    contains 1.2 million images with 1000 categories), and then use the
    ConvNet either as an initialization or a fixed feature extractor for
    the task of interest.

These two major transfer learning scenarios look as follows:

-  **Finetuning the ConvNet**: Instead of random initialization, we
   initialize the network with a pretrained network, like the one that is
   trained on imagenet 1000 dataset. Rest of the training looks as
   usual.
-  **ConvNet as fixed feature extractor**: Here, we will freeze the weights
   for all of the network except that of the final fully connected
   layer. This last fully connected layer is replaced with a new one
   with random weights and only this layer is trained.

"""
# License: BSD
# Author: Sasank Chilamkurthy

import os
import pickle
import time
from collections import Counter
from tempfile import TemporaryDirectory

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import _Loss
from torch.optim import lr_scheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import WeightedRandomSampler
from torchvision import datasets, models, transforms


def get_sampler() -> WeightedRandomSampler:
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


######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.


def train_model(_model, _criterion, _optimizer, _scheduler, dataloaders, device, dataset_sizes, _num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(_model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(_num_epochs):
            print(f'Epoch {epoch}/{_num_epochs - 1}')
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
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    _scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(_model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        _model.load_state_dict(torch.load(best_model_params_path))
    return _model


def freeze_model(model: nn.Module) -> nn.Module:
    ######################################################################
    # Freeze all the network except the final layer. We need
    # to set ``requires_grad = False`` to freeze the parameters so that the
    # gradients are not computed in ``backward()``.
    #
    # You can read more about this in the documentation
    # `here <https://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward>`__.
    #

    for param in model.parameters():
        param.requires_grad = False

    return model


def get_model(_models_dir: str, name: str, nclass: int, device, freeze: bool) -> (nn.Module, _Loss, Optimizer, lr_scheduler.StepLR):
    index = 0
    while os.path.exists(os.path.join(_models_dir, '%s_%d.pth' % (name, index))):
        index = index + 1
    else:
        if index > 0:
            _model = models.resnet152()
            if freeze:
                _model = freeze_model(_model)

            _model.fc = nn.Linear(_model.fc.in_features, nclass)

            _model.load_state_dict(torch.load(os.path.join(_models_dir, '%s_%d.pth' % (name, (index - 1)))))
            _model = _model.to(device)
            print('Loading model %d.' % (index - 1))
        else:
            _model = models.resnet152(pretrained='IMAGENET1K_V2')
            if freeze:
                _model = freeze_model(_model)

            # Here the size of each output sample is set to 2.
            # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
            _model.fc = nn.Linear(_model.fc.in_features, nclass)
            _model = _model.to(device)

    _criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized if not freeze,
    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    params = _model.fc.parameters() if freeze else _model.parameters()

    _optimizer_conv = optim.SGD(params, lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    _exp_lr_scheduler = lr_scheduler.StepLR(_optimizer_conv, step_size=7, gamma=0.1)

    return _model, _criterion, _optimizer_conv, _exp_lr_scheduler


def save_model(_model: nn.Module, _models_dir: str, name: str):
    # Save model
    idx = 0
    while os.path.exists(os.path.join(_models_dir, '%s_%d.pth' % (name, idx))):
        idx = idx + 1
    else:
        torch.save(_model.state_dict(), os.path.join(_models_dir, '%s_%d.pth' % (name, idx)))


cudnn.benchmark = True

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
# The problem we're going to solve today is to train a model to classify
# **ants** and **bees**. We have about 120 training images each for ants and bees.
# There are 75 validation images for each class. Usually, this is a very
# small dataset to generalize upon, if trained from scratch. Since we
# are using transfer learning, we should be able to generalize reasonably
# well.
#
# This dataset is a very small subset of imagenet.
#
# .. Note ::
#    Download the data from
#    `here <https://download.pytorch.org/tutorial/hymenoptera_data.zip>`_
#    and extract it to the current directory.

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
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'data'
models_dir = 'models'
model_name = 'model152'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

sampler: WeightedRandomSampler = get_sampler()

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, sampler=sampler, num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

######################################################################
# Finetuning the ConvNet
# ----------------------
#
# Load a pretrained model and reset final fully connected layer.
#

model_ft, criterion, optimizer_ft, exp_lr_scheduler = get_model(models_dir, model_name, len(class_names), device, freeze=False)

######################################################################
# Train and evaluate
#

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, device, dataset_sizes, _num_epochs=1000)

save_model(model_ft, models_dir, model_name)
