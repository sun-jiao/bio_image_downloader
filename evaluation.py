import datetime
import multiprocessing

import torch.multiprocessing
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train import get_model

# import pickle
# from collections import Counter

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_dataset = datasets.ImageFolder('./data/val', transform=transform)
class_labels = val_dataset.classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate(model):
    model = model.to(device)
    model.eval()

    num_cpus = multiprocessing.cpu_count()
    dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=num_cpus)
    nn.CrossEntropyLoss()

    correct = 0
    total = 0
    top3_correct = 0

    start = datetime.datetime.now().timestamp()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            _, top3 = torch.topk(outputs, k=3, dim=1)
            top3 = top3.squeeze().tolist()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            top3_correct += sum(labels.item() in top3_row for labels, top3_row in zip(labels, top3))

    end = datetime.datetime.now().timestamp()
    time = end - start

    accuracy = correct / total
    top3_accuracy = top3_correct / total
    print("Accuracy on the validation set: {:.2f}".format(accuracy))
    print("Top 3 accuracy on the validation set: {:.2f}".format(top3_accuracy))
    print(f"Time: {time}")


model = get_model('models', 'efv2l', len(class_labels), _freeze=False)

evaluate(model)
