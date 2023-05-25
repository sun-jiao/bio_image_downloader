import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# 定义训练数据路径
train_data_path = './data/train'
test_data_path = './data/test'
val_data_path = './data/val'

# 定义数据增强和归一化操作
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# 定义训练、测试和验证数据集
train_dataset = datasets.ImageFolder(train_data_path, transform=data_transforms['train'])
test_dataset = datasets.ImageFolder(test_data_path, transform=data_transforms['test'])
val_dataset = datasets.ImageFolder(val_data_path, transform=data_transforms['val'])

# 计算每个类别的数据量
class_count = {}
for _, label in train_dataset:
    if label not in class_count:
        class_count[label] = 0
    class_count[label] += 1

# 使用过采样、欠采样、权重调整等方法平衡数据集
balanced_train_dataset = []
for image, label in train_dataset:
    count = class_count[label]
    if count >= 2000:
        balanced_train_dataset.append((image, label))
    else:
        # 使用过采样、欠采样或权重调整来平衡数据集
        oversampling_ratio = int(2000 / count)
        if oversampling_ratio > 1:
            for i in range(oversampling_ratio):
                balanced_train_dataset.append((image, label))
        else:
            balanced_train_dataset.append((image, label))

# 构建数据加载器
batch_size = 32
train_loader = torch.utils.data.DataLoader(balanced_train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# 加载预训练模型
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 100)

# 冻结模型参数
for param in model.parameters():
    param.requires_grad = False

# 修改最后一层的参数
for param in model.fc.parameters():
    param.requires_grad = True

# 定义优化器和损失函数
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练模型
num_epochs = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    # 训练模型
    num_epochs = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        test_loss = 0.0
        test_acc = 0.0

        # 训练模型
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_acc += torch.sum(preds=labels.data)

        # 测试模型
        model.eval()
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            test_acc += torch.sum(preds=labels.data)

        # 计算训练和测试的平均损失和准确率
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_acc / len(train_loader.dataset)
        test_loss = test_loss / len(test_loader.dataset)
        test_acc = test_acc / len(test_loader.dataset)

        # 输出训练和测试的损失和准确率
        print('Epoch: {} Train Loss: {:.4f} Train Acc: {:.4f} Test Loss: {:.4f} Test Acc: {:.4f}'.format(
            epoch + 1, train_loss, train_acc, test_loss, test_acc))

    # 保存模型
    torch.save(model.state_dict(), 'checkpoint.pth')
# 在上述代码中，我们首先定义了数据集的路径和数据增强、归一化操作。然后通过遍历训练集中每个样本的标签，计算每个标签对应的样本数量，以便后面进行数据平衡处理。接着，我们通过过采样、欠采样、权重调整等方法，对数据集进行了平衡处理，以便训练模型时对各个类别的样本进行充分学习。最后，我们加载了预训练模型ResNet50，并修改最后一层的参数，以便用于分类任务。我们还定义了优化器和损失函数，并在训练过程中进行了10个epoch的训练，将模型参数保存在checkpoint.pth文件中。

# 请注意，以上代码只是一个示例，实际使用时可能需要根据具体情况进行调整，例如调整数据增强操作、修改模型结构等。