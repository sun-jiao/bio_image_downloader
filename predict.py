import random

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import datasets

# 加载测试集
val_dataset = datasets.ImageFolder('data/val')
class_labels = val_dataset.classes

# 加载预训练的 ResNet-152 模型
model = models.efficientnet_v2_l()
num_ftrs = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_ftrs, len(class_labels))  # 将最后一层的输出调整为你的问题的类别数
model.load_state_dict(torch.load('efv2l_zb.pth', map_location=torch.device('cpu')))

model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建一个2x3的子图布局
fig, axes = plt.subplots(3, 2, figsize=(10, 15))

for i in range(6):
    ax = list(axes.flatten())[i]

    # 随机选择一张测试图片
    random_index = random.randint(0, len(val_dataset) - 1)
    image_path, target = val_dataset.imgs[random_index]

    # 加载并预测图像
    image = Image.open(image_path).convert('RGB')
    # 加载和显示图片
    ax.imshow(image)
    ax.axis("off")  # 关闭坐标轴

    image = transform(image)
    image = torch.unsqueeze(image, 0)  # 添加一个维度作为 batch
    outputs = model(image)

    # 获取最可能的三个类别及其概率
    probs, indices = torch.topk(outputs, k=3, dim=1)
    probs = torch.nn.functional.hardsigmoid(probs)
    probs = probs.squeeze().tolist()
    indices = indices.squeeze().tolist()

    # 添加标题和内容
    title = f"label:{class_labels[target]}"
    content = ''
    for j in range(len(indices)):
        content += f"{class_labels[indices[j]]},"

    ax.set_title(title)
    ax.text(0.5, -0.1, content, transform=ax.transAxes, ha="center")

# 调整子图之间的间距和边界
plt.subplots_adjust(hspace=0.3, wspace=0.3, bottom=0.1, top=0.9)

# 显示图像列表
plt.show()
