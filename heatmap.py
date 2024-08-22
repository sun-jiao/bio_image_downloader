import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import resnet50
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor

import metafg_model


def to_tensor(img):
    transform_fn = Compose([Resize(249), CenterCrop(224), ToTensor(),
                            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform_fn(img)

def show_img(img):
    img = np.asarray(img)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def show_img2(img1, img2, alpha=0.8):
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    plt.figure(figsize=(10, 10))
    plt.imshow(img1)
    plt.imshow(img2, alpha=alpha)
    plt.axis('off')
    plt.show()

# 获取ResNet模型的中间层输出
def get_intermediate_output(model, x, target_layer):
    intermediate_output = None
    for name, layer in model.named_children():
        x = layer(x)
        if name == target_layer:
            intermediate_output = x
            break
    return intermediate_output


# 载入 ResNet 模型
model = metafg_model.get_metafg_model()
model.eval()

# 加载图片并预处理
img = Image.open('data/val/6151.White-spotted Wattle-eye/62171156.jpg')
x = to_tensor(img)

target_layer = 'layer4'  # ResNet 中的最后一个残差块
intermediate_output = get_intermediate_output(model, x.unsqueeze(0), target_layer)

# 从选定的层中提取特征图
feature_map = intermediate_output.squeeze(0).detach()

# 可视化
img_resized = x.permute(1, 2, 0) * 0.5 + 0.5

# 将特征图缩放到与原始图像相同的尺寸
cls_resized = F.interpolate(feature_map.mean(dim=0, keepdim=True).unsqueeze(0),
                            (224, 224), mode='bilinear').squeeze().cpu()

# 显示原图和特征图的叠加
show_img(img)
show_img(cls_resized)
show_img2(img_resized, cls_resized, alpha=0.8)