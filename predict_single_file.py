# import re

import torch
from PIL import Image
from torchvision.transforms import transforms

from train import get_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
imgs_path = "data/train/10937.Cryptic Honeyeater/181519057.jpg"


def image_proprecess(img_path):
    img = Image.open(img_path).convert('RGB')
    data = data_transforms(img)
    data = torch.unsqueeze(data, 0)
    img_resize = img.resize((384, 384))
    return img_resize, data


data_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

# 使用模型
model = get_model('models', 'efv2l_bird', 12000, _freeze=False)

model = model.to(device).eval()
# model = ipex.optimize(model)

img, data = image_proprecess(imgs_path)
data = data.to(device)

outputs = model(data)

# 获取最可能的三个类别及其概率
probs, indices = torch.topk(outputs, k=10, dim=1)
probs = torch.nn.functional.hardsigmoid(probs)
probs = probs.squeeze().tolist()
indices = indices.squeeze().tolist()

print(indices)
print(probs)

