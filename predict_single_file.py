# import re
import json

import torch
from torch import nn

from PIL import Image
from torchvision.models import resnet34
from torchvision.transforms import transforms

import metafg_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image_path = "test_images/86810585ED8E85C2CE8525BB8E17CF07.jpg"

with open('birdinfo.json', 'r') as f:
    data = f.read()

# 解码JSON格式的数据
bird_info = json.loads(data)


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
# model = metafg_model.get_metafg_model()

model = torch.jit.load('model20240824-2.pt')

# model = resnet34()
# model.fc = nn.Linear(model.fc.in_features, 11000)
# model.load_state_dict(torch.load('model20240824.pth'))

model = model.to(device).eval()
# model = ipex.optimize(model)

img, data = image_proprecess(image_path)
data = data.to(device)

outputs = model(data)

# get top 5 results
probs_full = torch.nn.functional.softmax(outputs, dim=1)
probs, indices = torch.topk(probs_full, k=5, dim=1)
probs = probs.squeeze().tolist()

indices = indices.squeeze().tolist()
indices_zh = [bird_info[indice][0] for indice in indices]
indices_en = [bird_info[indice][1] for indice in indices]
indices_scientific_name = [bird_info[indice][2] for indice in indices]
probs_round = [f'{round(prob * 100, 3)}%' for prob in probs]

print(indices)
print(indices_zh)
print(indices_en)
print(indices_scientific_name)
print(probs_round)

