# import re
import json

import torch
from PIL import Image
from torchvision.transforms import transforms

import metafg_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
imgs_path = "data/train/10937.Cryptic Honeyeater/181519057.jpg"
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
model = metafg_model.get_metafg_model()

model = model.to(device).eval()
# model = ipex.optimize(model)

img, data = image_proprecess(imgs_path)
data = data.to(device)

outputs = model(data)

# 获取最可能的三个类别及其概率
probs, indices = torch.topk(outputs, k=10, dim=1)
probs = torch.nn.functional.softmax(probs, dim=1)
probs = probs.squeeze().tolist()
indices = indices.squeeze().tolist()
indices_zh = [bird_info[indice][0] for indice in indices]
probs_round = [f'{round(prob * 100, 3)}%' for prob in probs]

print(indices_zh)
print(probs_round)

