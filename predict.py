import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms

model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 1000)
checkpoint = torch.load(Path('models/model_0.pth'))
model.load_state_dict(checkpoint)
trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

with torch.no_grad():
    for imagepath in os.listdir('predict'):
        image = Image.open(os.path.join('predict', imagepath))

        input = trans(image)

        input = input.view(1, 3, 224, 224)

        output = model(input)
        _, prediction = torch.max(output, 1)

        print(imagepath + '  ' + str(prediction))