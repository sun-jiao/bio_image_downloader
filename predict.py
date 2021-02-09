import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms

model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 7)
checkpoint = torch.load(Path('models/fine_tuned_best_model_5.pt'))
model.load_state_dict(checkpoint)
trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

for imagepath in os.listdir('predict'):
    image = Image.open(os.path.join('predict', imagepath))

    input = trans(image)

    input = input.view(1, 3, 224, 224)

    output = model(input)

    prediction = int(torch.max(output.data, 1)[1].numpy())

    print(imagepath + '  ' + str(prediction))