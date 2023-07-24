import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
import requests
from io import BytesIO


class NetworkImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.image_paths = []
        self.class_labels = []

        for index, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name + '.txt')
            with open(class_path, 'r') as file:
                lines = file.read().splitlines()
                for line in lines:
                    self.image_paths.append(line)
                    self.class_labels.append(index)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_url = self.image_paths[index]
        while True:
            try:
                response = requests.get(image_url)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                    break
                else:
                    raise RuntimeError(f"Failed to download image from {image_url}")
            except RuntimeError as e:
                print(f"Error loading image: {e}. Retrying with a new image...")
                self.image_paths.pop(index)
                self.class_labels.pop(index)
                index = random.randint(0, len(self.image_paths) - 1)
                image_url = self.image_paths[index]

        label = torch.tensor(self.class_labels[index], dtype=torch.long)

        if self.transform is not None:
            image = self.transform(image)

        return image, label
