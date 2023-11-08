# import pickle

from torchvision import datasets

# from collections import Counter

if __name__ == '__main__':
    dataset = datasets.ImageFolder('data/train')
    print(dataset.class_to_idx)


