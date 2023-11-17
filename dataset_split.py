import os
import random


def split(train=9, val=1):
    total = train + val
    train_dir = './data/train/'
    val_dir = './data/val/'
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    label_list = os.listdir(train_dir)
    for label in label_list:
        if not os.path.exists(os.path.join(val_dir, label)):
            os.makedirs(os.path.join(val_dir, label))
        else:
            continue

        label_dir = train_dir + label
        file_list = os.listdir(label_dir)
        random.shuffle(file_list)
        val_amount = int(val * len(file_list) / total)
        for i in range(val_amount):
            file_dir = os.path.join(label_dir, file_list[i])
            os.rename(file_dir, file_dir.replace('train', 'val'))
