import os
import random
import shutil


def split(train=9, val=1, test=0):
    total = train + val + test
    train_dir = './data/train/'
    val_dir = './data/val/'
    test_dir = './data/test/'
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    label_list = os.listdir(train_dir)
    for label in label_list:
        if not os.path.exists(os.path.join(val_dir, label)):
            os.makedirs(os.path.join(val_dir, label))
        if not os.path.exists(os.path.join(test_dir, label)):
            os.makedirs(os.path.join(test_dir, label))

        label_dir = train_dir + label
        file_list = os.listdir(label_dir)
        random.shuffle(file_list)
        val_amount = int(val * len(file_list) / total)
        for i in range(int((val + test) * len(file_list) / total)):
            file_dir = os.path.join(label_dir, file_list[i])
            if i < val_amount:
                shutil.move(file_dir, file_dir.replace('train', 'val'))
            else:
                shutil.move(file_dir, file_dir.replace('train', 'test'))
