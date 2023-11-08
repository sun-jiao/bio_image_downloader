import os
import random
import string

from PIL import Image

from augment_by_gpt import augment_data

def dataset_augment(data_dir):
    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path):
            num_images = len(os.listdir(subdir_path))
            if num_images < 50:
                for image_file in os.listdir(subdir_path):
                    if image_file.startswith('augmented'):
                        continue
                    image_path = os.path.join(subdir_path, image_file)
                    image = Image.open(image_path)
                    try:
                        augmented_image = augment_data(image)
                        new_image_path = check_and_rename_file(os.path.join(subdir_path, f"augmented_{image_file}"))
                        augmented_image.save(new_image_path)
                        print(f"Augmented image saved to {new_image_path}")
                    except:
                        print(f"Augment error for image {image_file}")
                        continue
            else:
                print(f"Subdirectory {subdir} contains {num_images} images.")


def check_and_rename_file(file_path):
    """
    检查文件是否存在，如果存在，则将其后面添加随机字符串，直到不存在类似文件为止。
    """
    if os.path.isfile(file_path):
        # 如果文件已经存在，则添加随机字符串
        file_name, file_ext = os.path.splitext(file_path) # 分离文件名和扩展名
        while True:
            # 生成一个随机字符串
            random_string = ''.join(str(random.randint(0, 1000)))
            new_file_path = file_name + '_' + random_string + file_ext
            if not os.path.isfile(new_file_path):
                # 如果新文件路径不存在，则返回新文件路径
                return new_file_path
    else:
        # 如果文件不存在，则直接返回原始文件路径
        return file_path


if __name__ == '__main__':
    dataset_augment("./data/train/")
