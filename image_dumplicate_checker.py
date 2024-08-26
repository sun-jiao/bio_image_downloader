import os
from os import listdir

from PIL import Image
import imagehash
from collections import defaultdict
from tqdm import tqdm


def find_duplicate_images(directory, hash_size=8, hash_threshold=5):
    """
    遍历给定目录下的所有图片，使用感知哈希检查是否存在相同的图片。

    :param directory: 图片所在的目录
    :param hash_size: 感知哈希的大小，默认是8x8
    :param hash_threshold: 哈希距离的阈值，决定了图片相似的标准
    :return: 重复图片组的列表
    """
    image_hashes = defaultdict(list)
    duplicates = []

    # 使用os.walk遍历目录和子目录
    for root, _, files in tqdm(os.walk(directory)):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
                file_path = os.path.join(root, file)

                # 打开图像并计算感知哈希
                try:
                    image = Image.open(file_path)
                    image_hash = imagehash.phash(image, hash_size=hash_size)
                    image_hashes[str(image_hash)].append(file_path)
                except Exception as e:
                    print(f"无法处理图像 {file_path}: {e}")
                    continue

    # 处理哈希分组，计算相似图片
    for hash_val, paths in image_hashes.items():
        if len(paths) > 1:
            group = []
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    hash1 = imagehash.phash(Image.open(paths[i]), hash_size=hash_size)
                    hash2 = imagehash.phash(Image.open(paths[j]), hash_size=hash_size)
                    hash_distance = hash1 - hash2
                    if hash_distance <= hash_threshold:
                        if paths[i] not in group:
                            group.append(paths[i])
                        if paths[j] not in group:
                            group.append(paths[j])
            if group:
                duplicates.append(group)

    return duplicates


def clear_intra_class_duplicates(parent):
    for directory in listdir(parent):
        print(f'正在处理目录：{directory}')

        duplicates = find_duplicate_images(os.path.join(parent, directory))

        if duplicates:
            print(f"找到 {len(duplicates)} 组重复图片：")
            for group in duplicates:
                for index, path in enumerate(group):
                    if index != 0:
                        os.remove(path)
                        print(f'删除重复文件：{path}')
        else:
            print("没有找到重复的图片。")


def clear_inter_class_duplicates(directory):
    print(f'正在处理目录：{directory}')

    duplicates = find_duplicate_images(os.path.join(parent, directory))

    if duplicates:
        print(f"找到 {len(duplicates)} 组重复图片：")
        for group in duplicates:
            for index, path in enumerate(group):
                os.remove(path)
                print(f'删除重复文件：{path}')
    else:
        print("没有找到重复的图片。")



if __name__ == "__main__":
    parent = "/run/media/sunjiao/wd5t/DeepBirdIdentification/Dongniao-DIB-10K/dib/"  # 修改为你的图片目录路径

    clear_inter_class_duplicates(parent)

