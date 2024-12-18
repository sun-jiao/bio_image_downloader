import os
import random
import shutil


def copy_random_image_per_subfolder(source_directory, target_directory):
    """
    从给定目录的每个子文件夹中随机选择一张图片，并复制到目标目录。

    :param source_directory: 源文件夹路径
    :param target_directory: 目标文件夹路径
    """
    # 创建目标目录，如果不存在的话
    os.makedirs(target_directory, exist_ok=True)

    # 遍历源目录的每个子文件夹
    for subdir in os.listdir(source_directory):
        subdir_path = os.path.join(source_directory, subdir)

        if os.path.isdir(subdir_path):  # 检查是否是子文件夹
            # 获取子文件夹中的所有图片文件
            image_files = [f for f in os.listdir(subdir_path)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'))]

            if image_files:  # 如果子文件夹中有图片
                # 随机选择一张图片
                random_image = random.choice(image_files)
                source_image_path = os.path.join(subdir_path, random_image)

                # 复制图片到目标目录
                shutil.copy(source_image_path, os.path.join(target_directory, f"{subdir}_{random_image}"))
                print(f"从 {subdir_path} 中选择了 {random_image} 并复制到 {target_directory}")


if __name__ == "__main__":
    source_directory = "/run/media/sunjiao/wd5t/DeepBirdIdentification/Dongniao-DIB-10K/dib/"  # 替换为你的源文件夹路径
    target_directory = "one_image_per_species"  # 替换为你的目标文件夹路径

    copy_random_image_per_subfolder(source_directory, target_directory)
