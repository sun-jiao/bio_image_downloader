import os
import random
import string


def add_prefix_suffix_to_files(directory):
    # 遍历指定目录下的所有文件夹
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            # 获取文件的扩展名
            file_name_without_ext, file_extension = os.path.splitext(file_name)

            # 获取文件夹名称作为前缀
            folder_name = os.path.basename(root)

            # 生成随机的大写字母和数字作为后缀
            suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))

            # 构建新的文件名
            new_name = f"{folder_name}_{file_name_without_ext}_{suffix}{file_extension}"

            # 获取文件的完整路径
            old_file_path = os.path.join(root, file_name)
            new_file_path = os.path.join(root, new_name)

            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {old_file_path} -> {new_file_path}")


# 使用该函数时，替换下面的路径为你指定的目录
directory = "data"
add_prefix_suffix_to_files(directory)
