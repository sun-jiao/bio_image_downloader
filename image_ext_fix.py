import os
import sys

from PIL import Image


# fix image file extension and delete error file
def fix_files(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)

            # 检查文件是否为有效的图片文件
            try:
                img = Image.open(file_path)
                img.verify()
                image_format = img.format.lower()
            except (IOError, SyntaxError):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
                continue

            # 修改文件后缀名
            file_extension : str
            file_name, file_extension = os.path.splitext(file_path)
            if file_extension.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                continue

            # 获取推断的图片格式对应的后缀名
            if image_format == "jpeg":
                new_extension = ".jpg"
            elif image_format == "png":
                new_extension = ".png"
            elif image_format == "gif":
                new_extension = ".gif"
            elif image_format == "bmp":
                new_extension = ".bmp"
            else:
                continue

            new_file_path = file_name + new_extension
            os.rename(file_path, new_file_path)
            print(f"Renamed file: {file_path} -> {new_file_path}")


work_dir = filename = sys.argv[1]  # python image_ext_fix.py './data'
fix_files(work_dir)
