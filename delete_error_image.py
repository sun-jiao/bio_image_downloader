import os
import sys

from PIL import Image


def delete_files(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img.verify()  # 验证图片文件的完整性
            except (IOError, SyntaxError):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")


work_dir = filename = sys.argv[1]  # python delete_error_image.py './data'
delete_files(work_dir)
