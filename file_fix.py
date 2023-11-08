import os
from PIL import Image


def filename():
    download = './data/train/'
    folder_list = os.listdir(download)
    count = 0
    for folder in folder_list:
        file_list = os.listdir(download + folder + '/')
        for file in file_list:
            if file.__contains__('?'):
                file_dir = download + folder + '/' + file
                os.rename(file_dir, file_dir.replace('?', ''))
                count += 1
    print(count, "files was renamed.")


# https://stackoverflow.com/a/29452103/13447926 CC BY-SA 3.0
def empty_file(rootdir='./data/train/'):
    for root, dirs, files in os.walk(rootdir):
        for d in ['RECYCLER', 'RECYCLED']:
            if d in dirs:
                dirs.remove(d)

        for f in files:
            fullname = os.path.join(root, f)
            try:
                if os.path.getsize(fullname) == 0:
                    print(fullname)
                    os.remove(fullname)
            except WindowsError:
                continue


def remove_duplicate_files(path):
    if not os.path.exists(path):
        return

    files = os.listdir(path)
    for file in files:
        fullpath = os.path.join(path, file)
        if os.path.isdir(fullpath):
            remove_duplicate_files(fullpath)
        else:
            lower_files = [f.lower() for f in files]
            if lower_files.count(file.lower()) > 1:
                os.remove(fullpath)


def dump_number_fix():
    work_folder = './data/train'
    folder_names = os.listdir(work_folder)

    # 用于存储已经出现的序号
    seen_numbers = {}

    for folder_name in folder_names:
        names = folder_name.split('.')
        if not len(names) == 2:
            continue

        number = int(names[0])
        name = names[1]
        if number:
            if number in seen_numbers.keys():
                file = os.listdir(os.path.join(work_folder, folder_name))[0]  # 这是在第二次匹配到的文件夹
                if file.startswith(str(number)):  # 所以说明第二次匹配到的是旧名字
                    old_name = name  # old_name 是旧名字
                    name = seen_numbers[number]  # name是新名字
                else:
                    old_name = seen_numbers[number]
                for file in os.listdir(os.path.join(work_folder, f'{number}.{old_name}')):
                    file_dir = os.path.join(os.path.join(work_folder, f'{number}.{old_name}'), file)
                    os.rename(file_dir, file_dir.replace(old_name, name))
                os.removedirs(os.path.join(work_folder, f'{number}.{old_name}'))
            else:
                seen_numbers[number] = name


def resize_images(folder_path, max_resolution):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            try:
                image_path = os.path.join(root, file)
                image = Image.open(image_path)
                width, height = image.size

                if width > max_resolution or height > max_resolution:
                    ratio = min(max_resolution / width, max_resolution / height)
                    new_width = int(width * ratio)
                    new_height = int(height * ratio)
                    resized_image = image.resize((new_width, new_height))
                    resized_image.save(image_path)
                    print(f"Resized image: {image_path} ({width}x{height} -> {new_width}x{new_height})")
                else:
                    print(f"Skipping image: {image_path} ({width}x{height})")
            except OSError:
                continue
