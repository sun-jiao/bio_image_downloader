import os


def fix():
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