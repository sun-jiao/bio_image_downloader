import os


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
