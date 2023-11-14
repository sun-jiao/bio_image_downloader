import csv
import sys

from gbif_downloader import GbifDownloader

filename = sys.argv[1]
print(filename)

with open(filename, encoding='utf-8') as file:
    f_csv = csv.reader(file)
    for row in f_csv:
        downloader = GbifDownloader(name=row[1], directory=row[2], check=True, folder_size=2000,
                                   base_directory='./data/train/')
        downloader.download()
