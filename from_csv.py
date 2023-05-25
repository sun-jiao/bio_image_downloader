# coding:utf-8

import csv
import os
import shutil

from cfh_downloader import CfhDownloader
from gbif_downloader import GbifDownloader


def from_csv_in_cfh(filename):
    with open(filename, encoding='utf-8') as file:
        f_csv = csv.reader(file)
        for row in f_csv:
            butterfly = CfhDownloader(name=row[1], directory=row[2], check=True, base_directory='./data/train/')
            butterfly.download()


def update_csv(filename, new_csv, directory='./data/train/'):
    new_csv_file = open(new_csv, 'a', encoding='utf-8')
    with open(filename, 'r', encoding='utf-8') as file:
        rows_csv = csv.reader(file)
        for row in rows_csv:
            folder_path = directory + row[2]
            if os.path.isdir(folder_path):
                length = str(len(os.listdir(folder_path)))
                new_csv_file.write(row[0] + ',' + row[2] + ',' + length + '\r\n')
            else:
                new_csv_file.write(row[0] + ',' + row[2] + ',' + '0' + '\r\n')


def from_csv_in_gbif(filename):
    with open(filename, encoding='utf-8') as file:
        f_csv = csv.reader(file)
        for row in f_csv:
            butterfly = GbifDownloader(name=row[1], directory=row[2], check=True, folder_size=2000,
                                       base_directory='./data/train/')
            butterfly.download()


def from_csv_move_file(filename):
    with open(filename, encoding='gbk') as file:
        f_csv = csv.reader(file)
        for row in f_csv:
            source_dir = './download/' + row[1]
            target_dir = './data/train/' + row[2]

            if os.path.exists(source_dir):
                file_names = os.listdir(source_dir)

                if (len(file_names) > 0) and (not os.path.exists(target_dir)):
                    os.makedirs(target_dir)

                for file_name in file_names:
                    shutil.move(os.path.join(source_dir, file_name), os.path.join(target_dir, file_name))

