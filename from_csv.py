#coding:utf-8

import csv
import os

from cfh_downloader import CfhDownloader
from gbif_downloader import GbifDownloader


def from_csv_in_cfh(filename):
    with open(filename, encoding='gbk') as file:
        f_csv = csv.reader(file)
        for row in f_csv:
            butterfly = CfhDownloader(name=row[0], directory=row[1], check=False)
            butterfly.download()

def update_csv(filename, new_csv, directory = './download/'):
    new_csv_file = open(new_csv, 'a', encoding='gbk')
    with open(filename, 'r', encoding='gbk') as file:
        rows_csv = csv.reader(file)
        for row in rows_csv:
            folder_path = directory + row[1]
            if os.path.isdir(folder_path):
                length = str(len(os.listdir(folder_path)))
                new_csv_file.write(row[0] + ',' + row[1] + ',' + length + '\r\n')
            else:
                new_csv_file.write(row[0] + ',' + row[1] + ',' + '0' + '\r\n')

def from_csv_in_gbif(filename):
    with open(filename, encoding='gbk') as file:
        f_csv = csv.reader(file)
        for row in f_csv:
            butterfly = GbifDownloader(name=row[1], directory=row[1], check=True, size=200)
            butterfly.download()