#coding:utf-8

import csv
import os

from cfh_downloader import CfhDownloader
from gbif_downloader import GbifDownloader


def open_csv(filename):
    with open(filename, encoding='gbk') as file:
        f_csv = csv.reader(file)
        for row in f_csv:
            butterfly = CfhDownloader(name=row[0], directory=row[1], check=False)
            butterfly.download()

def update_csv(filename, directory):
    with open(filename, 'r', encoding='gbk') as file:
        rows_csv = csv.reader(file)
        for row in rows_csv:
            folder_path = directory + row[1]
            if os.path.isdir(folder_path):
                length = str(len(os.listdir(folder_path)))
                print(row[0] + ',' + row[1] + ',' + length)
            else:
                print(row[0] + ',' + row[1] + ',' + '0')

def from_csv_in_gbif(filename):
    with open(filename, encoding='gbk') as file:
        f_csv = csv.reader(file)
        for row in f_csv:
            butterfly = GbifDownloader(name=row[1], directory=row[1], check=True, size=200)
            butterfly.download()