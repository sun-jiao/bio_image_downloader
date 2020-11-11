#coding:utf-8

import csv
from cfh_downloader import CfhSpecies

def open_csv(filename):
    with open(filename, encoding='gbk') as file:
        f_csv = csv.reader(file)
        for row in f_csv:
            butterfly = CfhSpecies(name=row[0], directory=row[1], check=False)
            butterfly.get_species_id()
            butterfly.download()