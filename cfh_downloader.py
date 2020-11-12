import asyncio
import json
import math
import os
from datetime import datetime

import aiofiles
import aiohttp
import requests
from fake_useragent import UserAgent
from base_downloader import BaseDownloader


class CfhDownloader(BaseDownloader):
    host = 'www.cfh.ac.cn'
    
    def get_query_url(self):
            return 'http://www.cfh.ac.cn/ajaxserver/speciesserv.ashx?action=spsearchzh&keyword=' + self.name

    def get_header(self):
        return {
            'User-Agent': self.ua.random,
            'Host': 'www.cfh.ac.cn', }

    def get_id_from_query(self, species_json):
        if len(species_json) > 0:
            position_in_json = 0
            if self.check is not None:
                for index in range(len(species_json)):
                    if self.check & (self.name == species_json[index]['SPName']):
                        position_in_json = index
                        break
                    elif (not self.check) & (self.name == species_json[index]['Name_Zh']):
                        position_in_json = index
                        break
                    else:
                        continue

            self.id = species_json[position_in_json]['ID']
            if self.size <= 0:
                self.size = species_json[position_in_json]['PhotoCount']

    def get_image_list_url(self, index):
        return "http://www.cfh.ac.cn/AjaxServer/Server.ashx?service=photoset&method=get&spid=" + str(
                self.id) + "&pagesize=" + str(self.size) + "&page=" + str(index + 1)

    def get_image_url(self, json_item):
        return 'http://www.cfh.ac.cn' + \
               str(json_item['thumbnail']).replace('Thumbnail', 'Normal')

# test:
if __name__ == '__main__':
    start = datetime.now()

    butterfly = CfhDownloader(name="金裳凤蝶", directory="Troides aeacus test", page_size= 25, check=False)
    butterfly.get_species_id()
    butterfly.download()

    end = datetime.now()
    print("aiohttp版爬虫花费时间为：")
    print(end - start)