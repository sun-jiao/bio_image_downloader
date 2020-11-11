import asyncio
import json
import math
import os
from datetime import datetime

import aiofiles
import aiohttp
import requests
from fake_useragent import UserAgent

async def fetch(client, url):
    ua = UserAgent(use_cache_server=False)
    async with client.get(url, headers={
        'User-Agent': ua.random,
        'Host': 'www.cfh.ac.cn', }) as resp:
        return await resp.read()


async def async_save(url, directory):
    async with aiohttp.ClientSession() as client:
        image = await fetch(client, url)
        split_list = url.split('/')
        f = await aiofiles.open(directory + '/' + split_list[len(split_list) - 1], mode='wb')
        await f.write(image)
        await f.close()


class CfhSpecies:
    def __init__(self, name, directory, size = 0, page_size = 25, check = None):
        self.name = name # species name, scientific or vernacular are both ok
        self.directory = './download/cfh/' + directory
        self.size = size
        self.downloaded = 0
        self.ua = UserAgent(use_cache_server=False)
        self.id = None
        self.page_size = page_size
        self.check = check # name check, None: DO NOT check; True: scientific name; False: chinese vernacular name

    def get_species_id(self):
        id_query_url = 'http://www.cfh.ac.cn/ajaxserver/speciesserv.ashx?action=spsearchzh&keyword=' + self.name

        species_data = requests.get(id_query_url, headers={
            'User-Agent': self.ua.random,
            'Host': 'www.cfh.ac.cn', })
        species_json = json.loads(species_data.text)

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
            if self.size == 0:
                self.size = species_json[position_in_json]['PhotoCount']

    def download(self):
        if (self.id is None) | (self.size <= 0):
            print("Nothing to download for " + self.name + " to " + self.directory)
            return
        for index in range(math.ceil(self.size/self.page_size)):
            image_list_url = "http://www.cfh.ac.cn/AjaxServer/Server.ashx?service=photoset&method=get&spid=" + str(
                self.id) + "&pagesize=" + str(self.size) + "&page=" + str(index + 1)
            image_list = requests.get(image_list_url, headers={
                'User-Agent': self.ua.random,
                'Host': 'www.cfh.ac.cn', })

            try:
                data = json.loads(image_list.text)['photolist']
            except KeyError:
                break

            # download links in this page.
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)

            photo_list = []

            while self.downloaded < self.size:
                img_url = 'http://www.cfh.ac.cn' + \
                          str(data[self.downloaded]['thumbnail']).replace('Thumbnail', 'Normal')
                print('Downloading ' + str(self.downloaded + 1) + '/' + str(self.size) +': ' + str(img_url))

                # sync download Start
                '''
                try:
                    pic = requests.get(img_url, timeout=10 * random.randint(1, 10))

                    pic_name = self.directory + '/' + str(self.downloaded + 1) + '.jpg'

                    with open(pic_name, 'wb') as f:
                        f.write(pic.content)
                    self.downloaded += 1

                    if self.downloaded >= self.size:
                        break
                except Exception as err:
                    self.downloaded += 1
                    print(err)
                    continue
                '''
                # sync download End

                # async download Start
                photo_list.append(img_url)

                self.downloaded += 1

                if len(photo_list) >= self.page_size:
                    loop = asyncio.get_event_loop()
                    tasks = [async_save(url, self.directory) for url in photo_list]
                    loop.run_until_complete(asyncio.wait(tasks))
                    photo_list = []

                if self.downloaded >= self.size:
                    loop = asyncio.get_event_loop()
                    tasks = [async_save(url, self.directory) for url in photo_list]
                    loop.run_until_complete(asyncio.wait(tasks))
                    break
                # async download End

        print("Download complete")


# test:
if __name__ == '__main__':
    start = datetime.now()

    butterfly = CfhSpecies(name="金裳凤蝶", directory="Troides aeacus test", page_size= 25, check=False)
    butterfly.get_species_id()
    butterfly.download()

    end = datetime.now()
    print("aiohttp版爬虫花费时间为：")
    print(end - start)