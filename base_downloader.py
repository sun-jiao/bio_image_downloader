import asyncio
import json
import math
import os

import aiofiles
import aiohttp
import requests
from fake_useragent import UserAgent
from abc import abstractmethod,ABCMeta


async def fetch(client, url, header):
    ua = UserAgent(use_cache_server=False)
    async with client.get(url, proxy='http://127.0.0.1:8124', headers=header) as resp:
        return await resp.read()


async def async_save(url, directory, header):
    async with aiohttp.ClientSession() as client:
        image = await fetch(client, url, header)
        split_list = url.split('/')
        filename = split_list[len(split_list)- 1]
        if not filename.lower().endswith('.jpg'):
            filename = filename + '.jpg'
        f = await aiofiles.open(directory + '/' + filename , mode='wb')
        await f.write(image)
        await f.close()

class BaseDownloader(metaclass=ABCMeta):
    photo_list_key = ''
    host = ''

    def __init__(self, name, directory, size = 0, page_size = 25, check = None, folder_size = 0):
        self.name = name # species name, scientific or vernacular are both ok
        self.directory = './download/' + directory
        self.size = size
        self.downloaded = 0
        self.ua = UserAgent(use_cache_server=False)
        self.id = None
        self.page_size = page_size
        self.check = check # name check, None: DO NOT check; True: scientific name; False: chinese vernacular name
        self.folder_size = folder_size

        self.get_species_id()

    @abstractmethod
    def get_species_id(self):pass

    @abstractmethod
    def get_image_list_url(self, index):pass

    @abstractmethod
    def get_image_url(self, json_item):pass

    def get_header(self):
        return {
            'User-Agent': self.ua.random,
            'Host': self.host, }

    def download(self):
        if (self.id is None) | (self.size <= 0):
            print("Nothing to download for " + self.name + " to " + self.directory)
            return
        else:
            print("Start downloading for " + self.name + " to " + self.directory)

        for index in range(math.ceil(self.size/self.page_size)):
            image_list_url = self.get_image_list_url(index)
            image_list = requests.get(image_list_url, headers=self.get_header())

            try:
                data = json.loads(image_list.text)[self.photo_list_key]
            except KeyError:
                break

            # download links in this page.
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)

            photo_list = []

            if 0 < self.folder_size < len(os.listdir(self.directory)):
                print('folder size meet,', end=' ')
                break

            for jndex in range(self.page_size):
                img_url = self.get_image_url(data[jndex])
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
                photo_list.extend(img_url)

                self.downloaded += 1

                if self.downloaded >= self.size:
                    break

            loop = asyncio.get_event_loop()
            tasks = [async_save(url, self.directory, self.get_header()) for url in photo_list]
            if len(tasks) > 0:
                loop.run_until_complete(asyncio.wait(tasks))
            # async download End

        print("Download complete")
