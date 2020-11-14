import asyncio
import json
import math
import os

import aiofiles
import aiohttp
import requests
from fake_useragent import UserAgent
from abc import abstractmethod,ABCMeta
from urllib.parse import urlsplit

async def fetch(client, url):
    """
    asyncly download image
    ----------
    :param client: see :func:`async_save`
    :param url: see :func:`async_save`
    """
    ua = UserAgent(use_cache_server=False)
    host_url = "{0.netloc}".format(urlsplit(url)) # get host from url automatically using urllib
    async with client.get(url, proxy='http://127.0.0.1:8124', headers={
            'User-Agent': ua.random,
            'Host': host_url, }) as resp:
        return await resp.read()


async def async_save(url, directory):
    """
    call 'fetch' to download image and save it in specified firectory.
    -----------
    :param url: string, url of image to be downloaded
    :param directory: string, path of the directory where image to be downloaded to
    """
    async with aiohttp.ClientSession() as client:
        image = await fetch(client, url)
        split_list = url.split('/')
        filename = split_list[len(split_list)- 1]
        if not filename.lower().endswith('.jpg'):
            filename = filename + '.jpg'
        f = await aiofiles.open(directory + '/' + filename , mode='wb')
        await f.write(image)
        await f.close()

class BaseDownloader(metaclass=ABCMeta):
    """
    abstracted base downloader.
    """

    photo_list_key = ''
    host = ''

    def __init__(self, name, directory, size = 0, page_size = 25, check = None, folder_size = 0):
        """

        :param name: string, species name, scientific or vernacular are both ok (or only one of it in some child class, such as GbifDownloader.
        :param directory: string, path of the directory where image to be downloaded to.
        :param size: int, total amount of images to be downloaded, 0 means as much as the website has.
        :param page_size: int, the number of images in a page,
                            too large size may take a long time to download all images of it, too small size will cause multiple requests.
        :param check: boolean, name check, None: DO NOT check; True: scientific name; False: chinese vernacular name.
        :param folder_size: int, maximum number of pictures in the folder, download will be interrupted after reaching it.
        """
        self.name = name
        self.directory = './download/' + directory
        self.size = size
        self.page_size = page_size
        self.check = check
        self.folder_size = folder_size

        self.downloaded = 0  # amount of images that already downloaded
        self.ua = UserAgent(use_cache_server=False)  # from fake_useragent to generate random Useragent
        self.id = None  # id of the species in this website
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
            tasks = [async_save(url, self.directory) for url in photo_list]
            if len(tasks) > 0:
                loop.run_until_complete(asyncio.wait(tasks))
            # async download End

        print("Download complete")
