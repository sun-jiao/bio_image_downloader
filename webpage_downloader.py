import os
import uuid

import requests
from bs4 import BeautifulSoup


def download_images(url, save_directory):
    # 创建保存图片的目录
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # 发送GET请求获取网页内容
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    download_by_soup(soup, save_directory)

    # next_url_element = soup.find(id="mmComponent_images_2")
    # if next_url_element is not None:
    #     next_url = 'https://www.bing.com/' + next_url_element.get("data-nextUrl")
    #
    #     response = requests.get(next_url)
    #     soup = BeautifulSoup(response.content, 'html.parser')
    #
    #     download_by_soup(soup, save_directory)


def download_by_soup(soup, save_directory):
    # 找到所有的img标签
    img_tags = soup.find_all('img')
    # 下载并保存图片
    for img in img_tags:
        img_url = img.get('src')
        if img_url:
            # 获取图片文件名
            filename = os.path.join(save_directory, str(uuid.uuid1()))
            try:
                # 发送GET请求下载图片
                image_data = requests.get(img_url).content

                # 保存图片文件
                with open(filename, 'wb') as f:
                    f.write(image_data)
                print(f'Saved image: {filename}')
            except requests.exceptions.RequestException as e:
                print(f'Failed to download image: {img_url}')
                print(str(e))


def bing_download(name, directory, base_directory='./data/train/'):
    name = name.replace(' ', '+')
    save_directory = os.path.join(base_directory, directory)
    url = f'https://www.bing.com/images/search?q={name}&form=QBIR&first=1'
    download_images(url, save_directory)


if __name__ == '__main__':
    bing_download('Bambusicola thoracicus', 'Bambusicola thoracicus', 'images')
