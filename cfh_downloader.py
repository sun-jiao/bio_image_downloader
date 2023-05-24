import json
from abc import ABC
from datetime import datetime
from json import JSONDecodeError

import requests

from base_downloader import BaseDownloader


class CfhDownloader(BaseDownloader, ABC):
    photo_list_key = 'photolist'
    host = 'www.cfh.ac.cn'

    def get_species_id(self):
        id_query_url = 'http://www.cfh.ac.cn/ajaxserver/speciesserv.ashx?action=spsearchzh&keyword=' + self.name

        species_data = requests.get(id_query_url, headers=self.get_header())
        try:
            species_json = json.loads(species_data.text)
        except JSONDecodeError:
            print(species_data.text)
            return

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
            self.id) + "&pagesize=" + str(self.page_size) + "&page=" + str(index + 1)

    def get_image_url(self, json_item):
        return ['http://www.cfh.ac.cn' + \
                str(json_item['thumbnail']).replace('Thumbnail', 'Normal')]


# test:
if __name__ == '__main__':
    start = datetime.now()

    butterfly = CfhDownloader(name="金裳凤蝶", directory="Troides aeacus test", page_size=25, check=False)
    butterfly.download()

    end = datetime.now()
    print("time cost: ")
    print(end - start)
