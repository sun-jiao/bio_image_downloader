import json
from datetime import datetime

import requests

from base_downloader import BaseDownloader


class GbifDownloader(BaseDownloader):
    photo_list_key = 'results'
    host = 'api.gbif.org'

    def get_species_id(self):
        id_query_url = 'https://api.gbif.org/v1/species?name=' + self.name

        species_data = requests.get(id_query_url, headers=self.get_header())
        species_json = json.loads(species_data.text)

        if len(species_json) > 0:
            position_in_json = 0
            if self.check is not None:
                species_json = species_json[self.photo_list_key]
                for index in range(len(species_json)):
                    if self.check & (self.name == species_json[index]['scientificName']) \
                            & (species_json[index]['taxonID'] == 'gbif:' + str(species_json[index]['key'])):
                        position_in_json = index
                        break
                    else:
                        continue

            self.id = species_json[position_in_json]['key']

            if self.size <= 0:
                image_list_url = "https://api.gbif.org/v1/occurrence/search?taxonkey=" + str(self.id) + "&limit=0&offset=0"
                image_list = requests.get(image_list_url, headers=self.get_header())

                self.size = json.loads(image_list.text)['count']

    def get_image_list_url(self, index):
        return "https://api.gbif.org/v1/occurrence/search?taxonkey=" + str(
            self.id) + "&limit=" + str(self.page_size) + "&offset=" + str(index * self.page_size)

    def get_image_url(self, json_item):
        url_list = []
        if json_item['basisOfRecord']=="HUMAN_OBSERVATION":
            json_list = json_item['media']
            for media_item in json_list:
                try:
                    url_list.append('https://api.gbif.org/v1/image/unsafe/' + str(media_item['identifier']))
                except:
                    pass
        return url_list

# test:
if __name__ == '__main__':
    start = datetime.now()

    butterfly = GbifDownloader(name="Hestina nama", directory="Hestina nama test", page_size= 25, check=False)
    butterfly.download()

    end = datetime.now()
    print("aiohttp版爬虫花费时间为：")
    print(end - start)