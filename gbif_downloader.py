# based on gbif api: https://www.gbif.org/developer/occurrence

import json
import os
from datetime import datetime
from json import JSONDecodeError
from ssl import SSLError

import requests

from base_downloader import BaseDownloader


class GbifDownloader(BaseDownloader):
    photo_list_key = 'results'
    host = 'api.gbif.org'

    first_only = False

    def get_species_id(self):
        id_query_url = 'https://api.gbif.org/v1/species?name=' + self.name

        while True:
            try:
                species_data = requests.get(id_query_url, headers=self.get_header())
                species_json = json.loads(species_data.text)
                break
            except SSLError or ConnectionError or JSONDecodeError:  # Try to catch something more specific
                pass

        if len(species_json) > 0:
            if self.check is not None:
                species_json = species_json[self.photo_list_key]
                for index in range(len(species_json)):
                    if self.check and 'canonicalName' in species_json[index] and 'taxonID' in species_json[index] and \
                            'key' in species_json[index] and (self.name == species_json[index]['canonicalName']) \
                            and (species_json[index]['taxonID'] == 'gbif:' + str(species_json[index]['key'])):

                        self.id = species_json[index]['key']

                        image_list_url = "https://api.gbif.org/v1/occurrence/search?taxonkey=" + str(
                            self.id) + "&limit=1&offset=0&mediaType=StillImage&basisOfRecord=HUMAN_OBSERVATION"
                        image_list = requests.get(image_list_url, headers=self.get_header())
                        server_size = json.loads(image_list.text)['count']

                        if not os.path.exists(self.directory) or \
                                (self.size < server_size and
                                 self.folder_size - len(os.listdir(self.directory)) < server_size):
                            self.first_only = True
                        if self.size <= 0 or self.size >= server_size:
                            self.size = server_size
                        break
                    else:
                        continue

    def get_image_list_url(self, index):
        return "https://api.gbif.org/v1/occurrence/search?taxonkey=" + str(
            self.id) + "&limit=" + str(self.page_size) + "&offset=" + str(
            index * self.page_size) + '&mediaType=StillImage&basisOfRecord=HUMAN_OBSERVATION'
        # the gbif api do not use 'page' parameter, instead, it use the 'offset', results start from the No.(offset + 1) image.

    def get_image_url(self, json_item):
        url_list = []
        media_list = json_item['media']
        for media_item in media_list:
            try:
                new_url = 'https://api.gbif.org/v1/image/unsafe/' + str(media_item['identifier']).\
                    replace('original', 'medium')  # thumb, small, medium, large
                if 'xeno-canto' not in new_url:
                    url_list.append(new_url)
                # with the 'api.gbif.org' prefix to get thumbnail, without it to get original size
                if self.first_only:
                    break
            except:
                pass
        return url_list


# test:
if __name__ == '__main__':
    start = datetime.now()

    butterfly = GbifDownloader(name="Nothocercus bonapartei", directory="Nothocercus_bonapartei", page_size=25, check=True)
    butterfly.download()

    end = datetime.now()
    print("time cost: ")
    print(end - start)
