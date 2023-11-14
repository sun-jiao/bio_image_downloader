import json
from abc import ABC
from datetime import datetime
from json import JSONDecodeError
from ssl import SSLError

import requests

from base_downloader import BaseDownloader


class EbirdDownloader(BaseDownloader, ABC):
    photo_list_key = 'photolist'
    host = 'media.ebird.org'
    current_cursor = None
    resolution = 480

    def get_species_id(self):
        self.page_size = 29

        id_query_url = f"https://media.ebird.org/api/v2/stats/media-count?taxonCode={self.name}&birdOnly=true"

        while True:
            try:
                requests.get("https://ebird.org/", headers=self.get_header())
                species_data = requests.get(id_query_url, headers=self.get_header())
                species_json = json.loads(species_data.text)
                self.size = species_json["photo"]
                break
            except SSLError or ConnectionError or JSONDecodeError:  # Try to catch something more specific
                pass

        return self.name  # use vlookup get codename in excel.

    def get_image_list_url(self, index):
        url = (f"https://media.ebird.org/api/v2/search?taxonCode={self.id}&sort=rating_rank_desc"
               f"&mediaType=photo&birdOnly=true")
        if self.current_cursor is not None:
            url = url + f"&initialCursorMark={self.current_cursor}"
        return url

    def get_photo_list_json(self, json_object):
        return json_object

    def get_image_url(self, json_item):
        return [f"https://cdn.download.ams.birds.cornell.edu/api/v2/asset/{json_item['assetId']}/{self.resolution}"]

    def get_filename(self, url):
        return url


# test:
if __name__ == '__main__':
    start = datetime.now()

    downloader = EbirdDownloader(name="brnpri2", directory="download_test", page_size=25, check=False)
    downloader.download()

    end = datetime.now()
    print("time cost: ")
    print(end - start)
