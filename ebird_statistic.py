import csv
import json

import requests
from fake_useragent import UserAgent
from json import JSONDecodeError
from ssl import SSLError

filename = 'IOC 13.1.csv'

key = 'jfekjedvescr'

with open(filename, encoding='utf-8') as file:
    new_csv_file = open('bird_new_csv.csv', 'a', encoding='utf-8')
    w_csv = csv.writer(new_csv_file)

    f_csv = csv.reader(file)
    for row in f_csv:

        code = ''

        print('working on', row[3])

        header = {
            'User-Agent': UserAgent().random,
            'Origin': 'https://ebird.org',
            'Referer': 'https://ebird.org/'
        }

        while True:
            try:
                url = f'https://api.ebird.org/v2/ref/taxon/find?key={key}&q={row[3]}'
                species_data = requests.get(url, headers=header)
                species_json = json.loads(species_data.text)
                if len(species_json) > 0:
                    code = species_json[0]['code']
                else:
                    code = 'unavailable'
                break
            except SSLError or ConnectionError or JSONDecodeError:  # Try to catch something more specific
                pass

        # audio = 0
        # photo = 0
        # video = 0
        #
        # header = {
        #     'User-Agent': UserAgent().random,
        #     'Authority':  'media.ebird.org',
        #     'Referer':  f'https://media.ebird.org/catalog?taxonCode={code}&sort=rating_rank_desc&mediaType=photo',
        #     'Cookie': '_gcl_au=1.1.688184762.1684166646; hubspotutk=dcfb5e8ff7e2ba292163da8cda777ae3; i18n_redirected=en; _ga=GA1.3.590654988.1684166646; _ga_YT7Y2S4MBX=GS1.1.1687392631.4.1.1687392691.0.0.0; _ga=GA1.2.1658410509.1685328262; __hstc=60209138.dcfb5e8ff7e2ba292163da8cda777ae3.1684166658219.1688646752081.1688814632952.29; _ga_QR4NVXZ8BM=GS1.1.1688823164.17.0.1688823164.60.0.0; _ga_4RP6YTYH7F=GS1.1.1688823164.33.0.1688823164.0.0.0; _0ad1c=http://10.0.85.16:8080; ml-search-session=eyJ1c2VyIjp7InVzZXJJZCI6IlVTRVIyMDI2ODIyIiwidXNlcm5hbWUiOiJKaWFvU3VuIiwiZmlyc3ROYW1lIjoiSmlhbyIsImxhc3ROYW1lIjoiU3VuIiwiZnVsbE5hbWUiOiJKaWFvIFN1biIsInJvbGVzIjpbXSwicHJlZnMiOnsiUFJPRklMRV9WSVNJVFNfT1BUX0lOIjoidHJ1ZSIsIlBSSVZBQ1lfUE9MSUNZX0FDQ0VQVEVEIjoidHJ1ZSIsIlVTRV8yNEgiOiJ0cnVlIiwiUFJPRklMRV9PUFRfSU4iOiJ0cnVlIiwiU0hPV19TVUJTUEVDSUVTIjoidHJ1ZSIsIlNNQVJUX0NIRUNLTElTVCI6InRydWUiLCJESVNQTEFZX05BTUVfUFJFRiI6Im4iLCJWSVNJVFNfT1BUX09VVCI6ImZhbHNlIiwiRElTUExBWV9DT01NT05fTkFNRSI6InRydWUiLCJESVNQTEFZX1NDSUVOVElGSUNfTkFNRSI6InRydWUiLCJQUk9GSUxFX1JFR0lPTiI6IkNOIiwiU0hPV19DT01NRU5UUyI6InRydWUiLCJDT01NT05fTkFNRV9MT0NBTEUiOiJ6aF9TSU0iLCJHTUFQX1RZUEUiOiJ0ZXJyYWluIiwiQUxFUlRTX09QVF9PVVQiOiJmYWxzZSIsIkVNQUlMX0NTIjoidHJ1ZSIsIlRPUDEwMF9PUFRfT1VUIjoiZmFsc2UiLCJESVNUX1VOSVQiOiJrbSJ9fX0=; ml-search-session.sig=iUU6p-8v1TgNkhSjcr6AaPVMLPk'
        # }
        #
        # while True:
        #     try:
        #         url = f'https://media.ebird.org/api/v2/stats/media-count?taxonCode={code}&birdOnly=true'
        #         media_data = requests.get(url)
        #         media_json = json.loads(media_data.text)
        #         audio = media_json.get('audio')
        #         photo = media_json.get('photo')
        #         video = media_json.get('video')
        #
        #         break
        #     except SSLError or ConnectionError or JSONDecodeError:  # Try to catch something more specific
        #         pass

        row.append(code)  # , audio, photo, video])

        print(code)  # , audio, photo, video)

        w_csv.writerow(row)
