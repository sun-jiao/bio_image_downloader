import aiohttp
import asyncio
from datetime import datetime

from fake_useragent import UserAgent

async def fetch(client):
    ua = UserAgent(use_cache_server=False)
    async with client.get('https://www.checkip.org/', proxy='http://127.0.0.1:8124', headers={
        'User-Agent': ua.random,
        'Host': 'www.checkip.org', }) as resp:
        assert resp.status == 200
        return await resp.text()


async def main():
    async with aiohttp.ClientSession() as client:
        html = await fetch(client)
        print(html)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()

    tasks = []
    for i in range(2):
        task = loop.create_task(main())
        tasks.append(task)

    start = datetime.now()

    loop.run_until_complete(main())

    end = datetime.now()

    print("aiohttp版爬虫花费时间为：")
    print(end - start)