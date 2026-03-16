# -*- coding: utf-8 -*-
import os
import asyncio
import aiohttp
import settings

SAVE_DIR = settings.BASE_DIR / "oss" / "media" / "groups" / "default" / "gallery"
os.makedirs(SAVE_DIR, exist_ok=True)

TOTAL = 10000
CONCURRENT = 50

URL = "https://picsum.photos/800/800"


async def download_image(session, sem, idx):
    async with sem:
        try:

            filename = os.path.join(SAVE_DIR, f"img_{idx:05d}.jpg")

            if os.path.exists(filename):
                return

            async with session.get(URL) as resp:

                if resp.status != 200:
                    return

                content = await resp.read()

                with open(filename, "wb") as f:
                    f.write(content)

                print("✅", filename)

        except Exception as e:
            print("❌", idx, e)


async def main():

    sem = asyncio.Semaphore(CONCURRENT)

    timeout = aiohttp.ClientTimeout(total=30)

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:

        tasks = []

        for i in range(TOTAL):
            tasks.append(download_image(session, sem, i))

        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())