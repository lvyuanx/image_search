# -*- coding: utf-8 -*-
import os
import asyncio
import aiohttp
import settings
from urllib.parse import urlparse

SAVE_DIR = settings.BASE_DIR / "oss" / "media" / "groups" / "default" / "gallery"
os.makedirs(SAVE_DIR, exist_ok=True)

MAX_IMAGES = 200
CONCURRENT = 10

OPENVERSE_API = "https://api.openverse.engineering/v1/images"
SEARCH_TERMS = [
    "clothing logo",
    "fashion brand logo",
    "apparel logo",
]
PER_PAGE = 50


def _get_ext_from_url(url: str) -> str:
    path = urlparse(url).path
    ext = os.path.splitext(path)[1].lower()
    if ext in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
        return ext
    return ".jpg"


async def fetch_openverse_image_urls(session, term, max_images, retries=3):
    urls = []
    page = 1

    while len(urls) < max_images:
        params = {
            "q": term,
            "page_size": PER_PAGE,
            "page": page,
        }

        data = None
        for attempt in range(retries):
            try:
                async with session.get(OPENVERSE_API, params=params) as resp:
                    if resp.status != 200:
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue
                    data = await resp.json()
                break
            except (asyncio.TimeoutError, aiohttp.ClientError):
                await asyncio.sleep(0.5 * (attempt + 1))

        if not data:
            break

        results = data.get("results", [])
        if not results:
            break

        for item in results:
            url = item.get("url")
            if url:
                urls.append(url)
                if len(urls) >= max_images:
                    break

        page += 1

    return urls


async def download_image(session, sem, url, idx):
    async with sem:
        try:
            ext = _get_ext_from_url(url)
            filename = os.path.join(SAVE_DIR, f"logo_{idx:05d}{ext}")

            if os.path.exists(filename):
                return

            async with session.get(url) as resp:
                if resp.status != 200:
                    return
                content = await resp.read()

            with open(filename, "wb") as f:
                f.write(content)

            print("downloaded", filename)

        except Exception as e:
            print("download failed", idx, e)


async def main():
    sem = asyncio.Semaphore(CONCURRENT)
    timeout = aiohttp.ClientTimeout(total=60, connect=15, sock_read=30)
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        urls = []
        for term in SEARCH_TERMS:
            if len(urls) >= MAX_IMAGES:
                break
            remain = MAX_IMAGES - len(urls)
            urls.extend(await fetch_openverse_image_urls(session, term, remain))

        tasks = []
        for i, url in enumerate(urls):
            tasks.append(download_image(session, sem, url, i))

        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
