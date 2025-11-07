# -*- coding: utf-8 -*-
import os
import uuid
import requests
from ddgs import DDGS
from concurrent.futures import ThreadPoolExecutor, as_completed

SAVE_DIR = "gallery"
os.makedirs(SAVE_DIR, exist_ok=True)

def download_image(url, idx):
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            filename = os.path.join(SAVE_DIR, f"player_{idx:04d}.jpg")
            with open(filename, "wb") as f:
                f.write(resp.content)
            print(f"✅ {filename}")
    except Exception as e:
        print(f"❌ {url}: {e}")

def main():
    results = []
    with DDGS() as ddgs:
        # 分页抓取，每次抓50张，共20页
        for page in range(1, 21):
            query = "NBA basketball player portrait"
            try:
                # ddgs.images(query, region, safesearch, max_results)
                for r in ddgs.images(
                    query=query,
                    region="wt-wt",
                    safesearch="Off",
                    max_results=50
                ):
                    results.append(r["image"])
            except Exception as e:
                print(f"[WARN] 第 {page} 页抓取失败: {e}")
            print(f"已获取 {len(results)} 张图片")

    print(f"共找到 {len(results)} 张图片，开始下载...")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(download_image, url, i) for i, url in enumerate(results)]
        for _ in as_completed(futures):
            pass

    print("下载完成 ✅")

if __name__ == "__main__":
    main()
