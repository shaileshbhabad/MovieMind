"""
build_poster_cache.py
Pre-caches poster paths for popular movies into data/posters_cache.json using concurrent threads.
"""

import os
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

API_KEY = "158402ec29d13b392ed3bff727897e84"
CACHE_FILE = "data/posters_cache.json"


def fetch_one(mid):
    url = f"https://api.themoviedb.org/3/movie/{mid}?api_key={API_KEY}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode())
            return mid, data.get("poster_path")
    except Exception:
        return mid, None


def main():
    movies = pd.read_csv("data/tmdb_5000_movies.csv")
    # Take top 1500 popular movies
    top_mids = movies.sort_values(by="popularity", ascending=False)["id"].head(1500).tolist()

    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    to_fetch = [mid for mid in top_mids if str(mid) not in cache]
    print(f"Fetching posters for {len(to_fetch)} movies concurrently...")

    with ThreadPoolExecutor(max_workers=25) as executor:
        futures = {executor.submit(fetch_one, mid): mid for mid in to_fetch}
        for future in as_completed(futures):
            mid, poster_path = future.result()
            if poster_path:
                cache[str(mid)] = poster_path

    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

    print(f"Done! Cached {len(cache)} posters in {CACHE_FILE}.")


if __name__ == "__main__":
    main()
