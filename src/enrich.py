"""Асинхронное обогащение датасета треков данными по артистам.

Базовый Kaggle-датасет содержит audio-features, но не содержит метаданных
артиста (жанры на уровне артиста, число подписчиков, его популярность).
Здесь мы собираем уникальных артистов и обогащаем их через внешний API.

Два источника:
- Spotify Web API (Client Credentials flow) — если заданы ключи в .env.
- MusicBrainz API (без ключа) — fallback, если ключей нет.

Запросы идут асинхронно (aiohttp + asyncio.gather) с ограничением
параллелизма через семафор и обработкой rate-limit (HTTP 429).
"""

from __future__ import annotations

import asyncio
import base64
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)

# Параллелизм и тайминги подобраны под лимиты публичных API
SPOTIFY_CONCURRENCY = 5
MUSICBRAINZ_CONCURRENCY = 1  # MusicBrainz требует не более 1 запроса в секунду
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3

SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_SEARCH_URL = "https://api.spotify.com/v1/search"
MUSICBRAINZ_URL = "https://musicbrainz.org/ws/2/artist"
USER_AGENT = "SpotifyAnalysisProject/1.0 (educational data analysis)"


def primary_artist(artists_field: str) -> str:
    """Берёт первого артиста из строки вида 'A;B;C' — он считается основным."""
    return str(artists_field).split(";")[0].strip()


async def _get_spotify_token(session: aiohttp.ClientSession,
                             client_id: str, client_secret: str) -> str:
    """Получает access-токен по Client Credentials flow."""
    creds = f"{client_id}:{client_secret}".encode()
    headers = {"Authorization": f"Basic {base64.b64encode(creds).decode()}"}
    data = {"grant_type": "client_credentials"}
    async with session.post(SPOTIFY_TOKEN_URL, headers=headers, data=data) as resp:
        resp.raise_for_status()
        payload = await resp.json()
        return payload["access_token"]


async def _fetch_json(session: aiohttp.ClientSession, url: str,
                      headers: dict, params: dict) -> dict | None:
    """Один GET-запрос с ретраями на 429 и устойчивостью к таймаутам."""
    for attempt in range(MAX_RETRIES):
        try:
            async with session.get(url, headers=headers, params=params) as resp:
                if resp.status == 429:
                    retry_after = int(resp.headers.get("Retry-After", "2"))
                    logger.warning("429 rate limit, ждём %s c", retry_after)
                    await asyncio.sleep(retry_after)
                    continue
                if resp.status != 200:
                    return None
                return await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.warning("Сетевая ошибка (%s), попытка %d", exc, attempt + 1)
            await asyncio.sleep(1 + attempt)
    return None


async def _enrich_one_spotify(session: aiohttp.ClientSession, sem: asyncio.Semaphore,
                              token: str, artist: str) -> dict:
    """Ищет артиста в Spotify и возвращает его метаданные."""
    headers = {"Authorization": f"Bearer {token}"}
    params = {"q": artist, "type": "artist", "limit": 1}
    async with sem:
        data = await _fetch_json(session, SPOTIFY_SEARCH_URL, headers, params)
    items = (data or {}).get("artists", {}).get("items", [])
    if not items:
        return {"artist": artist, "artist_genres": None,
                "artist_followers": None, "artist_popularity": None}
    top = items[0]
    return {
        "artist": artist,
        "artist_genres": ", ".join(top.get("genres", [])) or None,
        "artist_followers": top.get("followers", {}).get("total"),
        "artist_popularity": top.get("popularity"),
    }


async def _enrich_one_musicbrainz(session: aiohttp.ClientSession,
                                  sem: asyncio.Semaphore, artist: str) -> dict:
    """Fallback: тип/страна/теги артиста из MusicBrainz (без ключа)."""
    headers = {"User-Agent": USER_AGENT}
    params = {"query": f'artist:"{artist}"', "fmt": "json", "limit": 1}
    async with sem:
        await asyncio.sleep(1.0)  # уважаем лимит 1 req/sec
        data = await _fetch_json(session, MUSICBRAINZ_URL, headers, params)
    items = (data or {}).get("artists", [])
    if not items:
        return {"artist": artist, "artist_type": None,
                "artist_country": None, "artist_tags": None}
    top = items[0]
    tags = ", ".join(t["name"] for t in top.get("tags", [])) or None
    return {
        "artist": artist,
        "artist_type": top.get("type"),
        "artist_country": top.get("country"),
        "artist_tags": tags,
    }


async def _gather_spotify(artists: list[str], client_id: str,
                          client_secret: str) -> list[dict]:
    sem = asyncio.Semaphore(SPOTIFY_CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        token = await _get_spotify_token(session, client_id, client_secret)
        tasks = [_enrich_one_spotify(session, sem, token, a) for a in artists]
        return await asyncio.gather(*tasks)


async def _gather_musicbrainz(artists: list[str]) -> list[dict]:
    sem = asyncio.Semaphore(MUSICBRAINZ_CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [_enrich_one_musicbrainz(session, sem, a) for a in artists]
        return await asyncio.gather(*tasks)


def _run_async(coro):
    """Запускает корутину и из обычного скрипта, и из Jupyter.

    В Jupyter event loop уже крутится, поэтому asyncio.run() падает —
    в этом случае исполняем корутину в отдельном потоке со своим loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def enrich_artists(artists: list[str], client_id: str | None = None,
                   client_secret: str | None = None,
                   cache_path: str | Path | None = None) -> pd.DataFrame:
    """Обогащает список артистов. Кэширует результат, чтобы не дёргать API повторно.

    Если заданы ключи Spotify — идём через Spotify, иначе через MusicBrainz.
    Возвращает DataFrame с колонкой 'artist' для последующего merge.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            logger.info("Беру обогащение из кэша: %s", cache_path)
            return pd.read_csv(cache_path)

    unique_artists = sorted(set(artists))
    logger.info("Обогащаю %d уникальных артистов", len(unique_artists))

    if client_id and client_secret:
        records = _run_async(_gather_spotify(unique_artists, client_id, client_secret))
    else:
        logger.warning("Ключи Spotify не заданы — fallback на MusicBrainz")
        records = _run_async(_gather_musicbrainz(unique_artists))

    result = pd.DataFrame(records)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(cache_path, index=False)
        logger.info("Кэш сохранён: %s", cache_path)
    return result
