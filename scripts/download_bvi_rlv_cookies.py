#!/usr/bin/env python3
"""
download_bvi_rlv_cookies.py — скачать BVI-RLV через сессионные cookies.

Использование:
  1. Вставь cookies из браузера в COOKIES ниже (уже заполнено)
  2. Запусти: python scripts/download_bvi_rlv_cookies.py
  3. Данные сохранятся в data/BVI-RLV/

Cookies действуют несколько часов. Если истекут — обнови их из браузера.
"""

import base64
import json
import time
import sys
import os
import random
from pathlib import Path
import urllib.request
import urllib.error

# ──────────────────────────────────────────────
# Настройки
# ──────────────────────────────────────────────

BASE_URL    = "https://ieee-dataport.org"
DATASET_ID  = "11869"
PREFIX_BASE = f"open/420/{DATASET_ID}/BVI-Lowlight-videos_zip/BVI-Lowlight-videos"
OUTPUT_DIR  = Path(__file__).parent.parent / "data" / "BVI-RLV"

# Задержка между запросами (секунды) — не спамим сервер
DELAY_MIN = 0.5
DELAY_MAX = 1.5

# 20 сцен для скачивания
SCENES = [
    "S02_animals1",   "S03_animals2",   "S04_colour_sticks",
    "S05_bunnies",    "S06_lego",       "S07_hats",
    "S08_soft_toys",  "S09_kitchen",    "S10_messy_toy",
    "S11_gift_wrap",  "S12_toys",       "S13_books",
    "S14_plants",     "S15_desk",       "S16_food",
    "S17_sport",      "S18_outdoor1",   "S19_outdoor2",
    "S20_street",     "S21_nature",
]

LIGHT_LEVELS = ["low_light_10", "low_light_20", "normal_light_10", "normal_light_20"]

# ──────────────────────────────────────────────
# ВСТАВЬ COOKIES СЮДА (из браузера, строка -b '...')
# ──────────────────────────────────────────────

COOKIES = (
    "simplesamlphp_auth_returnto=https://ieee-dataport.org/open-access/bvi-lowlight-fully-registered-datasets-low-light-image-and-video-enhancement; "
    "SimpleSAMLSessionID=67e91362c2685e8955dac99cb3999f38; "
    "SimpleSAMLAuthToken=_d8a910f5c01dc70d65a48eb9c4b5dfedd1fb830c99; "
    "SSESSf80f5dcc65a99d0290714d0bd9d60a3a=MO8C%2CWveuOh7Xc9vMDJbo52Joc07n3m2CQ3mZUcLcJproG4g; "
    "osano_consentmanager_uuid=6192cc01-7d49-4a20-b484-6f42e573c43c"
)

HEADERS = {
    "accept":           "*/*",
    "accept-language":  "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "referer":          f"{BASE_URL}/open-access/bvi-lowlight-fully-registered-datasets-low-light-image-and-video-enhancement",
    "user-agent":       "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36",
    "x-requested-with": "XMLHttpRequest",
    "cookie":           COOKIES,
}


# ──────────────────────────────────────────────
# HTTP helpers
# ──────────────────────────────────────────────

def make_request(url: str, binary=False):
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read() if binary else resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {url}")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def sleep():
    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))


# ──────────────────────────────────────────────
# Получить список файлов в папке
# ──────────────────────────────────────────────

def list_directory(prefix: str) -> list[str]:
    """
    Получить список download URL через load-directory API.
    Ответ — HTML с data-download-url атрибутами вида:
      /dataport/s3-download-url/11869?key=BASE64==
    Возвращаем полные URL для скачивания.
    """
    import re
    encoded = urllib.parse.quote(prefix, safe="")
    url = f"{BASE_URL}/dataport/load-directory/{DATASET_ID}?prefix={encoded}"
    resp = make_request(url)
    if not resp:
        return []

    if "login" in resp.lower() and "SimpleSAML" not in resp:
        print("  ⚠️  Сессия истекла! Обнови cookies в скрипте.")
        sys.exit(1)

    # Парсим data-download-url из HTML
    # Пример: data-download-url="\/dataport\/s3-download-url\/11869?key=BASE64"
    pattern = r'data-download-url=\\"(\/dataport\/s3-download-url\/[^\\]+)\\"'
    matches = re.findall(pattern, resp)

    # Unescape unicode и слеши
    urls = []
    for m in matches:
        clean = m.replace("\\/", "/")
        urls.append(f"{BASE_URL}{clean}")

    return urls


# ──────────────────────────────────────────────
# Скачать один файл
# ──────────────────────────────────────────────

import urllib.parse

def get_filename_from_url(download_url: str) -> str:
    """Извлечь имя файла из base64 ключа в URL."""
    import urllib.parse as up
    parsed = up.urlparse(download_url)
    key_b64 = up.parse_qs(parsed.query).get("key", [""])[0]
    # base64 может быть URL-encoded
    key_b64 = up.unquote(key_b64)
    # добавляем padding если нужно
    padding = 4 - len(key_b64) % 4
    if padding != 4:
        key_b64 += "=" * padding
    try:
        s3_path = base64.b64decode(key_b64).decode()
        return Path(s3_path).name  # например "00001.png"
    except Exception:
        return "unknown.png"


def resolve_download_url(dataport_url: str) -> str | None:
    """Получить прямой S3 URL из /dataport/s3-download-url/..."""
    resp = make_request(dataport_url)
    if not resp:
        return None
    # Ответ может быть JSON {"url": "https://..."} или просто строка URL
    try:
        data = json.loads(resp)
        return data.get("url") or data.get("download_url")
    except Exception:
        return resp.strip() if resp.strip().startswith("http") else None


def download_file(dataport_url: str, local_path: Path) -> bool:
    """Скачать файл: dataport_url → resolve → download binary."""
    if local_path.exists() and local_path.stat().st_size > 1000:
        return True  # уже скачан

    # Шаг 1: получить прямой S3 URL
    direct_url = resolve_download_url(dataport_url)
    sleep()
    if not direct_url:
        return False

    # Шаг 2: скачать бинарные данные напрямую с S3
    try:
        req = urllib.request.Request(direct_url, headers={"user-agent": HEADERS["user-agent"]})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
    except Exception as e:
        print(f"    S3 error: {e}")
        return False

    if len(data) < 1000:
        return False

    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(data)
    return True


# ──────────────────────────────────────────────
# Основной цикл
# ──────────────────────────────────────────────

def main():
    import urllib.parse
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nBVI-RLV Download")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Scenes: {len(SCENES)}\n")

    total_files = 0
    total_done  = 0
    total_fail  = 0

    for scene_idx, scene in enumerate(SCENES, 1):
        print(f"[{scene_idx}/{len(SCENES)}] {scene}")

        for level in LIGHT_LEVELS:
            prefix = f"{PREFIX_BASE}/{scene}/{level}/"
            print(f"  {level} — получаем список...", end=" ", flush=True)

            download_urls = list_directory(prefix)
            sleep()

            if not download_urls:
                print("пусто, пропускаем")
                continue

            print(f"{len(download_urls)} файлов")

            for dl_url in download_urls:
                fname    = get_filename_from_url(dl_url)
                local    = OUTPUT_DIR / scene / level / fname
                total_files += 1

                ok = download_file(dl_url, local)
                if ok:
                    total_done += 1
                else:
                    total_fail += 1
                    print(f"    FAIL: {fname}")

                if total_files % 100 == 0:
                    mb = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*.png") if f.exists()) // 1024 // 1024
                    print(f"  ... {total_done} скачано / {total_fail} ошибок / {mb} MB")

                sleep()

        mb_scene = sum(
            f.stat().st_size for f in (OUTPUT_DIR / scene).rglob("*.png") if f.exists()
        ) // 1024 // 1024
        print(f"  ✓ {scene} — {mb_scene} MB\n")

    print("─" * 50)
    total_mb = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*.png") if f.exists()) // 1024 // 1024
    print(f"Готово : {total_done}/{total_files} файлов")
    print(f"Ошибок : {total_fail}")
    print(f"Размер : {total_mb} MB")
    print("─" * 50)


if __name__ == "__main__":
    main()
