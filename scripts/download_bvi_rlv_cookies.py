#!/usr/bin/env python3
"""
download_bvi_rlv_cookies.py — скачать BVI-RLV через сессионные cookies.

Использование:
  1. Вставь cookies из браузера в COOKIES ниже
  2. Запусти: python scripts/download_bvi_rlv_cookies.py
  3. Данные сохранятся в data/BVI-RLV/

Cookies действуют несколько часов. Если истекут — обнови их из браузера.

Скрипт автоматически пропускает уже скачанные папки (по числу файлов).
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
import urllib.parse

# ──────────────────────────────────────────────
# Настройки
# ──────────────────────────────────────────────

BASE_URL    = "https://ieee-dataport.org"
DATASET_ID  = "11869"
PREFIX_BASE = f"open/420/{DATASET_ID}/BVI-Lowlight-videos_zip/BVI-Lowlight-videos"
OUTPUT_DIR  = Path(__file__).parent.parent / "data" / "BVI-RLV"

# Задержка между запросами (секунды)
DELAY_MIN = 0.5
DELAY_MAX = 1.5

# Все 20 сцен (реальные имена с сервера, S12–S21 исправлены)
SCENES = [
    "S02_animals1",   "S03_animals2",   "S04_colour_sticks",
    "S05_bunnies",    "S06_lego",       "S07_hats",
    "S08_soft_toys",  "S09_kitchen",    "S10_messy_toy",
    "S11_gift_wrap",  "S12_woods",      "S13_wires",
    "S14_colour_balls", "S15_flowers",  "S16_faces1",
    "S17_faces2",     "S18_books1",     "S19_books2",
    "S20_books3",     "S21_mario1",
]

LIGHT_LEVELS = ["low_light_10", "low_light_20", "normal_light_10", "normal_light_20"]

# ──────────────────────────────────────────────
# ВСТАВЬ COOKIES СЮДА (из браузера, строка -b '...')


# ──────────────────────────────────────────────

COOKIES = (
    "SSESSf80f5dcc65a99d0290714d0bd9d60a3a=MO8C%2CWveuOh7Xc9vMDJbo52Joc07n3m2CQ3mZUcLcJproG4g; "
    "osano_consentmanager_uuid=6192cc01-7d49-4a20-b484-6f42e573c43c; "
    "osano_consentmanager=loj1swStXZPGV6ML083jWZMC1bsp9fS8T4uc6LE_UsCLHu4LVkvcXnfmjce45vVX7xYT3lov0_SJOgZFpSS_93KeQo7Fww7HJu6m-BcImtN4PRVL-pW-DEeEdXtGI8OEY1hgZm5ZoBt6J6BZah-t36O4Ekhx8nYwPAbbdIs4ctItNFzmy8EA4kcqdQmE_HsJ612hcPcapezCdhIDOwGiO6Rym6OxTLFaH-xbq_FqT1sU4YjPRQ36bZVLjw2TQ9YXqj9NixYhUlMhjNhigqXezgPwbClslXvgJTjBPl5ltDm_AN7fhTjqJQTnOxm80vzM; "
    "SimpleSAMLSessionID=6adae791be5a4abe3aaa94d7951df72e; "
    "refresh-get-customer=1781291985"
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
# Проверка: папка уже скачана?
# ──────────────────────────────────────────────

def is_level_complete(scene: str, level: str, expected: int) -> bool:
    """True если в папке уже >= expected файлов."""
    level_dir = OUTPUT_DIR / scene / level
    if not level_dir.exists():
        return False
    existing = sum(1 for f in level_dir.iterdir() if f.suffix == ".png")
    return existing >= expected


# ──────────────────────────────────────────────
# Получить список файлов в папке
# ──────────────────────────────────────────────

def list_directory(prefix: str) -> list[str]:
    import re
    encoded = urllib.parse.quote(prefix, safe="")
    url = f"{BASE_URL}/dataport/load-directory/{DATASET_ID}?prefix={encoded}"
    resp = make_request(url)
    if not resp:
        return []

    try:
        data = json.loads(resp)
        html = data.get("html", "")
    except json.JSONDecodeError:
        html = resp

    if not html:
        return []

    if "login" in html.lower() and "dataport" not in html.lower():
        print("  ⚠️  Сессия истекла! Обнови cookies в скрипте.")
        sys.exit(1)

    matches = re.findall(r'data-download-url="(/dataport/s3-download-url/[^"]+)"', html)
    return [f"{BASE_URL}{m}" for m in matches]


# ──────────────────────────────────────────────
# Скачать один файл
# ──────────────────────────────────────────────

def get_filename_from_url(download_url: str) -> str:
    parsed = urllib.parse.urlparse(download_url)
    key_b64 = urllib.parse.parse_qs(parsed.query).get("key", [""])[0]
    key_b64 = urllib.parse.unquote(key_b64)
    padding = 4 - len(key_b64) % 4
    if padding != 4:
        key_b64 += "=" * padding
    try:
        s3_path = base64.b64decode(key_b64).decode()
        return Path(s3_path).name
    except Exception:
        return "unknown.png"


def resolve_download_url(dataport_url: str) -> str | None:
    resp = make_request(dataport_url)
    if not resp:
        return None
    try:
        data = json.loads(resp)
        url = data.get("download_url") or data.get("url")
        if url:
            return url.replace("\\/", "/")
    except Exception:
        pass
    return resp.strip() if resp.strip().startswith("http") else None


def download_file(dataport_url: str, local_path: Path) -> bool:
    if local_path.exists() and local_path.stat().st_size > 1000:
        return True  # уже скачан

    direct_url = resolve_download_url(dataport_url)
    sleep()
    if not direct_url:
        return False

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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nBVI-RLV Download")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Scenes: {len(SCENES)}\n")

    total_files = 0
    total_done  = 0
    total_skip  = 0
    total_fail  = 0

    for scene_idx, scene in enumerate(SCENES, 1):
        print(f"[{scene_idx}/{len(SCENES)}] {scene}")

        for level in LIGHT_LEVELS:
            prefix = f"{PREFIX_BASE}/{scene}/{level}/"
            level_dir = OUTPUT_DIR / scene / level

            # ── Быстрая проверка: получить список файлов один раз ──
            print(f"  {level} — получаем список...", end=" ", flush=True)
            download_urls = list_directory(prefix)
            sleep()

            if not download_urls:
                print("пусто, пропускаем")
                continue

            expected = len(download_urls)

            # ── Если папка уже полная — пропустить без единого лишнего запроса ──
            if is_level_complete(scene, level, expected):
                existing = sum(1 for f in level_dir.iterdir() if f.suffix == ".png")
                print(f"✓ уже скачано ({existing}/{expected}), пропускаем")
                total_skip += expected
                total_files += expected
                continue

            existing_count = sum(1 for f in level_dir.iterdir() if f.suffix == ".png") if level_dir.exists() else 0
            print(f"{expected} файлов (уже есть: {existing_count})")

            for dl_url in download_urls:
                fname = get_filename_from_url(dl_url)
                local = level_dir / fname
                total_files += 1

                ok = download_file(dl_url, local)
                if ok:
                    total_done += 1
                    if local.stat().st_size > 1000 and not (total_done % 50):
                        # прогресс каждые 50 файлов
                        mb = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*.png") if f.exists()) // 1024 // 1024
                        print(f"    ... {total_done} скачано / {total_fail} ошибок / {mb} MB")
                else:
                    total_fail += 1
                    print(f"    FAIL: {fname}")

                sleep()

        mb_scene = sum(
            f.stat().st_size for f in (OUTPUT_DIR / scene).rglob("*.png") if f.exists()
        ) // 1024 // 1024 if (OUTPUT_DIR / scene).exists() else 0
        print(f"  ✓ {scene} — {mb_scene} MB\n")

    print("─" * 50)
    total_mb = sum(f.stat().st_size for f in OUTPUT_DIR.rglob("*.png") if f.exists()) // 1024 // 1024
    print(f"Скачано  : {total_done} новых файлов")
    print(f"Пропущено: {total_skip} (уже были)")
    print(f"Ошибок   : {total_fail}")
    print(f"Размер   : {total_mb} MB")
    print("─" * 50)


if __name__ == "__main__":
    main()
