"""
Download free sample data from the internet and batch-ingest into the MMDB API.

Usage:
    python3 scripts/download_and_ingest.py

Requirements:
    - FastAPI server running at http://127.0.0.1:8000
    - requests, os
"""

import os
import requests
import time

API_BASE = "http://127.0.0.1:8000/v1/ingestion"
DOWNLOAD_DIR = "sample_data"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# ── Sources ──────────────────────────────────────────────────────────

IMAGES = [
    # Picsum — random real photographs (public, free)
    ("nature_landscape.jpg",    "https://picsum.photos/seed/nature/800/600"),
    ("city_street.jpg",         "https://picsum.photos/seed/city/800/600"),
    ("portrait_person.jpg",     "https://picsum.photos/seed/face/800/600"),
    ("food_plate.jpg",          "https://picsum.photos/seed/food/800/600"),
    ("animal_wildlife.jpg",     "https://picsum.photos/seed/animal/800/600"),
    ("architecture_building.jpg","https://picsum.photos/seed/building/800/600"),
    ("ocean_beach.jpg",         "https://picsum.photos/seed/ocean/800/600"),
    ("mountain_view.jpg",       "https://picsum.photos/seed/mountain/800/600"),
    ("forest_trees.jpg",        "https://picsum.photos/seed/forest/800/600"),
    ("sunset_sky.jpg",          "https://picsum.photos/seed/sunset/800/600"),
    # Text-heavy images (good for OCR testing)
    ("text_quote.jpg",          "https://dummyimage.com/800x400/222/fff.jpg&text=The+only+way+to+do+great+work+is+to+love+what+you+do+-+Steve+Jobs"),
    ("text_code.jpg",           "https://dummyimage.com/800x400/1a1a2e/0ff.jpg&text=def+hello():%0A++++print('Hello+World')"),
    ("text_invoice.jpg",        "https://dummyimage.com/800x400/fff/000.jpg&text=INVOICE+%231042%0ATotal:+$250.00%0ADate:+2025-01-15"),
    ("text_warning.jpg",        "https://dummyimage.com/800x400/ff0/000.jpg&text=WARNING:+Do+not+operate%0Aheavy+machinery"),
    ("text_menu.jpg",           "https://dummyimage.com/800x400/2d2d2d/gold.jpg&text=CAFE+MENU%0AEspresso+$3%0ALatte+$5%0ACappuccino+$4.50"),
]

AUDIO_FILES = [
    # Free public domain sound/speech samples
    ("cantina_band.wav",       "https://www2.cs.uic.edu/~i101/SoundFiles/CantinaBand3.wav"),
    ("star_wars_theme.wav",    "https://www2.cs.uic.edu/~i101/SoundFiles/StarWars3.wav"),
    ("imperial_march.wav",     "https://www2.cs.uic.edu/~i101/SoundFiles/ImperialMarch60.wav"),
    ("preamble_speech.wav",    "https://www2.cs.uic.edu/~i101/SoundFiles/preamble10.wav"),
    ("gettysburg_speech.wav",  "https://www2.cs.uic.edu/~i101/SoundFiles/gettysburg10.wav"),
    ("taunt_speech.wav",       "https://www2.cs.uic.edu/~i101/SoundFiles/taunt.wav"),
]

VIDEO_FILES = [
    # Free sample videos
    ("bunny_clip.mp4",         "https://www.w3schools.com/html/mov_bbb.mp4"),
    ("nature_clip.mp4",        "https://www.w3schools.com/html/movie.mp4"),
    ("test_video.mp4",         "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4"),
]


def download_file(name, url, folder):
    """Download a file if it doesn't already exist."""
    path = os.path.join(folder, name)
    if os.path.exists(path):
        print(f"  ⏭️  Already exists: {name}")
        return path

    print(f"  ⬇️  Downloading: {name} ... ", end="", flush=True)
    try:
        resp = requests.get(url, timeout=30, allow_redirects=True)
        resp.raise_for_status()
        with open(path, "wb") as f:
            f.write(resp.content)
        size_kb = len(resp.content) / 1024
        print(f"✅ ({size_kb:.0f} KB)")
        return path
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None


def ingest_file(file_path, media_type):
    """Send a file to the ingestion API."""
    endpoint = f"{API_BASE}/{media_type}"
    filename = os.path.basename(file_path)

    mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".wav": "audio/wav", ".mp3": "audio/mpeg",
        ".mp4": "video/mp4", ".avi": "video/x-msvideo",
        ".pdf": "application/pdf", ".txt": "text/plain",
    }
    ext = os.path.splitext(filename)[1].lower()
    mime = mime_map.get(ext, "application/octet-stream")

    print(f"  📤 Ingesting: {filename} → /v1/ingestion/{media_type} ... ", end="", flush=True)
    try:
        # Since celery workers process tasks asynchronously, the API will return 200/202 status code almost instantly
        with open(file_path, "rb") as f:
            files = {"file": (filename, f, mime)}
            resp = requests.post(endpoint, files=files, timeout=120)

        if resp.status_code == 200 or resp.status_code == 202:
            data = resp.json()
            print(f"✅ queued task_id={data.get('task_id', '?')[:8]}...")
            return True
        else:
            print(f"❌ {resp.status_code}: {resp.text[:100]}")
            return False
    except Exception as e:
        print(f"❌ {e}")
        return False


def main():
    print("=" * 60)
    print("📦 MMDB Sample Data Downloader & Ingester")
    print("=" * 60)

    # Check API is running (wait up to 120s for models to load)
    print("⏳ Waiting for FastAPI server at http://127.0.0.1:8000 ...")
    server_ready = False
    for i in range(60):  # 60 attempts × 2s = 120s max
        try:
            requests.get("http://127.0.0.1:8000/", timeout=3)
            server_ready = True
            break
        except Exception:
            print(f"   Attempt {i+1}/60 — server not ready yet, retrying in 2s...")
            time.sleep(2)

    if not server_ready:
        print("\n❌ FastAPI server is not running at http://127.0.0.1:8000")
        print("   Start it with: uvicorn src.main:app --reload --host 127.0.0.1 --port 8000")
        return

    print("\n✅ API is running!\n")

    # ── Download Phase ──
    print("📥 DOWNLOADING SAMPLE FILES")
    print("-" * 40)

    downloaded = {"image": [], "audio": [], "video": [], "document": []}

    print("\n🖼️  Images:")
    for name, url in IMAGES:
        path = download_file(name, url, DOWNLOAD_DIR)
        if path:
            downloaded["image"].append(path)

    print("\n🎵 Audio:")
    for name, url in AUDIO_FILES:
        path = download_file(name, url, DOWNLOAD_DIR)
        if path:
            downloaded["audio"].append(path)

    print("\n🎬 Video:")
    for name, url in VIDEO_FILES:
        path = download_file(name, url, DOWNLOAD_DIR)
        if path:
            downloaded["video"].append(path)

    # Also include existing files in project root
    for existing in ["image.png", "sample.wav", "harvard.wav", "sample.mp4", "implementation_plan.md"]:
        if os.path.exists(existing):
            if existing.endswith((".png", ".jpg", ".jpeg")):
                downloaded["image"].append(existing)
            elif existing.endswith((".wav", ".mp3")):
                downloaded["audio"].append(existing)
            elif existing.endswith((".mp4", ".avi")):
                downloaded["video"].append(existing)
            elif existing.endswith((".md", ".txt", ".pdf")):
                if existing.endswith(".md"):
                    txt_path = existing.replace(".md", ".txt")
                    import shutil
                    shutil.copy(existing, txt_path)
                    downloaded["document"].append(txt_path)
                else:
                    downloaded["document"].append(existing)

    total = sum(len(v) for v in downloaded.values())
    print(f"\n📊 Total files ready: {total}")
    print(f"   Images: {len(downloaded['image'])}, Audio: {len(downloaded['audio'])}, Video: {len(downloaded['video'])}, Documents: {len(downloaded['document'])}")

    # ── Ingestion Phase ──
    print("\n📤 INGESTING INTO DATABASE")
    print("-" * 40)

    success = 0
    failed = 0

    for media_type, files in downloaded.items():
        emoji = '🖼️' if media_type == 'image' else '🎵' if media_type == 'audio' else '🎬' if media_type == 'video' else '📄'
        print(f"\n{emoji} {media_type.upper()}:")
        for path in files:
            if ingest_file(path, media_type):
                success += 1
            else:
                failed += 1
            time.sleep(0.5)

    # ── Summary ──
    print("\n" + "=" * 60)
    print(f"✅ Done! Queued {success} files, {failed} failed.")
    print(f"🔍 Try: curl 'http://127.0.0.1:8000/v1/search/?query=nature'")
    print("=" * 60)



if __name__ == "__main__":
    main()
