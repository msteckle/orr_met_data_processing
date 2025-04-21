#!/usr/bin/env python3
"""
download_zenodo_latest.py

1) Start from a known Zenodo record ID (any version).
2) Discover the conceptRecId.
3) Use the search API to find the most recent version ID.
4) Download all files from that version using the fallback URL pattern.
5) Unzip any .zip archives then delete the .zip file.
6) Print debug info at every step.
"""

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile

# ─── CONFIG ───────────────────────────────────────────────────────────────────

INITIAL_RECORD_ID = "14744006"  # one of your existing record IDs
DATA_DIR = "zenodo_archive_data"

API_BASE = "https://zenodo.org/api/records"
RECORD_URL = API_BASE + "/{recid}"

# ─── HELPERS ──────────────────────────────────────────────────────────────────


def fetch_json(url):
    print(f"[INFO] GET {url}")
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def download_file(url, dest_path):
    print(f"[INFO] Downloading {url} → {dest_path}")
    with urllib.request.urlopen(url) as r, open(dest_path, "wb") as f:
        while True:
            chunk = r.read(8192)
            if not chunk:
                break
            f.write(chunk)


# ─── MAIN ─────────────────────────────────────────────────────────────────────


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1) Get metadata for our known version
    initial_meta = fetch_json(RECORD_URL.format(recid=INITIAL_RECORD_ID))
    concept_id = initial_meta.get("conceptrecid") or initial_meta.get("conceptRecid")
    if not concept_id:
        print("[ERROR] could not find conceptRecId", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] conceptRecId = {concept_id}")

    # 2) Find the latest version via search API
    search_url = f"{API_BASE}?q=conceptrecid:{concept_id}&sort=mostrecent&size=1"
    hits = fetch_json(search_url).get("hits", {}).get("hits", [])
    if not hits:
        print(f"[ERROR] no search hits for conceptrecid:{concept_id}", file=sys.stderr)
        sys.exit(1)

    latest_id = hits[0]["id"]
    print(f"[INFO] latest version ID = {latest_id}")

    # 3) Fetch file list for that version
    latest_meta = fetch_json(RECORD_URL.format(recid=latest_id))
    files = latest_meta.get("files", [])
    if not files:
        print("[ERROR] no files listed in latest record", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] found {len(files)} files:")
    for f in files:
        print("   •", f["key"])

    # 4) Download & unpack using fallback URL
    for f in files:
        name = f["key"]
        quoted = urllib.parse.quote(name, safe="")
        dl_url = f"https://zenodo.org/api/records/{latest_id}/files/{quoted}/content"

        out_path = os.path.join(DATA_DIR, name)
        if os.path.exists(out_path):
            print(f"[SKIP] {name} already exists")
        else:
            download_file(dl_url, out_path)

        if name.lower().endswith(".zip"):
            folder = os.path.join(DATA_DIR, name[:-4])
            if os.path.isdir(folder):
                print(f"[SKIP] {folder} already extracted")
            else:
                print(f"[INFO] Extracting {name} → {folder}")
                with zipfile.ZipFile(out_path, "r") as z:
                    z.extractall(folder)
            os.remove(out_path)
            print(f"[INFO] Removed archive {name}")

    print(f"[DONE] All files for version {latest_id} are in '{DATA_DIR}'")


if __name__ == "__main__":
    main()
