#!/usr/bin/env bash
set -euo pipefail

INITIAL_RECORD_ID="14744006"
DATA_DIR="data"
API="https://zenodo.org/api/records"

# Fixed list of the four files you actually want in the latest release:
FILES=(
  "met_towers_2017-2022_manual-outlier-id.zip"
  "source_data_015m.zip"
  "source_data_060m.zip"
  "supplementary.zip"
)

mkdir -p "$DATA_DIR"

# 1) get conceptrecid
raw_json=$(curl -s "${API}/${INITIAL_RECORD_ID}")
concept_id=$(printf '%s' "$raw_json" \
  | sed -n 's/.*"conceptrecid"[[:space:]]*:[[:space:]]*"\{0,1\}\([0-9][0-9]*\)"\{0,1\}.*/\1/p' \
  | head -n1
)
echo "Using conceptrecid = $concept_id"

# 2) find most recent version
search_url="${API}?q=conceptrecid:${concept_id}&sort=mostrecent&size=1"
latest_id=$(curl -s "$search_url" \
  | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' \
  | head -n1
)
echo "Latest version = $latest_id"

# 3) download & extract
for name in "${FILES[@]}"; do
  quoted=$(printf '%s' "$name" | sed 's/ /%20/g')
  url="https://zenodo.org/api/records/${latest_id}/files/${quoted}/content"
  out="$DATA_DIR/$name"

  if [[ -f "$out" ]]; then
    echo "SKIP: $name already downloaded"
  else
    echo "Downloading $name"
    curl -L -o "$out" "$url"
  fi

  if [[ "$name" == *.zip ]]; then
    folder="$DATA_DIR/${name%.zip}"

    # If the folder exists, remove it so we overwrite cleanly
    if [[ -d "$folder" ]]; then
      echo "Overwriting existing folder $folder"
      rm -rf "$folder"
    fi

    echo "Extracting $name → $folder"
    unzip -q "$out" -d "$folder"

    # If the zip itself contains a top-level folder of the same name, collapse it:
    nested="$folder/${name%.zip}"
    if [[ -d "$nested" ]]; then
      echo "Flattening nested dir → $folder"
      shopt -s dotglob
      mv "$nested"/* "$folder"/
      rmdir "$nested"
      shopt -u dotglob
    fi

    echo "Removing $name"
    rm -f "$out"
  fi
done

echo "Done. Your four datasets are in '$DATA_DIR/'."