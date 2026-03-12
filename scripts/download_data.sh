#!/usr/bin/env bash
# Download Expansion Rx ADMET dataset from HuggingFace to data/raw/
set -euo pipefail

REPO="openadmet/openadmet-expansionrx-challenge-data"
BASE_URL="https://huggingface.co/datasets/${REPO}/resolve/main"
OUT_DIR="$(dirname "$0")/../data/raw"
FILES=(expansion_data_raw.csv expansion_data_train.csv expansion_data_test.csv)

mkdir -p "$OUT_DIR"

# Check if all files already exist
all_present=true
for file in "${FILES[@]}"; do
    if [[ ! -f "${OUT_DIR}/${file}" ]]; then
        all_present=false
        break
    fi
done

if $all_present; then
    echo "Data already downloaded. Skipping."
    exit 0
fi

for file in "${FILES[@]}"; do
    if [[ -f "${OUT_DIR}/${file}" ]]; then
        echo "Already exists: ${file}"
    else
        echo "Downloading ${file}..."
        curl -L -o "${OUT_DIR}/${file}" "${BASE_URL}/${file}"
    fi
done

echo "Done. Files saved to ${OUT_DIR}/"
ls -lh "${OUT_DIR}"/*.csv
