#!/usr/bin/env bash
# Download Expansion Rx ADMET dataset from HuggingFace to data/raw/
set -euo pipefail

REPO="openadmet/openadmet-expansionrx-challenge-data"
BASE_URL="https://huggingface.co/datasets/${REPO}/resolve/main"
OUT_DIR="$(dirname "$0")/../data/raw"

mkdir -p "$OUT_DIR"

for file in expansion_data_raw.csv expansion_data_train.csv expansion_data_test.csv; do
    echo "Downloading ${file}..."
    curl -L -o "${OUT_DIR}/${file}" "${BASE_URL}/${file}"
done

echo "Done. Files saved to ${OUT_DIR}/"
ls -lh "${OUT_DIR}"/*.csv
