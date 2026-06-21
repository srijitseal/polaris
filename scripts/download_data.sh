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

# Biogen ADME public set (Fang et al. 2023, JCIM) — external comparison dataset
# for the cross-dataset characterization case study (notebooks/4.01).
EXT_DIR="$(dirname "$0")/../data/external"
BIOGEN_FILE="biogen_adme_public_set_3521.csv"
BIOGEN_URL="https://raw.githubusercontent.com/molecularinformatics/Computational-ADME/main/ADME_public_set_3521.csv"
mkdir -p "$EXT_DIR"
if [[ -f "${EXT_DIR}/${BIOGEN_FILE}" ]]; then
    echo "Already exists: ${BIOGEN_FILE}"
else
    echo "Downloading ${BIOGEN_FILE}..."
    curl -L -o "${EXT_DIR}/${BIOGEN_FILE}" "$BIOGEN_URL"
fi
