# Polaris

Generalization evaluation framework for molecular ML — Polaris Model Validation Paper

GitHub: https://github.com/srijitseal/polaris

## Overview

This project evaluates how well molecular ML models generalize beyond their training distributions, using the **Expansion Tx ADMET dataset** (7,618 molecules, 10 ADME endpoints, 4 CROs + internal data). The framework implements splitting strategies that mimic real-world deployment scenarios and measures performance degradation as a function of distance from training data.

## Setup

```bash
pixi install
bash scripts/download_data.sh
```

### ChEMBL data (notebook 0.03)

Notebook `0.03-araripe-chembl-tanimoto.py` uses `chembl_downloader` to query the ChEMBL 36 SQLite database (~4GB). The database is downloaded automatically on first run and cached at `~/.data/chembl/` by [pystow](https://github.com/cthoyt/pystow). Fingerprint caches (~600MB) are stored in `data/interim/`.

## Data

The dataset is from the [OpenADMET Expansion Rx Challenge](https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-data) — real-world ADMET data from Expansion Therapeutics' RNA-targeted drug discovery campaigns.

| File | Molecules | Description |
|------|-----------|-------------|
| `expansion_data_raw.csv` | 7,618 | Full dataset with out-of-range modifiers |
| `expansion_data_train.csv` | 5,326 | ML-ready train split |
| `expansion_data_test.csv` | 2,282 | ML-ready test split |

**Endpoints** (10): LogD, KSOL, HLM CLint, RLM CLint, MLM CLint, Caco-2 Papp A>B, Caco-2 Efflux, MPPB, MBPB, MGMB
