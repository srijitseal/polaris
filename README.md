# Polaris

Generalization evaluation framework for molecular ML — Polaris Model Validation Paper

GitHub: https://github.com/srijitseal/polaris

## Overview

This project evaluates how well molecular ML models generalize beyond their training distributions, using the **Expansion Tx ADMET dataset** (7,618 molecules, 10 ADME endpoints, 4 CROs + internal data). The framework implements splitting strategies that mimic real-world deployment scenarios and measures performance degradation as a function of distance from training data.

## Setup

```bash
pixi install
```
