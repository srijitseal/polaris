"""Shared Chemprop D-MPNN training utility for single-task regression."""

import os
from pathlib import Path

import numpy as np
from loguru import logger

# Default training duration. Chemprop's own default is 50; increase for longer runs.
DEFAULT_MAX_EPOCHS = 50


def train_chemprop(
    train_smiles: list[str],
    train_y: np.ndarray,
    test_smiles: list[str],
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    val_fraction: float = 0.1,
    seed: int = 42,
    cache_dir: Path | None = None,
    cache_key: str | None = None,
) -> np.ndarray:
    """Train a D-MPNN regression model and return predictions on test SMILES.

    Parameters
    ----------
    train_smiles : protonated SMILES for training molecules
    train_y : target values shape (n_train,), already log-transformed if applicable
    test_smiles : protonated SMILES to predict on
    max_epochs : training epochs (default DEFAULT_MAX_EPOCHS)
    val_fraction : fraction of train carved out as validation (for LR scheduler)
    seed : controls val split and model weight initialization
    cache_dir : directory for caching predictions as .npy files
    cache_key : unique key for this job, e.g. "2.08_LogD_baseline"

    Returns
    -------
    np.ndarray of shape (n_test,) in the same scale as train_y
    """
    env_cache = os.environ.get("CHEMPROP_CACHE_DIR")
    if env_cache is not None:
        cache_dir = Path(env_cache)

    safe_key = None
    if cache_dir and cache_key:
        safe_key = cache_key.replace(">", "_").replace("<", "_").replace(" ", "_")
        cache_file = Path(cache_dir) / f"{safe_key}.npy"
        if cache_file.exists():
            logger.debug(f"Chemprop cache hit: {cache_file.name}")
            return np.load(cache_file)

    # Lazy imports — avoids loading PyTorch when loading cached results
    import logging

    import torch
    from chemprop import data, featurizers, models
    from chemprop import nn as chemprop_nn
    from lightning import pytorch as pl

    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
    logging.getLogger("lightning").setLevel(logging.WARNING)

    pl.seed_everything(seed, workers=True)

    # Deterministic val split — same indices every run for same seed
    rng = np.random.default_rng(seed)
    n_total = len(train_smiles)
    n_val = max(1, int(val_fraction * n_total))
    val_idx = rng.choice(n_total, size=n_val, replace=False)
    train_idx = np.setdiff1d(np.arange(n_total), val_idx)

    # Build datapoints — y must be a sequence for MoleculeDatapoint
    all_pts = [
        data.MoleculeDatapoint.from_smi(smi, [float(y)])
        for smi, y in zip(train_smiles, train_y)
    ]
    train_pts = [all_pts[i] for i in train_idx]
    val_pts = [all_pts[i] for i in val_idx]
    test_pts = [data.MoleculeDatapoint.from_smi(smi) for smi in test_smiles]

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dset = data.MoleculeDataset(train_pts, featurizer)
    val_dset = data.MoleculeDataset(val_pts, featurizer)
    test_dset = data.MoleculeDataset(test_pts, featurizer)

    # Normalize targets; bake inverse into model so predictions arrive in original scale
    scaler = train_dset.normalize_targets()
    val_dset.normalize_targets(scaler)

    output_transform = chemprop_nn.UnscaleTransform.from_standard_scaler(scaler)
    mp = chemprop_nn.BondMessagePassing()
    agg = chemprop_nn.MeanAggregation()
    ffn = chemprop_nn.RegressionFFN(n_tasks=1, output_transform=output_transform)
    mpnn = models.MPNN(mp, agg, ffn)

    train_loader = data.build_dataloader(train_dset, num_workers=0)
    val_loader = data.build_dataloader(val_dset, shuffle=False, num_workers=0)
    test_loader = data.build_dataloader(test_dset, shuffle=False, num_workers=0)

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="auto",
        devices=[0],
        max_epochs=max_epochs,
    )
    trainer.fit(mpnn, train_loader, val_loader)
    logger.info(f"Chemprop trained | device={trainer.accelerator.__class__.__name__} | epochs={max_epochs} | key={safe_key or 'no-cache'}")

    with torch.inference_mode():
        preds_batched = trainer.predict(mpnn, test_loader)

    # preds_batched: list of tensors shape (batch, 1) → flatten to (n_test,)
    preds = np.concatenate([p.numpy() for p in preds_batched], axis=0).squeeze(-1)

    if cache_dir and safe_key:
        cache_file = Path(cache_dir) / f"{safe_key}.npy"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, preds)

    return preds
