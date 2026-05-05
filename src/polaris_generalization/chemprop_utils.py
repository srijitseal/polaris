"""Shared Chemprop D-MPNN training utility for single-task regression."""

import os
from pathlib import Path

import numpy as np
from loguru import logger

from polaris_generalization.config import INTERIM_DATA_DIR

# Default training duration. Chemprop's own default is 50; increase for longer runs.
DEFAULT_MAX_EPOCHS = 50
ENSEMBLE_SIZE = 3
MIN_VAL_FOR_EARLY_STOPPING = 5  # Skip early stopping if val set is smaller

# Foundation model weights are version-pinned and shipped via the data/ folder so
# collaborators inherit them without needing to download anything themselves.
# Source: https://doi.org/10.5281/zenodo.15426600 (Zenodo record 15460715).
CHEMELEON_WEIGHTS = INTERIM_DATA_DIR / "chemeleon_mp.pt"
CHEMELEON_ZENODO_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"


def _ensure_chemeleon_weights(weights_path: Path = CHEMELEON_WEIGHTS) -> Path:
    """Download CheMeleon weights from Zenodo into `weights_path` if missing."""
    weights_path = Path(weights_path)
    if weights_path.exists():
        return weights_path

    from urllib.request import urlretrieve

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"CheMeleon weights not found at {weights_path}; downloading from {CHEMELEON_ZENODO_URL}"
    )
    tmp_path = weights_path.with_suffix(weights_path.suffix + ".part")
    urlretrieve(CHEMELEON_ZENODO_URL, tmp_path)
    tmp_path.rename(weights_path)
    logger.info(f"Saved CheMeleon weights to {weights_path}")
    return weights_path


def train_chemprop(
    train_smiles: list[str],
    train_y: np.ndarray,
    test_smiles: list[str],
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    val_fraction: float = 0.1,
    seed: int = 42,
    cache_dir: Path | None = None,
    cache_key: str | None = None,
    checkpoint_dir: Path | None = None,
) -> np.ndarray:
    """Train an ensemble of D-MPNN regression models and return averaged predictions on test SMILES.

    Parameters
    ----------
    train_smiles : protonated SMILES for training molecules
    train_y : target values shape (n_train,), already log-transformed if applicable
    test_smiles : protonated SMILES to predict on
    max_epochs : training epochs (default DEFAULT_MAX_EPOCHS)
    val_fraction : fraction of train carved out as validation (for early stopping)
    seed : base seed; each ensemble member uses seed+i for val split and weight init
    cache_dir : directory for caching predictions as .npy files
    cache_key : unique key for this job, e.g. "2.08_LogD_baseline"
    checkpoint_dir : if set, saves {cache_key}_e{i}.ckpt here after training

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
        safe_key = f"{safe_key}_e{ENSEMBLE_SIZE}"
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

    from lightning.pytorch.callbacks import EarlyStopping

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = [0] if torch.cuda.is_available() else "auto"

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    test_pts = [data.MoleculeDatapoint.from_smi(smi) for smi in test_smiles]

    # Track best weights in memory so we can restore after early stopping
    class _BestWeights(pl.Callback):
        def __init__(self):
            self.best_val_loss = float("inf")
            self.best_state: dict | None = None

        def on_validation_end(self, trainer, pl_module):
            val_loss = trainer.callback_metrics.get("val_loss")
            if val_loss is not None and float(val_loss) < self.best_val_loss:
                self.best_val_loss = float(val_loss)
                self.best_state = {k: v.clone() for k, v in pl_module.state_dict().items()}

        def on_fit_end(self, trainer, pl_module):
            if self.best_state is not None:
                pl_module.load_state_dict(self.best_state)

    ensemble_preds = []
    for i in range(ENSEMBLE_SIZE):
        member_seed = seed + i
        pl.seed_everything(member_seed, workers=True)

        rng = np.random.default_rng(member_seed)
        n_total = len(train_smiles)
        n_val = max(1, int(val_fraction * n_total))
        val_idx = rng.choice(n_total, size=n_val, replace=False)
        train_idx = np.setdiff1d(np.arange(n_total), val_idx)

        all_pts = [data.MoleculeDatapoint.from_smi(smi, [float(y)]) for smi, y in zip(train_smiles, train_y)]
        train_pts = [all_pts[j] for j in train_idx]
        val_pts = [all_pts[j] for j in val_idx]

        train_dset = data.MoleculeDataset(train_pts, featurizer)
        val_dset = data.MoleculeDataset(val_pts, featurizer)
        test_dset = data.MoleculeDataset(test_pts, featurizer)

        scaler = train_dset.normalize_targets()
        val_dset.normalize_targets(scaler)

        output_transform = chemprop_nn.UnscaleTransform.from_standard_scaler(scaler)
        mp = chemprop_nn.BondMessagePassing()
        agg = chemprop_nn.MeanAggregation()
        ffn = chemprop_nn.RegressionFFN(n_tasks=1, output_transform=output_transform)
        mpnn = models.MPNN(mp, agg, ffn)

        train_loader = data.build_dataloader(train_dset, num_workers=4)
        val_loader = data.build_dataloader(val_dset, shuffle=False, num_workers=4)
        test_loader = data.build_dataloader(test_dset, shuffle=False, num_workers=4)

        best_weights = _BestWeights()
        callbacks = [best_weights]
        if n_val >= MIN_VAL_FOR_EARLY_STOPPING:
            callbacks.append(EarlyStopping(monitor="val_loss", patience=10, mode="min"))
        else:
            logger.warning(f"Val set too small ({n_val} samples), skipping early stopping")

        trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            enable_model_summary=False,
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            callbacks=callbacks,
        )
        trainer.fit(mpnn, train_loader, val_loader)
        n_epochs_trained = min(trainer.current_epoch + 1, max_epochs)
        logger.info(
            f"Chemprop ensemble {i + 1}/{ENSEMBLE_SIZE} | device={trainer.accelerator.__class__.__name__} | trained {n_epochs_trained}/{max_epochs} epochs | key={safe_key or 'no-cache'}"
        )

        if checkpoint_dir and safe_key:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(checkpoint_dir / f"{safe_key}_e{i}.ckpt")
            config_file = checkpoint_dir / "training_config.json"
            if not config_file.exists():
                import json

                config_file.write_text(
                    json.dumps(
                        {
                            "max_epochs": max_epochs,
                            "val_fraction": val_fraction,
                            "seed": seed,
                            "n_ensemble": ENSEMBLE_SIZE,
                            "architecture": "D-MPNN (BondMessagePassing + MeanAggregation + RegressionFFN)",
                            "hidden_dim": 300,
                            "depth": 3,
                        },
                        indent=2,
                    )
                )

        with torch.inference_mode():
            preds_batched = trainer.predict(mpnn, test_loader)

        member_preds = np.concatenate([p.numpy() for p in preds_batched], axis=0).squeeze(-1)
        ensemble_preds.append(member_preds)

    # Average predictions across ensemble members
    preds = np.mean(ensemble_preds, axis=0)

    if cache_dir and safe_key:
        cache_file = Path(cache_dir) / f"{safe_key}.npy"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, preds)

    return preds


def train_chemeleon(
    train_smiles: list[str],
    train_y: np.ndarray,
    test_smiles: list[str],
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    val_fraction: float = 0.1,
    seed: int = 42,
    cache_dir: Path | None = None,
    cache_key: str | None = None,
    checkpoint_dir: Path | None = None,
    weights_path: Path = CHEMELEON_WEIGHTS,
) -> np.ndarray:
    """Fine-tune a CheMeleon foundation model for single-task regression.

    Initializes the message-passing network from pre-trained CheMeleon weights
    and trains a single model (no ensemble). The FFN is trained from scratch
    with input_dim matched to CheMeleon's output dimension.

    The CheMeleon foundation weights are downloaded from Zenodo into
    `data/interim/chemeleon_mp.pt` on first call and reused on subsequent runs.

    Cache files use the suffix _cm1 to distinguish from standard ensemble (_e3) cache.
    """
    env_cache = os.environ.get("CHEMPROP_CACHE_DIR")
    if env_cache is not None:
        cache_dir = Path(env_cache)

    safe_key = None
    if cache_dir and cache_key:
        safe_key = cache_key.replace(">", "_").replace("<", "_").replace(" ", "_")
        safe_key = f"{safe_key}_cm1"
        cache_file = Path(cache_dir) / f"{safe_key}.npy"
        if cache_file.exists():
            logger.debug(f"CheMeleon cache hit: {cache_file.name}")
            return np.load(cache_file)

    import logging

    import torch
    from chemprop import data, featurizers, models
    from chemprop import nn as chemprop_nn
    from lightning import pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping

    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
    logging.getLogger("lightning").setLevel(logging.WARNING)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = [0] if torch.cuda.is_available() else "auto"

    pl.seed_everything(seed, workers=True)

    weights_path = _ensure_chemeleon_weights(weights_path)
    chemeleon_ckpt = torch.load(weights_path, weights_only=True)
    mp = chemprop_nn.BondMessagePassing(**chemeleon_ckpt["hyper_parameters"])
    mp.load_state_dict(chemeleon_ckpt["state_dict"])

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    rng = np.random.default_rng(seed)
    n_total = len(train_smiles)
    n_val = max(1, int(val_fraction * n_total))
    val_idx = rng.choice(n_total, size=n_val, replace=False)
    train_idx = np.setdiff1d(np.arange(n_total), val_idx)

    all_pts = [data.MoleculeDatapoint.from_smi(smi, [float(y)]) for smi, y in zip(train_smiles, train_y)]
    train_pts = [all_pts[j] for j in train_idx]
    val_pts = [all_pts[j] for j in val_idx]
    test_pts = [data.MoleculeDatapoint.from_smi(smi) for smi in test_smiles]

    train_dset = data.MoleculeDataset(train_pts, featurizer)
    val_dset = data.MoleculeDataset(val_pts, featurizer)
    test_dset = data.MoleculeDataset(test_pts, featurizer)

    scaler = train_dset.normalize_targets()
    val_dset.normalize_targets(scaler)

    output_transform = chemprop_nn.UnscaleTransform.from_standard_scaler(scaler)
    agg = chemprop_nn.MeanAggregation()
    ffn = chemprop_nn.RegressionFFN(n_tasks=1, output_transform=output_transform, input_dim=mp.output_dim, hidden_dim=1024, n_layers=2)
    mpnn = models.MPNN(mp, agg, ffn, batch_norm=False)

    train_loader = data.build_dataloader(train_dset, num_workers=4)
    val_loader = data.build_dataloader(val_dset, shuffle=False, num_workers=4)
    test_loader = data.build_dataloader(test_dset, shuffle=False, num_workers=4)

    class _BestWeights(pl.Callback):
        def __init__(self):
            self.best_val_loss = float("inf")
            self.best_state: dict | None = None

        def on_validation_end(self, trainer, pl_module):
            val_loss = trainer.callback_metrics.get("val_loss")
            if val_loss is not None and float(val_loss) < self.best_val_loss:
                self.best_val_loss = float(val_loss)
                self.best_state = {k: v.clone() for k, v in pl_module.state_dict().items()}

        def on_fit_end(self, trainer, pl_module):
            if self.best_state is not None:
                pl_module.load_state_dict(self.best_state)

    best_weights = _BestWeights()
    callbacks = [best_weights]
    if n_val >= MIN_VAL_FOR_EARLY_STOPPING:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=10, mode="min"))
    else:
        logger.warning(f"Val set too small ({n_val} samples), skipping early stopping")

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        callbacks=callbacks,
    )
    trainer.fit(mpnn, train_loader, val_loader)
    n_epochs_trained = min(trainer.current_epoch + 1, max_epochs)
    logger.info(
        f"CheMeleon | device={trainer.accelerator.__class__.__name__} | trained {n_epochs_trained}/{max_epochs} epochs | key={safe_key or 'no-cache'}"
    )

    if checkpoint_dir and safe_key:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(checkpoint_dir / f"{safe_key}.ckpt")
        config_file = checkpoint_dir / "training_config.json"
        if not config_file.exists():
            import json
            config_file.write_text(
                json.dumps(
                    {
                        "model": "CheMeleon",
                        "weights_path": str(weights_path),
                        "max_epochs": max_epochs,
                        "val_fraction": val_fraction,
                        "seed": seed,
                        "n_ensemble": 1,
                        "ffn_hidden_dim": 1024,
                        "ffn_n_layers": 2,
                        "mp_output_dim": mp.output_dim,
                        "batch_norm": False,
                    },
                    indent=2,
                )
            )

    preds_batched = trainer.predict(mpnn, test_loader)
    preds = np.concatenate([p.numpy() for p in preds_batched], axis=0).squeeze(-1)

    if cache_dir and safe_key:
        cache_file = Path(cache_dir) / f"{safe_key}.npy"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, preds)

    return preds
