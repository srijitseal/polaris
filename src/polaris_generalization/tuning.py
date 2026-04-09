"""Shared Optuna-based hyperparameter tuning for XGBoost."""

import json
from pathlib import Path

import numpy as np
import optuna
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 30,
    cv: int = 3,
    random_state: int = 42,
    cache_dir: Path | None = None,
    cache_key: str | None = None,
) -> tuple[XGBRegressor, dict, optuna.study.Study | None]:
    """Tune XGBoost with Optuna TPE sampler.

    Parameters
    ----------
    X_train : array of shape (n_samples, n_features)
    y_train : array of shape (n_samples,)
    n_trials : number of Optuna trials (default 30)
    cv : cross-validation folds for scoring (default 3)
    random_state : random seed for reproducibility
    cache_dir : directory for caching best params as JSON (optional)
    cache_key : unique key for this tuning job, e.g. "LogD_cluster_fold0" (optional)

    Returns
    -------
    (best_model, best_params, study) where study is None if loaded from cache.
    """
    # Check cache
    if cache_dir and cache_key:
        safe_key = cache_key.replace(">", "_").replace("<", "_")
        cache_file = Path(cache_dir) / f"{safe_key}.json"
        if cache_file.exists():
            params = json.loads(cache_file.read_text())
            model = XGBRegressor(**params, tree_method="hist", random_state=random_state, verbosity=0)
            model.fit(X_train, y_train)
            return model, params, None

    # Run Optuna optimization
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
        }
        model = XGBRegressor(**params, tree_method="hist", random_state=random_state, verbosity=0)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error")
        return -scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)

    # Train final model with best params
    best_model = XGBRegressor(**study.best_params, tree_method="hist", random_state=random_state, verbosity=0)
    best_model.fit(X_train, y_train)

    # Save to cache
    if cache_dir and cache_key:
        safe_key = cache_key.replace(">", "_").replace("<", "_")
        cache_file = Path(cache_dir) / f"{safe_key}.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(study.best_params, indent=2))

    return best_model, study.best_params, study
