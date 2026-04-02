# ADR-002: Switch from HalvingRandomSearchCV to Optuna TPE for hyperparameter tuning

**Status**: Accepted

## Context

Only 1 of 8 XGBoost notebooks (2.08) performed hyperparameter tuning, using scikit-learn's HalvingRandomSearchCV with 50 random candidates and successive halving. The remaining 7 notebooks used default XGBoost parameters, meaning analysis results depended on arbitrary defaults rather than optimized configurations. HalvingRandomSearchCV draws candidates randomly upfront with no learning between evaluations.

## Decision

Switch all notebooks to Optuna TPE (Tree-structured Parzen Estimator) optimization with a shared utility function and JSON-based parameter caching. TPE is Bayesian — each trial informs the next via kernel density estimation — making it more sample-efficient than random search for the same budget (30 trials). Cached parameters are keyed by (endpoint, split_strategy, repeat, fold) and stored in `data/interim/optuna_cache/`.

## Implementation

- Shared utility: `src/polaris_generalization/tuning.py` (`tune_xgboost` function)
- 9 hyperparameters: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`
- Cache: JSON files in `data/interim/optuna_cache/`, keyed by endpoint + split + fold
- All 8 notebooks updated to call `tune_xgboost` with `cache_dir` and `cache_key`
- 30 trials per tuning job, 3-fold CV, MAE scoring

## Consequences

**Benefits:**
- All analyses use optimized models — results reflect best achievable performance per endpoint
- Bayesian search finds better params in fewer trials than random
- Caching means tuning cost is paid once; subsequent runs load params instantly
- Shared utility eliminates duplicated model creation code across 8 notebooks

**Tradeoffs:**
- First run is slower (30 trials x 3-fold CV per endpoint per fold)
- Additional dependency (optuna)
- Cached params must be invalidated if the feature set or split strategy changes (delete `data/interim/optuna_cache/`)
