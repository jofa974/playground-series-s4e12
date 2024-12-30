import pickle
from pathlib import Path

import catboost as cb
import numpy as np
import optuna
import pandas as pd

from dvclive import Live
from insurance.common import OUT_PATH, TARGET_COLUMN
from insurance.data_pipeline import get_feat_columns, get_folds
from insurance.logger import setup_logger

logger = setup_logger(name="catboost")


def get_oof_preds(X_train: pd.DataFrame, model_path: Path) -> np.ndarray[np.float64]:
    """Assumes data is transformed."""
    X_train = X_train.copy()
    models = pickle.load(model_path.open("rb"))

    oof_preds = np.zeros(len(X_train))
    folds = get_folds()
    splits = folds.split(X_train)
    for i, ((_, test_index), model) in enumerate(zip(splits, models)):
        logger.info(f"Predicting OOF -- {i+1}/{len(models)}")
        oof_preds[test_index] = model.predict(data=X_train.loc[test_index, :])
    return oof_preds


def get_avg_preds(X: pd.DataFrame, model_path: Path) -> np.ndarray[np.float64]:
    """Assumes data is transformed."""
    X = X.copy()
    models = pickle.load(model_path.open("rb"))

    preds = np.zeros(len(X))
    for i, model in enumerate(models):
        logger.info(f"Predicting on Test Data -- {i+1}/{len(models)}")
        preds += model.predict(
            data=X,
        )
    preds = preds / len(models)
    return preds


def tune_catboost(train_pool: cb.Pool):
    def objective(trial):
        param = {
            "loss_function": "RMSE",
            "iterations": 40,
            "learning_rate": 0.5,
            "devices": [0],
            "task_type": "GPU",
            # "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 1, 10),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            ),
        }
        if param["bootstrap_type"] == "Bayesian":
            param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
        elif param["bootstrap_type"] == "Bernoulli":
            param["subsample"] = trial.suggest_float("subsample", 0.1, 1, log=True)
        history = cb.cv(
            pool=train_pool,
            params=param,
            fold_count=5,
            partition_random_seed=0,
            shuffle=True,
            as_pandas=True,
            verbose=False,
            type="Classical",
            return_models=False,
        )
        mean_rmse = history["test-RMSE-mean"].values[-1]

        print(f"Out-of-fold RMSLE: {mean_rmse:.4f}")
        return mean_rmse

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=50)

    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("catboost_params = {")
    for key, value in trial.params.items():
        if isinstance(value, str):
            print('    "{}": "{}",'.format(key, value))
        else:
            print('    "{}": {},'.format(key, value))
    print("}")


def train(
    params: dict, model_name: str, layer: int, train_data: pd.DataFrame, test_data: pd.DataFrame
) -> tuple[np.ndarray]:
    feat_cols = get_feat_columns()
    X_train = train_data.drop(columns=[TARGET_COLUMN])
    y_train = train_data[TARGET_COLUMN]

    logger.info(f"Train shape: {X_train.shape=}")
    logger.info(f"Columns: {X_train.columns=}")

    train_pool = cb.Pool(
        data=X_train, label=y_train, cat_features=feat_cols.categorical, has_header=True
    )

    folds = get_folds(n_splits=5)

    tune = False
    if tune:
        tune_catboost(train_pool=train_pool)
        return

    logger.info(f"{params=}")
    history, cv_boosters = cb.cv(
        pool=train_pool,
        params=params,
        folds=folds,
        as_pandas=True,
        verbose=False,
        type="Classical",
        return_models=True,
    )

    live_dir = Path(f"dvclive/{model_name}_layer_{layer}/")
    live_dir.mkdir(parents=True, exist_ok=True)
    with Live(dir=str(live_dir)) as live:
        live.log_plot(
            f"Catboost CV Loss {layer}",
            history,
            x="booster",
            y=["train-RMSE-mean", "test-RMSE-mean"],
            template="linear",
            y_label="RMSLE",
            x_label="Booster",
        )

        live.log_metric(f"{model_name}/train-cv-loss", history["train-RMSE-mean"].iloc[-1])
        live.log_metric(f"{model_name}/test-cv-loss", history["test-RMSE-mean"].iloc[-1])

    model_path = OUT_PATH / f"models/{model_name}_model_layer_{layer}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(cv_boosters, open(model_path, "wb"))
    logger.info(f"Model saved at {model_path}")

    oof_preds = get_oof_preds(X_train=X_train, model_path=model_path)
    avg_preds = get_avg_preds(X=test_data, model_path=model_path)

    return oof_preds, avg_preds
