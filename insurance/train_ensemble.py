import copy
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import typer
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
)
from sklearn.impute import SimpleImputer

from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error

from dvclive import Live
from insurance.common import OUT_PATH
from insurance.data_pipeline import get_folds
from insurance.logger import setup_logger
from insurance.train_catboost import get_oof_preds as catboost_oof_preds
from insurance.train_xgboost import get_oof_preds as xgboost_oof_preds

MODEL_PATH = OUT_PATH / "models/ensemble_model.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

DATA_PIPELINE_PATH = OUT_PATH / "data_pipeline_train_ensemble.pkl"

log_file = datetime.now().strftime("ensemble_train_log_%Y-%m-%d_%H-%M-%S.log")
logger = setup_logger(log_file=log_file, name="ensemble trainer")

PREV_LAYER_OOF = {"xgboost": xgboost_oof_preds, "catboost": catboost_oof_preds}

PARAMS = {
    "alpha": 1.6333478843987628,
    "solver": "saga",
    "random_state": 42,
}


def custom_learning_rate(current_iter):
    base_learning_rate = 0.3
    lr = base_learning_rate * np.power(0.95, current_iter)
    return lr if lr > 1e-3 else 1e-3


def plot_train_test(history: pd.DataFrame):
    # Plotting
    fig, ax = plt.subplots()

    ax.plot(history["booster"], history["train-rmse-mean"], label="Train RMSE Mean")
    ax.plot(history["booster"], history["test-rmse-mean"], label="Test RMSE Mean")

    # Adding labels, title, and legend
    ax.set_xlabel("Booster")
    ax.set_ylabel("RMSE Mean")
    ax.set_title("Train and Test RMSE Mean vs Booster")
    plt.legend()
    plt.grid(True)

    last_booster = history["booster"].iloc[-1]
    train_rmse_last = history["train-rmse-mean"].iloc[-1]
    test_rmse_last = history["test-rmse-mean"].iloc[-1]

    ax.text(
        last_booster,
        train_rmse_last,
        f"{train_rmse_last:.6f}",
        fontsize=10,
        ha="left",
        va="bottom",
        color="blue",
        bbox=dict(facecolor="white", alpha=0.8),
    )
    ax.text(
        last_booster,
        test_rmse_last,
        f"{test_rmse_last:.6f}",
        fontsize=10,
        ha="left",
        va="bottom",
        color="orange",
        bbox=dict(facecolor="white", alpha=0.8),
    )
    plt.draw()

    fig_path = OUT_PATH / "ensemble_training.png"
    fig.savefig(fig_path, dpi=300)
    logger.info(f"Training loss saved at {fig_path}")


class SaveBestModel(xgb.callback.TrainingCallback):
    def __init__(self, cvboosters):
        self._cvboosters = cvboosters

    def after_training(self, model):
        self._cvboosters[:] = [cvpack.bst for cvpack in model.cvfolds]
        return model


def tune_ensemble(X_train: pd.DataFrame, y_train: pd.Series):
    def objective(trial):
        param = {
            "alpha": trial.suggest_float("alpha", 1e-1, 100, log=True),
            "solver": trial.suggest_categorical(
                "solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"]
            ),
        }

        if param["solver"] in ["sag", "saga"]:
            param["random_state"] = 42
        if param["solver"] == "lbfgs":
            param["positive"] = True

        n_splits = 5
        folds = get_folds(n_splits=5)
        mean_rmse = 0
        for train_idx, val_idx in folds.split(X_train):
            model = Ridge(**param)
            X, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X=X, y=y)
            test_preds = model.predict(X=X_val)
            test_rmse = root_mean_squared_error(y_true=y_val, y_pred=test_preds)
            mean_rmse += test_rmse / n_splits

        print(f"Out-of-fold RMSLE: {mean_rmse:.4f}")
        return mean_rmse

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=100)

    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("PARAMS = {")
    for key, value in trial.params.items():
        if isinstance(value, str):
            print('    "{}": "{}",'.format(key, value))
        else:
            print('    "{}": {},'.format(key, value))
    print("}")


def make_ensemble_pipeline() -> Pipeline:
    num_transformer = Pipeline([("scaler", StandardScaler())])
    imputer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    pipeline = Pipeline(
        [
            (
                "claims_imputer",
                ColumnTransformer(
                    transformers=[
                        (
                            "impute",
                            imputer,
                            [
                                "Previous Claims",
                            ],
                        ),
                    ],
                    remainder="passthrough",
                    verbose_feature_names_out=False,
                ),
            ),
            (
                "num_scaler",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            num_transformer,
                            ["xgboost_preds", "catboost_preds", "Previous Claims"],
                        ),
                    ],
                    remainder="passthrough",
                    verbose_feature_names_out=False,
                ),
            ),
        ]
    )
    pipeline.set_output(transform="pandas")
    return pipeline


def main(prep_data_path: Path):
    target_column = "Premium Amount"

    df = pd.read_feather(prep_data_path)

    X_train = df.drop(columns=[target_column])
    y_train = df[target_column]
    y_train = np.log1p(y_train)

    columns = X_train.columns
    for model, oof_func in PREV_LAYER_OOF.items():
        logger.info(f"{model} OOF predictions...")
        # Ensure that OOF predictions of a model do not use previous model OOF preds.
        X_train[f"{model}_preds"] = oof_func(X_train=X_train[columns])

    data_pipeline = make_ensemble_pipeline()
    X_train = data_pipeline.fit_transform(X_train)
    X_train = X_train[["xgboost_preds", "catboost_preds", "Previous Claims"]]

    logger.info(f"Train shape: {X_train.shape=}")
    logger.info(f"Columns: {X_train.columns}")

    n_splits = 5
    folds = get_folds(n_splits=n_splits)

    tune = False
    if tune:
        tune_ensemble(X_train=X_train, y_train=y_train)
    else:
        ensemble_regressors = []
        metrics = {"train-rmse-mean": 0.0, "test-rmse-mean": 0.0}
        for train_idx, val_idx in folds.split(X_train):
            model = Ridge()
            X, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X=X, y=y)
            train_preds = model.predict(X=X)
            train_rmse = root_mean_squared_error(y_true=y, y_pred=train_preds)
            test_preds = model.predict(X=X_val)
            test_rmse = root_mean_squared_error(y_true=y_val, y_pred=test_preds)
            metrics["train-rmse-mean"] += train_rmse / n_splits
            metrics["test-rmse-mean"] += test_rmse / n_splits
            ensemble_regressors.append(model)

        live_dir = Path("dvclive/ensemble/")
        live_dir.mkdir(parents=True, exist_ok=True)
        with Live(dir=str(live_dir)) as live:
            live.log_metric("ensemble/train-cv-loss", metrics["train-rmse-mean"])
            live.log_metric("ensemble/test-cv-loss", metrics["test-rmse-mean"])
        pickle.dump(data_pipeline, open(DATA_PIPELINE_PATH, "wb"))
        logger.info(f"Data pipeline saved at {DATA_PIPELINE_PATH}")
        pickle.dump(ensemble_regressors, open(MODEL_PATH, "wb"))
        logger.info(f"Model saved at {MODEL_PATH}")


if __name__ == "__main__":
    typer.run(main)
