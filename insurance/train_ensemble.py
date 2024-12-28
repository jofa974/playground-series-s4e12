import pickle
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import typer
from typing import Annotated
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
)
from sklearn.impute import SimpleImputer

from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error

from dvclive import Live
from insurance.common import OUT_PATH, TARGET_COLUMN, OOF_PREDS_PATH
from insurance.data_pipeline import get_folds
from insurance.logger import setup_logger

logger = setup_logger(name="ensemble")


PARAMS = {
    "alpha": 1.6333478843987628,
    "solver": "saga",
    "random_state": 42,
}


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


def make_ensemble_pipeline(pred_columns: list[str]) -> Pipeline:
    num_transformer = Pipeline([("scaler", StandardScaler())])
    imputer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    pipeline = Pipeline(
        [
            # (
            #     "claims_imputer",
            #     ColumnTransformer(
            #         transformers=[
            #             (
            #                 "impute",
            #                 imputer,
            #                 [
            #                     "Previous Claims",
            #                 ],
            #             ),
            #         ],
            #         remainder="passthrough",
            #         verbose_feature_names_out=False,
            #     ),
            # ),
            (
                "num_scaler",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            num_transformer,
                            pred_columns,
                        ),
                    ],
                    remainder="drop",
                    verbose_feature_names_out=False,
                ),
            ),
        ]
    )
    pipeline.set_output(transform="pandas")
    return pipeline


def main(
    previous_layer: Annotated[int, typer.Option(help="Previous layer number")],
):
    df = pd.concat(
        [
            pd.read_feather(OOF_PREDS_PATH / f"xgboost_layer_{previous_layer}.feather"),
            pd.read_feather(OOF_PREDS_PATH / f"catboost_layer_{previous_layer}.feather")[
                f"catboost_layer_{previous_layer}"
            ],
            pd.read_feather(OOF_PREDS_PATH / f"lgbm_layer_{previous_layer}.feather")[
                f"lgbm_layer_{previous_layer}"
            ],
        ],
        axis=1,
    )

    X_train = df.drop(columns=[TARGET_COLUMN])
    y_train = df[TARGET_COLUMN]
    y_train = np.log1p(y_train)

    data_pipeline = make_ensemble_pipeline(
        pred_columns=[
            f"xgboost_layer_{previous_layer}",
            f"catboost_layer_{previous_layer}",
            f"lgbm_layer_{previous_layer}",
        ]
    )
    X_train = data_pipeline.fit_transform(X_train)

    data_pipeline_path = OUT_PATH / "data_pipeline_train_ensemble.pkl"
    pickle.dump(data_pipeline, open(data_pipeline_path, "wb"))
    logger.info(f"Data pipeline saved at {data_pipeline_path}")

    logger.info(f"Train shape: {X_train.shape=}")
    logger.info(f"Columns: {X_train.columns}")

    n_splits = 5
    folds = get_folds(n_splits=n_splits)

    tune = False
    if tune:
        tune_ensemble(X_train=X_train, y_train=y_train)
        return
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

    model_path = OUT_PATH / "models/ensemble_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(ensemble_regressors, open(model_path, "wb"))
    logger.info(f"Model saved at {model_path}")


if __name__ == "__main__":
    typer.run(main)
