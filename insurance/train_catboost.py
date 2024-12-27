import pickle
from pathlib import Path
from typing import Annotated

import catboost as cb
import numpy as np
import optuna
import pandas as pd
import typer

from dvclive import Live
from insurance.common import OUT_PATH, OOF_PREDS_PATH, TARGET_COLUMN
from insurance.data_pipeline import get_feat_columns, get_folds, make_pipeline
from insurance.logger import setup_logger

logger = setup_logger(name="catboost")
DATA_PIPELINE_PATH = OUT_PATH / "data_pipeline_train_catboost.pkl"

catboost_params = {
    "loss_function": "RMSE",
    "iterations": 200,
    "learning_rate": 0.5,
    "devices": [0],
    "task_type": "GPU",
    "depth": 10,
    "boosting_type": "Ordered",
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8106434906203719,
}


def get_oof_preds(X_train: pd.DataFrame, model_path: Path) -> np.ndarray[np.float64]:
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


def main(
    input_data_path: Annotated[Path, typer.Option(help="Input data path")],
    layer: Annotated[int, typer.Option(help="Stack layer number")],
):
    df = pd.read_feather(input_data_path)

    X_train = df.drop(columns=[TARGET_COLUMN])
    logger.info(f"Train shape: {X_train.shape=}")
    y_train = df.loc[X_train.index, TARGET_COLUMN]
    y_train = np.log1p(y_train)

    feat_cols = get_feat_columns()

    if layer == 0:
        logger.info("Transforming data...")
        feat_cols = get_feat_columns()
        data_pipeline = make_pipeline()
        X_train = data_pipeline.fit_transform(X_train)
        for col in feat_cols.categorical:
            X_train[col] = X_train[col].astype("category")
        pickle.dump(data_pipeline, open(DATA_PIPELINE_PATH, "wb"))
        logger.info(f"Data pipeline saved at {DATA_PIPELINE_PATH}")

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

    history, cv_boosters = cb.cv(
        pool=train_pool,
        params=catboost_params,
        folds=folds,
        as_pandas=True,
        verbose=False,
        type="Classical",
        return_models=True,
    )

    live_dir = Path(f"dvclive/catboost_layer_{layer}/")
    live_dir.mkdir(parents=True, exist_ok=True)
    with Live(dir=str(live_dir)) as live:
        live.log_plot(
            "Catboost CV Loss",
            history,
            x="booster",
            y=["train-RMSE-mean", "test-RMSE-mean"],
            template="linear",
            y_label="Booster",
            x_label="RMSLE",
        )

        live.log_metric("catboost/train-cv-loss", history["train-RMSE-mean"].iloc[-1])
        live.log_metric("catboost/test-cv-loss", history["test-RMSE-mean"].iloc[-1])

    model_path = OUT_PATH / f"models/catboost_model_layer_{layer}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(cv_boosters, open(model_path, "wb"))
    logger.info(f"Model saved at {model_path}")

    preds = get_oof_preds(X_train=X_train, model_path=model_path)
    X_train[f"catboost_layer_{layer}"] = preds
    X_train[TARGET_COLUMN] = df[TARGET_COLUMN]

    OOF_PREDS_PATH.mkdir(parents=True, exist_ok=True)
    X_train.to_feather(OOF_PREDS_PATH / f"catboost_layer_{layer}.feather")


if __name__ == "__main__":
    typer.run(main)
