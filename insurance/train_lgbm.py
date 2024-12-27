import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
import lightgbm as lgb

from dvclive import Live
from insurance.common import OUT_PATH
from insurance.data_pipeline import get_feat_columns, make_boosters_pipeline, get_folds
from insurance.logger import setup_logger
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error, root_mean_squared_error

MODEL_PATH = OUT_PATH / "models/lgbm_model.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

DATA_PIPELINE_PATH = OUT_PATH / "data_pipeline_train_lgbm.pkl"

log_file = datetime.now().strftime("lgbm_train_log_%Y-%m-%d_%H-%M-%S.log")
logger = setup_logger(log_file=log_file, name="lgbm trainer")

lgbm_params = {
    "objective": "regression",
    "metric": "rmse",
    "feature_pre_filter": False,
    "lambda_l1": 1.6575159147685196e-08,
    "lambda_l2": 0.0017415507514917511,
    "num_leaves": 67,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "min_child_samples": 20,
}


def get_oof_preds(X_train: pd.DataFrame) -> np.ndarray[np.float64]:
    X_train = X_train.copy()
    models = pickle.load(MODEL_PATH.open("rb"))
    models = models["cvbooster"].boosters

    data_pipeline = pickle.load(DATA_PIPELINE_PATH.open("rb"))
    X_train = data_pipeline.transform(X_train)
    feat_cols = get_feat_columns()
    for col in feat_cols.categorical:
        X_train[col] = X_train[col].astype("category")

    oof_preds = np.zeros(len(X_train))
    folds = get_folds(n_splits=5)
    splits = folds.split(X_train)
    for i, ((_, test_index), model) in enumerate(zip(splits, models)):
        logger.info(f"Predicting OOF -- {i+1}/{len(models)}")
        oof_preds[test_index] = model.predict(
            X_train.loc[test_index, :],
        )
    return oof_preds


def get_avg_preds(X: pd.DataFrame) -> np.ndarray[np.float64]:
    X = X.copy()
    models = pickle.load(MODEL_PATH.open("rb"))
    models = models["cvbooster"].boosters

    data_pipeline = pickle.load(DATA_PIPELINE_PATH.open("rb"))
    X = data_pipeline.transform(X)
    feat_cols = get_feat_columns()
    for col in feat_cols.categorical:
        X[col] = X[col].astype("category")

    preds = np.zeros(len(X))
    for i, model in enumerate(models):
        logger.info(f"Predicting on Test Data -- {i+1}/{len(models)}")
        preds += model.predict(X)
    preds = preds / len(models)
    return preds


def main(prep_data_path: Path):
    target_column = "Premium Amount"

    df = pd.read_feather(prep_data_path)

    X_train = df.drop(columns=[target_column])
    logger.info(f"Train shape: {X_train.shape=}")
    y_train = df.loc[X_train.index, target_column]
    y_train = np.log1p(y_train)

    feat_cols = get_feat_columns()

    data_pipeline = make_boosters_pipeline()
    X_train = data_pipeline.fit_transform(X_train)
    for col in feat_cols.categorical:
        X_train[col] = X_train[col].astype("category")

    logger.info(f"Train shape: {X_train.shape=}")

    train_data = lgb.Dataset(
        data=X_train,
        label=y_train,
        feature_name=X_train.columns.to_list(),
        categorical_feature=feat_cols.categorical,
    )

    folds = get_folds(n_splits=5)

    num_boost_round = 100
    cv_boosters = lgb.cv(
        params=lgbm_params,
        train_set=train_data,
        num_boost_round=num_boost_round,
        folds=folds,
        seed=42,
        return_cvbooster=True,
    )

    cv_boosters["x"] = np.arange(num_boost_round)
    datapoints = pd.DataFrame(dict((k, v) for k, v in cv_boosters.items() if k != "cvbooster"))
    live_dir = Path("dvclive/lgbm/")
    live_dir.mkdir(parents=True, exist_ok=True)
    with Live(dir=str(live_dir)) as live:
        live.log_plot(
            "LGBM CV Loss",
            datapoints,
            x="x",
            y=["valid rmse-mean"],
            template="linear",
            y_label="Booster",
            x_label="RMSLE",
        )

        live.log_metric("lgbm/test-cv-loss", cv_boosters["valid rmse-mean"][-1])
    pickle.dump(data_pipeline, open(DATA_PIPELINE_PATH, "wb"))
    logger.info(f"Data pipeline saved at {DATA_PIPELINE_PATH}")
    pickle.dump(cv_boosters, open(MODEL_PATH, "wb"))
    logger.info(f"Model saved at {MODEL_PATH}")


if __name__ == "__main__":
    typer.run(main)
