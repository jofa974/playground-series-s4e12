import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
import catboost as cb

from dvclive import Live
from insurance.common import OUT_PATH
from insurance.data_pipeline import get_feat_columns, make_boosters_pipeline, get_folds
from insurance.logger import setup_logger

MODEL_PATH = OUT_PATH / "models/catboost_model.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

DATA_PIPELINE_PATH = OUT_PATH / "data_pipeline_train_catboost.pkl"

log_file = datetime.now().strftime("catboost_train_log_%Y-%m-%d_%H-%M-%S.log")
logger = setup_logger(log_file=log_file, name="catboost trainer")

catboost_params = {
    "loss_function": "RMSE",
    "iterations": 1000,
    "learning_rate": 0.5,
    "devices": [0],
    "task_type": "GPU",
    "depth": 9,
    "boosting_type": "Ordered",
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8106434906203719,
}


def get_oof_preds(X_train: pd.DataFrame) -> np.ndarray[np.float64]:
    models = pickle.load(MODEL_PATH.open("rb"))

    data_pipeline = pickle.load(DATA_PIPELINE_PATH.open("rb"))
    X_train = data_pipeline.transform(X_train)
    feat_cols = get_feat_columns()
    for col in feat_cols.categorical:
        X_train[col] = X_train[col].astype("category")

    oof_preds = np.zeros(len(X_train))
    folds = get_folds(df_train=X_train)
    splits = folds.split(X_train)
    for i, ((_, test_index), model) in enumerate(zip(splits, models)):
        logger.info(f"Predicting OOF -- {i+1}/{len(models)}")
        oof_preds[test_index] = model.predict(data=X_train.loc[test_index, :])
    return oof_preds


def main(prep_data_path: Path):
    target_column = "Premium Amount"

    df = pd.read_feather(prep_data_path)

    X_train = df.drop(columns=[target_column])
    logger.info(f"Train shape: {X_train.shape=}")
    X_train = X_train.loc[
        pd.to_datetime(X_train["Policy Start Date"], format="%Y%m%d").dt.year >= 2020
    ]
    logger.info(f"Train shape: {X_train.shape=}")
    y_train = df.loc[X_train.index, target_column]
    y_train = np.log1p(y_train)

    feat_cols = get_feat_columns()

    data_pipeline = make_boosters_pipeline()
    X_train = data_pipeline.fit_transform(X_train)
    for col in feat_cols.categorical:
        X_train[col] = X_train[col].astype("category")

    print(f"Train shape: {X_train.shape=}")
    train_pool = cb.Pool(
        data=X_train, label=y_train, cat_features=feat_cols.categorical, has_header=True
    )

    folds = get_folds(df_train=X_train, n_splits=5)

    history, cv_boosters = cb.cv(
        pool=train_pool,
        params=catboost_params,
        folds=folds,
        as_pandas=True,
        verbose=False,
        type="Classical",
        return_models=True,
    )

    history = history.reset_index()
    history["index"] = history["index"] + 1
    history = history.rename(columns={"index": "booster"})

    live_dir = Path("dvclive/catboost/")
    live_dir.mkdir(parents=True, exist_ok=True)
    with Live(dir=str(live_dir)) as live:
        live.log_plot(
            "catboost CV Loss",
            history,
            x="booster",
            y=["train-RMSE-mean", "test-RMSE-mean"],
            template="linear",
            y_label="Booster",
            x_label="RMSLE",
        )

        live.log_metric("catboost/train-cv-loss", history["train-RMSE-mean"].iloc[-1])
        live.log_metric("catboost/test-cv-loss", history["test-RMSE-mean"].iloc[-1])
    pickle.dump(data_pipeline, open(DATA_PIPELINE_PATH, "wb"))
    logger.info(f"Data pipeline saved at {DATA_PIPELINE_PATH}")
    pickle.dump(cv_boosters, open(MODEL_PATH, "wb"))
    logger.info(f"Model saved at {MODEL_PATH}")


if __name__ == "__main__":
    typer.run(main)
