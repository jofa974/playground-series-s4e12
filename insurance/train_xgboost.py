import pickle
from datetime import datetime
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
import xgboost as xgb

from dvclive import Live
from insurance.common import OUT_PATH, OOF_PREDS_PATH
from insurance.data_pipeline import get_feat_columns, get_folds, make_pipeline
from insurance.logger import setup_logger

logger = setup_logger(name="xgboost")
DATA_PIPELINE_PATH = OUT_PATH / "data_pipeline_train_xgboost.pkl"


xgb_params = {
    "n_estimators": 1000,
    "learning_rate": 0.01,
    "max_depth": 10,
    "reg_lambda": 100,
    "reg_alpha": 0.1,
    "random_state": 42,
    # "num_leaves": None,
    "min_child_weight": 1,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "device": "cuda",
    "tree_method": "hist",
    "verbosity": 0,
    "enable_categorical": True,
}


class SaveBestModel(xgb.callback.TrainingCallback):
    def __init__(self, cvboosters):
        self._cvboosters = cvboosters

    def after_training(self, model):
        self._cvboosters[:] = [cvpack.bst for cvpack in model.cvfolds]
        return model


def get_oof_preds(X_train: pd.DataFrame, model_path: Path) -> np.ndarray[np.float64]:
    X_train = X_train.copy()
    models = pickle.load(model_path.open("rb"))

    oof_preds = np.zeros(len(X_train))
    folds = get_folds(n_splits=5)
    splits = folds.split(X_train)
    for i, ((_, test_index), model) in enumerate(zip(splits, models)):
        logger.info(f"Predicting OOF -- {i+1}/{len(models)}")
        data = xgb.DMatrix(
            data=X_train.loc[test_index, :],
            enable_categorical=True,
            feature_names=X_train.columns.to_list(),
        )
        oof_preds[test_index] = model.predict(data=data)
    return oof_preds


def get_avg_preds(X: pd.DataFrame, model_path: Path) -> np.ndarray[np.float64]:
    X = X.copy()
    models = pickle.load(model_path.open("rb"))

    data_pipeline = pickle.load(DATA_PIPELINE_PATH.open("rb"))
    X = data_pipeline.transform(X)
    feat_cols = get_feat_columns()
    for col in feat_cols.categorical:
        X[col] = X[col].astype("category")

    preds = np.zeros(len(X))
    for i, model in enumerate(models):
        print(f"Predicting on Test Data -- {i+1}/{len(models)}")
        data = xgb.DMatrix(
            data=X,
            enable_categorical=True,
            feature_names=X.columns.to_list(),
        )
        preds += model.predict(data=data)
    preds = preds / len(models)
    return preds


def main(
    input_data_path: Annotated[Path, typer.Option(help="Input data path")],
    layer: Annotated[int, typer.Option(help="Stack layer number")],
):
    target_column = "Premium Amount"

    df = pd.read_feather(input_data_path)

    X_train = df.drop(columns=[target_column])
    logger.info(f"Train shape: {X_train.shape=}")
    y_train = df[target_column]
    y_train = np.log1p(y_train)

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

    dtrain = xgb.DMatrix(
        X_train,
        label=y_train,
        enable_categorical=True,
        feature_names=X_train.columns.to_list(),
    )

    folds = get_folds(n_splits=5)

    cv_boosters = []
    history = xgb.cv(
        xgb_params,
        dtrain,
        num_boost_round=500,
        early_stopping_rounds=10,
        callbacks=[SaveBestModel(cv_boosters)],
        folds=folds,
        verbose_eval=True,
    )
    live_dir = Path(f"dvclive/xgboost_layer_{layer}/")
    live_dir.mkdir(parents=True, exist_ok=True)
    with Live(dir=str(live_dir)) as live:
        live.log_plot(
            f"XGBoost CV Loss Layer {layer}",
            history,
            x="booster",
            y=["train-rmse-mean", "test-rmse-mean"],
            template="linear",
            y_label="Booster",
            x_label="RMSLE",
        )

        live.log_metric(f"xgboost_layer_{layer}/train-cv-loss", history["train-rmse-mean"].iloc[-1])
        live.log_metric(f"xgboost_layer_{layer}/test-cv-loss", history["test-rmse-mean"].iloc[-1])

    model_path = OUT_PATH / f"models/xgboost_model_layer_{layer}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(cv_boosters, open(model_path, "wb"))
    logger.info(f"Model saved at {model_path}")

    preds = get_oof_preds(X_train=X_train, model_path=model_path)
    X_train[f"xgboost_layer_{layer}"] = preds
    X_train[target_column] = df[target_column]

    OOF_PREDS_PATH.mkdir(parents=True, exist_ok=True)
    X_train.to_feather(OOF_PREDS_PATH / f"xgboost_layer_{layer}.feather")


if __name__ == "__main__":
    typer.run(main)
