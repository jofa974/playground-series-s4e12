import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from dvclive import Live
from insurance.common import OUT_PATH, TARGET_COLUMN
from insurance.data_pipeline import get_folds
from insurance.logger import setup_logger

logger = setup_logger(name="xgboost")


class SaveBestModel(xgb.callback.TrainingCallback):
    def __init__(self, cvboosters):
        self._cvboosters = cvboosters

    def after_training(self, model):
        self._cvboosters[:] = [cvpack.bst for cvpack in model.cvfolds]
        return model


def get_oof_preds(X_train: pd.DataFrame, model_path: Path) -> np.ndarray[np.float64]:
    """Assumes data is transformed."""
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
    """Assumes data is transformed."""
    X = X.copy()
    models = pickle.load(model_path.open("rb"))
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


def train(
    params: dict, model_name: str, layer: int, train_data: pd.DataFrame, test_data: pd.DataFrame
) -> tuple[np.ndarray]:
    X_train = train_data.drop(columns=[TARGET_COLUMN])
    y_train = train_data[TARGET_COLUMN]

    logger.info(f"Train shape: {X_train.shape=}")
    logger.info(f"Columns: {X_train.columns=}")

    dtrain = xgb.DMatrix(
        X_train,
        label=y_train,
        enable_categorical=True,
        feature_names=X_train.columns.to_list(),
    )

    folds = get_folds(n_splits=5)

    logger.info(f"{params=}")
    cv_boosters = []
    history = xgb.cv(
        params,
        dtrain,
        num_boost_round=params.get("n_estimators", 10),
        early_stopping_rounds=10,
        callbacks=[SaveBestModel(cv_boosters)],
        folds=folds,
        verbose_eval=200,
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

        live.log_metric(f"{model_name}/train-cv-loss", history["train-rmse-mean"].iloc[-1])
        live.log_metric(f"{model_name}/test-cv-loss", history["test-rmse-mean"].iloc[-1])

    model_path = OUT_PATH / f"models/{model_name}_model_layer_{layer}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(cv_boosters, open(model_path, "wb"))
    logger.info(f"Model saved at {model_path}")

    oof_preds = get_oof_preds(X_train=X_train, model_path=model_path)
    avg_preds = get_avg_preds(X=test_data, model_path=model_path)

    return oof_preds, avg_preds
