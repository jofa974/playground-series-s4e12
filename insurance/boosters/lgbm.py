import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from dvclive import Live
from insurance.common import OUT_PATH, TARGET_COLUMN
from insurance.data_pipeline import get_feat_columns, get_folds
from insurance.logger import setup_logger

logger = setup_logger(name="lgbm")


def get_oof_preds(X_train: pd.DataFrame, model_path: Path) -> np.ndarray[np.float64]:
    """Assumes data is transformed."""
    X_train = X_train.copy()
    models = pickle.load(model_path.open("rb"))
    models = models["cvbooster"].boosters

    oof_preds = np.zeros(len(X_train))
    folds = get_folds(n_splits=5)
    splits = folds.split(X_train)
    for i, ((_, test_index), model) in enumerate(zip(splits, models)):
        logger.info(f"Predicting OOF -- {i+1}/{len(models)}")
        oof_preds[test_index] = model.predict(
            X_train.loc[test_index, :],
        )
    return oof_preds


def get_avg_preds(X: pd.DataFrame, model_path: Path) -> np.ndarray[np.float64]:
    """Assumes data is transformed."""
    X = X.copy()
    models = pickle.load(model_path.open("rb"))
    models = models["cvbooster"].boosters

    preds = np.zeros(len(X))
    for i, model in enumerate(models):
        logger.info(f"Predicting on Test Data -- {i+1}/{len(models)}")
        preds += model.predict(X)
    preds = preds / len(models)
    return preds


def train(
    params: dict, model_name: str, layer: int, train_data: pd.DataFrame, test_data: pd.DataFrame
) -> tuple[np.ndarray]:
    feat_cols = get_feat_columns()
    X_train = train_data.drop(columns=[TARGET_COLUMN])
    y_train = train_data[TARGET_COLUMN]

    logger.info(f"Train shape: {X_train.shape=}")
    logger.info(f"Columns: {X_train.columns=}")

    train_data = lgb.Dataset(
        data=X_train,
        label=y_train,
        feature_name=X_train.columns.to_list(),
        categorical_feature=feat_cols.categorical,
    )

    folds = get_folds(n_splits=5)

    logger.info(f"{params=}")
    num_boost_round = 10
    cv_boosters = lgb.cv(
        params=params,
        train_set=train_data,
        num_boost_round=num_boost_round,
        folds=folds,
        seed=42,
        return_cvbooster=True,
    )

    cv_boosters["x"] = np.arange(num_boost_round)
    datapoints = pd.DataFrame(dict((k, v) for k, v in cv_boosters.items() if k != "cvbooster"))
    live_dir = Path(f"dvclive/{model_name}_layer_{layer}/")
    live_dir.mkdir(parents=True, exist_ok=True)
    with Live(dir=str(live_dir)) as live:
        live.log_plot(
            f"LGBM CV Loss {layer}",
            datapoints,
            x="x",
            y=["valid rmse-mean"],
            template="linear",
            y_label="Booster",
            x_label="RMSLE",
        )

        live.log_metric(f"{model_name}/test-cv-loss", cv_boosters["valid rmse-mean"][-1])

    model_path = OUT_PATH / f"models/{model_name}_model_layer_{layer}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(cv_boosters, open(model_path, "wb"))
    logger.info(f"Model saved at {model_path}")

    oof_preds = get_oof_preds(X_train=X_train, model_path=model_path)
    avg_preds = get_avg_preds(X=test_data, model_path=model_path)

    return oof_preds, avg_preds
