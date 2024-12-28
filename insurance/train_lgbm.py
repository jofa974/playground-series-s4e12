import pickle
from pathlib import Path
from typing import Annotated

import lightgbm as lgb
import numpy as np
import pandas as pd
import typer

from dvclive import Live
from insurance.common import OOF_PREDS_PATH, OUT_PATH, TARGET_COLUMN, PREP_DATA_PATH
from insurance.data_pipeline import get_feat_columns, get_folds, make_pipeline
from insurance.logger import setup_logger

logger = setup_logger(name="lgbm")
DATA_PIPELINE_PATH = OUT_PATH / "data_pipeline_train_lgbm.pkl"

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


def get_oof_preds(X_train: pd.DataFrame, model_path: Path) -> np.ndarray[np.float64]:
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
    X = X.copy()
    models = pickle.load(model_path.open("rb"))
    models = models["cvbooster"].boosters

    preds = np.zeros(len(X))
    for i, model in enumerate(models):
        logger.info(f"Predicting on Test Data -- {i+1}/{len(models)}")
        preds += model.predict(X)
    preds = preds / len(models)
    return preds


def main(
    layer: Annotated[int, typer.Option(help="Stack layer number")],
):
    feat_cols = get_feat_columns()
    if layer == 0:
        df = pd.read_feather(PREP_DATA_PATH / "prepared_data.feather")
        logger.info("Transforming data...")
        feat_cols = get_feat_columns()
        data_pipeline = make_pipeline()
        df = data_pipeline.fit_transform(df)
        for col in feat_cols.categorical:
            df[col] = df[col].astype("category")
        pickle.dump(data_pipeline, open(DATA_PIPELINE_PATH, "wb"))
        logger.info(f"Data pipeline saved at {DATA_PIPELINE_PATH}")
    else:
        df = pd.concat(
            [
                pd.read_feather(OOF_PREDS_PATH / f"xgboost_layer_{layer-1}.feather"),
                pd.read_feather(OOF_PREDS_PATH / f"catboost_layer_{layer-1}.feather")[
                    f"catboost_layer_{layer-1}"
                ],
                pd.read_feather(OOF_PREDS_PATH / f"lgbm_layer_{layer-1}.feather")[
                    f"lgbm_layer_{layer-1}"
                ],
            ],
            axis=1,
        )

    X_train = df.drop(columns=[TARGET_COLUMN])
    logger.info(f"Train shape: {X_train.shape=}")
    y_train = df[TARGET_COLUMN]
    y_train = np.log1p(y_train)

    logger.info(f"Train shape: {X_train.shape=}")
    logger.info(f"Columns: {X_train.columns=}")

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
    live_dir = Path(f"dvclive/lgbm_layer_{layer}/")
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

    model_path = OUT_PATH / f"models/lgbm_model_layer_{layer}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(cv_boosters, open(model_path, "wb"))
    logger.info(f"Model saved at {model_path}")

    preds = get_oof_preds(X_train=X_train, model_path=model_path)
    X_train[f"lgbm_layer_{layer}"] = preds
    X_train[TARGET_COLUMN] = df[TARGET_COLUMN]

    OOF_PREDS_PATH.mkdir(parents=True, exist_ok=True)
    X_train.to_feather(OOF_PREDS_PATH / f"lgbm_layer_{layer}.feather")


if __name__ == "__main__":
    typer.run(main)
