import pickle
from pathlib import Path
from typing import Annotated

import dvc.api
import lightgbm as lgb
import numpy as np
import pandas as pd
import typer

from dvclive import Live
from insurance.common import OOF_PREDS_PATH, OUT_PATH, TARGET_COLUMN
from insurance.data_pipeline import get_feat_columns, get_folds
from insurance.logger import setup_logger
from insurance.data_booster import get_booster_training_data

logger = setup_logger(name="lgbm")
DATA_PIPELINE_PATH = OUT_PATH / "data_pipeline_train_lgbm.pkl"


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


def main(
    layer: Annotated[int, typer.Option(help="Stack layer number")],
    model_name: Annotated[str, typer.Option(help="Model name")],
):
    params = dvc.api.params_show()
    lgbm_params = params[f"layer_{layer}"][model_name]

    X_train, y_train = get_booster_training_data(layer=layer, model_name=model_name)
    feat_cols = get_feat_columns()

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

        live.log_metric(f"{model_name}/test-cv-loss", cv_boosters["valid rmse-mean"][-1])

    model_path = OUT_PATH / f"models/{model_name}_model_layer_{layer}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(cv_boosters, open(model_path, "wb"))
    logger.info(f"Model saved at {model_path}")

    preds = get_oof_preds(X_train=X_train, model_path=model_path)
    X_train[f"{model_name}_layer_{layer}"] = preds
    X_train[TARGET_COLUMN] = np.expm1(y_train)

    OOF_PREDS_PATH.mkdir(parents=True, exist_ok=True)
    X_train.to_feather(OOF_PREDS_PATH / f"{model_name}_layer_{layer}.feather")


if __name__ == "__main__":
    typer.run(main)
