import pickle
from pathlib import Path
from typing import Annotated

import dvc.api
import numpy as np
import pandas as pd
import typer
import xgboost as xgb

from dvclive import Live
from insurance.common import OOF_PREDS_PATH, OUT_PATH, PREP_DATA_PATH, TARGET_COLUMN
from insurance.data_pipeline import get_feat_columns, get_folds, make_pipeline
from insurance.logger import setup_logger

logger = setup_logger(name="xgboost")
DATA_PIPELINE_PATH = OUT_PATH / "data_pipeline_train_xgboost.pkl"


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
    layer: Annotated[int, typer.Option(help="Stack layer number")],
    model_name: Annotated[str, typer.Option(help="Model name")],
):
    params = dvc.api.params_show()
    xgb_params = params[f"layer_{layer}"][model_name]

    logger.info(f"{xgb_params=}")
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
        prev_layer_models = list(params[f"layer_{layer-1}"].keys())
        df = pd.read_feather(OOF_PREDS_PATH / f"{prev_layer_models[0]}_layer_{layer-1}.feather")
        prev_oofs = [df]
        for i in range(1, len(prev_layer_models)):
            prev_oofs.append(
                pd.read_feather(OOF_PREDS_PATH / f"{prev_layer_models[i]}_layer_{layer-1}.feather")[
                    f"{prev_layer_models[i]}_layer_{layer-1}"
                ],
            )
        df = pd.concat(
            prev_oofs,
            axis=1,
        )

    X_train = df.drop(columns=[TARGET_COLUMN])
    logger.info(f"Train shape: {X_train.shape=}")
    y_train = df[TARGET_COLUMN]
    y_train = np.log1p(y_train)

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

    preds = get_oof_preds(X_train=X_train, model_path=model_path)
    X_train[f"{model_name}_layer_{layer}"] = preds
    X_train[TARGET_COLUMN] = df[TARGET_COLUMN]

    OOF_PREDS_PATH.mkdir(parents=True, exist_ok=True)
    X_train.to_feather(OOF_PREDS_PATH / f"{model_name}_layer_{layer}.feather")


if __name__ == "__main__":
    typer.run(main)
