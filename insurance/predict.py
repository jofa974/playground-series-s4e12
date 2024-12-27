from datetime import datetime
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import typer
import xgboost as xgb

from insurance.common import OUT_PATH, PREP_DATA_PATH, RAW_DATA_PATH
from insurance.data_pipeline import get_feat_columns
from insurance.logger import setup_logger
from insurance.prepare_basic import prepare
from insurance.train_ensemble import DATA_PIPELINE_PATH as ENSEMBLE_DATA_PIPELINE
from insurance.train_ensemble import MODEL_PATH as ENSEMBLE_MODEL_PATH
from insurance.train_xgboost import get_avg_preds as xgboost_avg_preds
from insurance.train_catboost import get_avg_preds as catboost_avg_preds


log_file = datetime.now().strftime("ensemble_predict_log_%Y-%m-%d_%H-%M-%S.log")
logger = setup_logger(log_file=log_file, name="ensemble predict")

PREV_LAYER_AVG_PREDS = {"xgboost": xgboost_avg_preds, "catboost": catboost_avg_preds}


def get_avg_preds(X: pd.DataFrame) -> np.ndarray[np.float64]:
    models = pickle.load(ENSEMBLE_MODEL_PATH.open("rb"))

    data_pipeline = pickle.load(ENSEMBLE_DATA_PIPELINE.open("rb"))
    X = data_pipeline.transform(X)
    feat_cols = get_feat_columns()
    for col in feat_cols.categorical:
        X[col] = X[col].astype("category")

    preds = np.zeros(len(X))
    for i, model in enumerate(models):
        logger.info(f"Predicting on Test Data -- {i+1}/{len(models)}")
        data = xgb.DMatrix(
            data=X,
            enable_categorical=True,
            feature_names=X.columns.to_list(),
        )
        preds += model.predict(data=data)
    preds = preds / len(models)
    return preds


def main():
    """Predicts on test.csv for submission."""
    df_test = pd.read_csv(RAW_DATA_PATH / "test.csv")
    ids = df_test["id"].values
    df_test = prepare(df=df_test)
    df_test.to_feather(PREP_DATA_PATH / "test_prepared.feather")

    columns = df_test.columns

    for model, avg_func in PREV_LAYER_AVG_PREDS.items():
        logger.info(f"{model} AVG predictions...")
        # Ensure that avg predictions of a model do not use previous model avg preds.
        df_test[f"{model}_preds"] = avg_func(X=df_test[columns])

    data_pipeline = pickle.load(open(ENSEMBLE_DATA_PIPELINE, "rb"))
    df_test = data_pipeline.transform(df_test)

    logger.info(f"Test shape: {df_test.shape=}")
    X_test = df_test

    ensemble_model = pickle.load(ENSEMBLE_MODEL_PATH.open("rb"))
    predictions = np.zeros(len(df_test))
    for model in ensemble_model:
        predictions += np.expm1(model.predict(X_test)) / len(ensemble_model)

    # Prepare submission file
    submission = pd.DataFrame(
        {
            "id": ids,
            "Premium Amount": predictions,
        }
    )
    pred_file = OUT_PATH / "preds.csv"
    submission.to_csv(pred_file, index=False)
    print(f"Submission file saved to {pred_file}")


if __name__ == "__main__":
    typer.run(main)
