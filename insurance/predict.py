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
from insurance.train_xgboost import DATA_PIPELINE_PATH


def main(model: Path):
    """Predicts on test.csv for submission."""
    log_file = datetime.now().strftime("xgboost_predict_log_%Y-%m-%d_%H-%M-%S.log")
    logger = setup_logger(log_file=log_file, name="xgboost predict")

    df_test = pd.read_csv(RAW_DATA_PATH / "test.csv")
    ids = df_test["id"].values
    df_test = prepare(df=df_test)
    df_test.to_feather(PREP_DATA_PATH / "test_prepared.feather")

    feat_cols = get_feat_columns()
    feat_names = feat_cols.names
    df_test = df_test[feat_names]

    data_pipeline = pickle.load(open(DATA_PIPELINE_PATH, "rb"))
    df_test = data_pipeline.transform(df_test)
    for col in feat_cols.categorical:
        df_test[col] = df_test[col].astype("category")

    logger.info(f"Train shape: {df_test.shape=}")
    X_test = xgb.DMatrix(df_test, enable_categorical=True, feature_names=df_test.columns.to_list())

    model = pickle.load(model.open("rb"))

    predictions = np.expm1(model.predict(X_test))

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
