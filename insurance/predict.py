import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import typer
import xgboost as xgb
from insurance.common import OUT_PATH, RAW_DATA_PATH
from insurance.prepare_basic import prepare
from insurance.data_pipeline import make_pipeline


def main(model: Path):
    """Predicts on test.csv for submission."""
    df_test = pd.read_csv(RAW_DATA_PATH / "test.csv")
    ids = df_test["id"].values
    df_test = prepare(df=df_test)

    categorical_columns = df_test.select_dtypes(include=["object", "category"]).columns
    for col in categorical_columns:
        df_test[col] = df_test[col].astype("category")
    _, feat_cols = make_pipeline(features=df_test)

    df_test = df_test[feat_cols]

    X_test = xgb.DMatrix(
        df_test,
        enable_categorical=True,
    )

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
