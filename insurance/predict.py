import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from insurance.common import OUT_PATH, RAW_DATA_PATH
from insurance.prepare_basic import prepare


def main(model: Path):
    """Predicts on test.csv for submission."""
    df_test = pd.read_csv(RAW_DATA_PATH / "test.csv")
    ids = df_test["id"].values
    df_test = prepare(df=df_test)

    pipeline = pickle.load(model.open("rb"))

    predictions = np.expm1(pipeline.predict(df_test))

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
