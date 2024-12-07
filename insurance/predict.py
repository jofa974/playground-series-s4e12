import typer
from pathlib import Path
import pandas as pd
from insurance.common import OUT_PATH, PREP_DATA_PATH, RAW_DATA_PATH
from insurance.prepare_basic import prepare
import pickle


def main(model: Path):
    """Predicts on test.csv for submission."""
    df_test = pd.read_csv(RAW_DATA_PATH / "test.csv")
    ids = df_test["id"].values
    df_test = prepare(df=df_test)

    pipeline = pickle.load(model.open("rb"))

    predictions = pipeline.predict(df_test)

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
