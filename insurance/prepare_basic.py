import pandas as pd
from sklearn.utils import shuffle
import dvc.api
from insurance.common import RAW_DATA_PATH, PREP_DATA_PATH


# Load and preprocess data


def main():
    PREP_DATA_PATH.mkdir(parents=True, exist_ok=True)

    params = dvc.api.params_show()["data"]

    df = pd.read_csv(RAW_DATA_PATH / "train.csv")
    df = shuffle(df, random_state=params["random_state"])
    df["Policy Start Date"] = pd.to_datetime(df["Policy Start Date"]).dt.strftime("%Y%m%d")
    df.to_csv(PREP_DATA_PATH / "prepared_data.csv", index=False)


if __name__ == "__main__":
    main()
