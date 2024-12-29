import pandas as pd
from insurance.logger import setup_logger
from insurance.common import RAW_DATA_PATH, PREP_DATA_PATH, OUT_PATH, TARGET_COLUMN
from insurance.data_pipeline import get_feat_columns, make_pipeline
import numpy as np
import pickle

logger = setup_logger(name="Train/Test Data preparation")


def main():
    PREP_DATA_PATH.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_csv(RAW_DATA_PATH / "train.csv")
    target = df_train[TARGET_COLUMN]
    df_train = df_train.drop(columns=["id", TARGET_COLUMN])
    df_train["Policy Start Date"] = (
        pd.to_datetime(df_train["Policy Start Date"]).dt.strftime("%Y%m%d").astype(np.int64)
    )
    logger.info("Transforming training data...")
    feat_cols = get_feat_columns()
    data_pipeline = make_pipeline()
    df_train = data_pipeline.fit_transform(df_train)
    for col in feat_cols.categorical:
        df_train[col] = df_train[col].astype("category")

    df_train[TARGET_COLUMN] = np.log1p(target)
    logger.info(f"{TARGET_COLUMN} transformed with log1p")

    data_pipeline_path = OUT_PATH / "data_pipeline_train.pkl"
    pickle.dump(data_pipeline, open(data_pipeline_path, "wb"))
    logger.info(f"Data pipeline saved at {data_pipeline_path}")

    train_data_path = PREP_DATA_PATH / "train_data.feather"
    df_train.to_feather(train_data_path)
    logger.info(f"Training data saved to {train_data_path}...")

    df_test = pd.read_csv(RAW_DATA_PATH / "test.csv")
    df_test = df_test.drop(columns=["id"])
    df_test["Policy Start Date"] = (
        pd.to_datetime(df_test["Policy Start Date"]).dt.strftime("%Y%m%d").astype(np.int64)
    )
    logger.info("Transforming test data...")
    df_test = data_pipeline.transform(df_test)
    for col in feat_cols.categorical:
        df_test[col] = df_test[col].astype("category")
    test_data_path = PREP_DATA_PATH / "test_data.feather"
    df_test.to_feather(test_data_path)
    logger.info(f"Test data saved to {test_data_path}...")


if __name__ == "__main__":
    main()
