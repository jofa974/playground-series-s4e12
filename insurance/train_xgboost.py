import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import KFold

from insurance.common import OUT_PATH
from insurance.data_pipeline import get_feat_columns
from insurance.logger import setup_logger
from insurance.tune_xgboost import BEST_PARAMS_PATH, DATA_PIPELINE_PATH
import typer

model_label = Path(__file__).stem.split("_")[-1]

MODEL_PATH = OUT_PATH / "models/"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


def main(prep_data_path: Path):
    log_file = datetime.now().strftime("xgboost_tune_log_%Y-%m-%d_%H-%M-%S.log")
    logger = setup_logger(log_file=log_file, name="xgboost trainer")

    target_column = "Premium Amount"

    df = pd.read_feather(prep_data_path)

    df["Policy Start Date"] = pd.to_datetime(df["Policy Start Date"])
    df["year"] = df["Policy Start Date"].dt.year
    df["month"] = df["Policy Start Date"].dt.month
    df["day"] = df["Policy Start Date"].dt.day
    df["dayofweek"] = df["Policy Start Date"].dt.dayofweek
    df = df.drop(columns=["Policy Start Date"])

    features = df.drop(columns=[target_column])
    labels = df[target_column]

    feat_cols = get_feat_columns()
    feat_names = feat_cols.names

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    with open(BEST_PARAMS_PATH, "r") as file:
        params = yaml.safe_load(file)

    score = 0
    models = {}
    for fold, (train_idx, test_idx) in enumerate(kf.split(features[feat_names])):
        logger.info(f"Fold {fold + 1}")
        X_train, X_test = (
            features[feat_names].iloc[train_idx],
            features[feat_names].iloc[test_idx],
        )
        y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]

        # Fit the pipeline
        data_pipeline = pickle.load(open(str(DATA_PIPELINE_PATH) + f"_fold_{fold}.pkl", "rb"))
        X_train = data_pipeline.transform(X_train)
        for col in feat_cols.categorical:
            X_train[col] = X_train[col].astype("category")
        dtrain = xgb.DMatrix(
            X_train, label=np.log1p(y_train), enable_categorical=True, feature_names=feat_names
        )

        # Predict and evaluate
        X_test = data_pipeline.transform(X_test)
        for col in feat_cols.categorical:
            X_test[col] = X_test[col].astype("category")
        dtest = xgb.DMatrix(
            X_test, label=np.log1p(y_test), enable_categorical=True, feature_names=feat_names
        )

        bst = xgb.train(params, dtrain)

        # Predict and evaluate
        y_pred = np.expm1(bst.predict(dtest))
        rmsle = root_mean_squared_log_error(y_test, y_pred)
        logger.info(f" !!! Fold {fold+1} !!! Root Mean Squared Logarithmic Error: {rmsle:.4f}")
        score += rmsle
        models[f"model_{fold+1}"] = bst

    pickle.dump(models, open(MODEL_PATH / "xgboost_folds.pkl", "wb"))

    avg = score / n_splits
    logger.info(f"Average RMSLE across folds: {avg}")


if __name__ == "__main__":
    typer.run(main)
