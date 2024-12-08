import pickle
from pathlib import Path

import dvc.api
import numpy as np
import pandas as pd
import xgboost as xgb
from dvclive import Live
from sklearn.base import clone
from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import KFold

from insurance.common import OUT_PATH, PREP_DATA_PATH
from insurance.data_pipeline import make_pipeline

model_label = Path(__file__).stem.split("_")[-1]

MODEL_PATH = OUT_PATH / f"models/model_{model_label}.pkl"
DATA_PIPELINE_PATH = MODEL_PATH.parent / f"data_pipeline_{model_label}.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


def main():
    params = dvc.api.params_show()["train"]

    target_column = "Premium Amount"

    df = pd.read_csv(PREP_DATA_PATH / "prepared_data.csv")
    features = df.drop(columns=[target_column])
    labels = df[target_column]

    data_pipeline, feat_cols = make_pipeline(features=features)

    # Stratified K-Fold Cross-Validation
    n_splits = params["n_splits"]
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    with Live(OUT_PATH, resume=True) as live:
        rmsle_scores = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(features[feat_cols])):
            print(f"Fold {fold + 1}")
            X_train, X_test = (
                features[feat_cols].iloc[train_idx],
                features[feat_cols].iloc[test_idx],
            )
            y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]
            print(f"{X_train.shape=}")
            # Fit the pipeline
            X_train = data_pipeline.fit_transform(X_train)
            dtrain = xgb.DMatrix(X_train, label=np.log1p(y_train))
            bst = xgb.train(params["xgboost"], dtrain)

            # Predict and evaluate
            X_test = data_pipeline.transform(X_test)
            dvalid = xgb.DMatrix(X_test, label=np.log1p(y_test))
            y_pred = np.expm1(bst.predict(dvalid))
            rmsle = root_mean_squared_log_error(y_test, y_pred)
            rmsle_scores.append(rmsle)
            print(f"Root Mean Squared Logarithmic Error: {rmsle:.4f}")
            live.log_metric(f"rmsle/{fold}/{model_label}", rmsle)

        # Overall performance
        average_rmsle = np.mean(rmsle_scores)
        live.log_metric(f"rmsle/average/{model_label}", average_rmsle)
        live.next_step()
        print(
            f"Average Mean Squared Logarithmic Error across {n_splits} folds: {average_rmsle:.4f}"
        )

    # Re-train on entire dataset
    X_train = features[feat_cols]
    y_train = labels
    re_pipeline = clone(data_pipeline)
    X_train = re_pipeline.fit_transform(X_train)
    dtrain = xgb.DMatrix(X_train, label=np.log1p(y_train))
    bst = xgb.train(params["xgboost"], dtrain)
    pickle.dump(re_pipeline, (DATA_PIPELINE_PATH).open("wb"))
    pickle.dump(bst, MODEL_PATH.open("wb"))
    live.log_artifact(MODEL_PATH)


if __name__ == "__main__":
    main()
