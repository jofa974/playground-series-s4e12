import pickle
from pathlib import Path

import dvc.api
import numpy as np
import pandas as pd
from dvclive import Live
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor

from insurance.common import OUT_PATH, PREP_DATA_PATH

model_label = Path(__file__).stem.split("_")[-1]

MODEL_PATH = OUT_PATH / f"models/model_{model_label}.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


def main():
    params = dvc.api.params_show()["train"]

    target_column = "Premium Amount"

    df = pd.read_csv(PREP_DATA_PATH / "prepared_data.csv")
    features = df.drop(columns=[target_column])
    labels = df[target_column]
    numeric_features = [
        "Age",
        "Number of Dependents",
        "Vehicle Age",
        "Previous Claims",
        "Health Score",
        "Credit Score",
        "Insurance Duration",
    ]
    numeric_log_features = ["Annual Income"]
    categorical_features = features.select_dtypes(include=["object", "category"]).columns.tolist()
    ordinal_features = []

    feat_cols = numeric_features + numeric_log_features + categorical_features + ordinal_features

    # Preprocessing pipeline
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    log_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, validate=True)),
            ("scaler", StandardScaler()),
        ]
    )

    oh_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    ord_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            (
                "num_log",
                log_transformer,
                numeric_log_features,
            ),
            ("oh", oh_transformer, categorical_features),
            ("ord", ord_transformer, ordinal_features),
        ],
        remainder="drop",
    )

    model = XGBRegressor(**params["xgboost"])
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

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
            pipeline.fit(X_train, np.log1p(y_train))

            # Predict and evaluate
            y_pred = np.expm1(pipeline.predict(X_test))
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
    re_pipeline = clone(pipeline)
    re_pipeline.fit(X_train, np.log1p(y_train))
    pickle.dump(re_pipeline, (MODEL_PATH).open("wb"))
    live.log_artifact(MODEL_PATH)


if __name__ == "__main__":
    main()
