import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler


def make_pipeline(features: pd.DataFrame) -> tuple[Pipeline, list[str]]:
    numeric_features = [
        "Age",
        "Health Score",
        "Credit Score",
        "Insurance Duration",
        "Number of Dependents",
        "Vehicle Age",
    ]
    numeric_log_features = ["Annual Income"]
    categorical_features = features.select_dtypes(include=["object", "category"]).columns.tolist()
    ordinal_features = [
        "Previous Claims",
    ]

    feat_cols = numeric_features + numeric_log_features + categorical_features + ordinal_features

    # Preprocessing pipeline
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            # ("scaler", StandardScaler()),
        ]
    )
    log_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            # ("log", FunctionTransformer(np.log1p, validate=True)),
            # ("scaler", StandardScaler()),
        ]
    )

    oh_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    ord_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
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

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
        ]
    )
    pipeline.set_output(transform="pandas")
    return pipeline, feat_cols
