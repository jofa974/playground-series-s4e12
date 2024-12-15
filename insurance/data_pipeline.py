import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from dataclasses import dataclass, field, fields


@dataclass
class Features:
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    ordinal: list[str] = field(default_factory=list)
    numeric_log: list[str] = field(default_factory=list)

    @property
    def names(self):
        to_return = []
        for f in fields(self):
            vals = getattr(self, f.name)
            if vals:
                to_return.extend(vals)
        return to_return


def get_feat_columns():
    numeric_feat_cols = [
        "Age",
        "Health Score",
        "Credit Score",
        "Insurance Duration",
        "Number of Dependents",
        "Vehicle Age",
        "Previous Claims",
        "year",
        "month",
        "day",
        "dayofweek",
    ]
    numeric_log_feat_cols = ["Annual Income"]
    categorical_feat_cols = [
        "Gender",
        "Marital Status",
        "Education Level",
        "Occupation",
        "Location",
        "Policy Type",
        "Customer Feedback",
        "Smoking Status",
        "Exercise Frequency",
        "Property Type",
    ]
    ordinal_feat_cols = []

    feat_cols = Features(
        numeric=numeric_feat_cols,
        numeric_log=numeric_log_feat_cols,
        categorical=categorical_feat_cols,
        ordinal=ordinal_feat_cols,
    )
    return feat_cols


def make_pipeline(feat_cols: Features | None = None) -> Pipeline:
    if feat_cols is None:
        feat_cols = get_feat_columns()

    # Preprocessing pipeline
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    log_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, validate=True, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
        ]
    )

    # No need to OH encode bc XGBoost can deal with that.
    cat_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    ord_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    transformers = []
    if feat_cols.numeric:
        transformers.append(("num", numeric_transformer, feat_cols.numeric))
    if feat_cols.numeric_log:
        transformers.append(("num_log", log_transformer, feat_cols.numeric_log))
    if feat_cols.categorical:
        transformers.append(("cat", cat_transformer, feat_cols.categorical))
    if feat_cols.ordinal:
        transformers.append(("ord", ord_transformer, feat_cols.ordinal))
    preprocessor = ColumnTransformer(
        transformers=transformers, remainder="passthrough", verbose_feature_names_out=False
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
        ]
    )
    pipeline.set_output(transform="pandas")
    return pipeline


def make_xgboost_pipeline(feat_cols: Features | None = None) -> Pipeline:
    if feat_cols is None:
        feat_cols = get_feat_columns()

    # Preprocessing pipeline
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    log_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, validate=True, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
        ]
    )

    # No need to OH encode bc XGBoost can deal with that.
    cat_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    ord_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    transformers = []
    if feat_cols.numeric:
        transformers.append(("num", numeric_transformer, feat_cols.numeric))
    if feat_cols.numeric_log:
        transformers.append(("num_log", log_transformer, feat_cols.numeric_log))
    if feat_cols.categorical:
        transformers.append(("cat", cat_transformer, feat_cols.categorical))
    if feat_cols.ordinal:
        transformers.append(("ord", ord_transformer, feat_cols.ordinal))
    preprocessor = ColumnTransformer(
        transformers=transformers, remainder="passthrough", verbose_feature_names_out=False
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
        ]
    )
    pipeline.set_output(transform="pandas")
    return pipeline
