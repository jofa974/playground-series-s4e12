from dataclasses import dataclass, field, fields

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.utils.metaestimators import _safe_indexing
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import KFold


@dataclass
class Features:
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    ordinal: list[str] = field(default_factory=list)
    numeric_log: list[str] = field(default_factory=list)
    date_time: list[str] = field(default_factory=list)

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
    ]
    numeric_log_feat_cols = ["Annual Income"]
    categorical_feat_cols = [
        "Gender",
        "Marital Status",
        "Education Level",
        "Occupation",
        "Location",
        "Smoking Status",
        "Exercise Frequency",
        "Property Type",
    ]
    ordinal_feat_cols = ["Policy Type", "Customer Feedback"]
    date_time_cols = ["Policy Start Date"]

    feat_cols = Features(
        numeric=numeric_feat_cols,
        numeric_log=numeric_log_feat_cols,
        categorical=categorical_feat_cols,
        ordinal=ordinal_feat_cols,
        date_time=date_time_cols,
    )
    return feat_cols


class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, date_column, date_format="%Y%m%d"):
        self.date_column = date_column
        self.date_format = date_format
        self._is_fitted = False

    def fit(self, X, y=None):
        # Store feature names for compatibility with set_output
        self.feature_names_in_ = X.columns if hasattr(X, "columns") else None
        self._is_fitted = True
        return self

    def transform(self, X):
        # Check if the transformer has been fitted
        check_is_fitted(self, attributes=["_is_fitted"])
        X = pd.DataFrame(_safe_indexing(X, None))  # Handle different types of inputs safely

        # Convert the date column to datetime
        X[self.date_column] = pd.to_datetime(X[self.date_column], format=self.date_format)
        # Extract year, month, day, and dayofweek
        X["year"] = X[self.date_column].dt.year
        X["month"] = X[self.date_column].dt.month
        X["day"] = X[self.date_column].dt.day
        X["dayofweek"] = X[self.date_column].dt.dayofweek
        X["Avg Claims Per Month Since Policy Started"] = X["Previous Claims"] / X[
            self.date_column
        ].apply(lambda x: (2024 - x.year) * 12 + 12 - x.month)
        # Drop the original date column
        X = X.drop(columns=[self.date_column])

        # Return a pandas DataFrame with consistent column names
        return X

    def set_output(self, transform="default"):
        """
        Enable output configuration.
        """
        self._output_config = transform
        return self

    def _more_tags(self):
        """
        Add 'requires_fit' tag for compatibility with set_output.
        """
        return {"requires_fit": True}


class ClaimsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._is_fitted = False

    def fit(self, X, y=None):
        # Store feature names for compatibility with set_output
        self.feature_names_in_ = X.columns if hasattr(X, "columns") else None
        self._is_fitted = True
        return self

    def transform(self, X):
        # Check if the transformer has been fitted
        check_is_fitted(self, attributes=["_is_fitted"])
        X = pd.DataFrame(_safe_indexing(X, None))  # Handle different types of inputs safely

        # Convert the date column to datetime
        X.loc[X["Previous Claims"].isna(), "Previous Claims"] = X["Previous Claims"].median()
        start_date = pd.to_datetime(X["Policy Start Date"], format="%Y%m%d")
        X["Avg Claims Per Month Since Policy Started"] = X["Previous Claims"] / start_date.apply(
            lambda x: (2024 - x.year) * 12 + 12 - x.month
        )
        # Return a pandas DataFrame with consistent column names
        return X

    def set_output(self, transform="default"):
        """
        Enable output configuration.
        """
        self._output_config = transform
        return self

    def _more_tags(self):
        """
        Add 'requires_fit' tag for compatibility with set_output.
        """
        return {"requires_fit": True}


def make_pipeline(feat_cols: Features | None = None, *, do_scale=True) -> Pipeline:
    if feat_cols is None:
        feat_cols = get_feat_columns()

    # Preprocessing pipeline
    trans = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if do_scale:
        trans.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(trans)

    trans = [
        ("imputer", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, validate=True, feature_names_out="one-to-one")),
    ]
    if do_scale:
        trans.append(("scaler", StandardScaler()))
    log_transformer = Pipeline(trans)

    cat_transformer = Pipeline(
        [("imputer", SimpleImputer(strategy="constant", fill_value="unknown"))]
    )

    policy_ord_transformer = Pipeline(
        [
            # ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[["Basic", "Comprehensive", "Premium"]],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )
    feedback_ord_transformer = Pipeline(
        [
            # ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[["Poor", "Average", "Good"]],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )
    date_transformer = Pipeline(
        [
            ("date_transformer", DateTransformer(date_column="Policy Start Date")),
        ]
    )

    # claims_transformer = Pipeline(
    #     [
    #         ("claims_transformer", ClaimsTransformer()),
    #     ]
    # )

    transformers = []
    # if feat_cols.numeric:
    #     transformers.append(("num", numeric_transformer, feat_cols.numeric))
    # if feat_cols.numeric_log:
    #     transformers.append(("num_log", log_transformer, feat_cols.numeric_log))
    if feat_cols.categorical:
        transformers.append(("cat", cat_transformer, feat_cols.categorical))
    # if feat_cols.ordinal:
    #     transformers.append(("ord", ord_transformer, feat_cols.ordinal))
    if feat_cols.date_time:
        transformers.append(("date", date_transformer, ["Policy Start Date", "Previous Claims"]))

    # transformers.append(("avg_claims", claims_transformer, ["Previous Claims", "Insurance Duration"]))
    transformers.append(("policy_type", policy_ord_transformer, ["Policy Type"]))
    transformers.append(("customer_feedback", feedback_ord_transformer, ["Customer Feedback"]))

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


def make_boosters_pipeline(feat_cols: Features | None = None) -> Pipeline:
    return make_pipeline(feat_cols=feat_cols, do_scale=False)


def get_folds(df_train: pd.DataFrame, labels: pd.Series, n_splits: int = 5):
    kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)
    return kf.split(X=df_train)
