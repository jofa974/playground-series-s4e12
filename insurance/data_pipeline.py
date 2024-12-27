from dataclasses import dataclass, field, fields

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
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
        "Annual Income",
    ]
    numeric_log_feat_cols = []
    categorical_feat_cols = [
        "Gender",
        "Marital Status",
        "Education Level",
        "Occupation",
        "Location",
        "Smoking Status",
        "Exercise Frequency",
        "Property Type",
        "Policy Type",
        "Customer Feedback",
    ]
    ordinal_feat_cols = []
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
        X["quarter"] = X[self.date_column].dt.quarter
        X["year_sin"] = np.sin(2 * np.pi * X["year"])
        X["year_cos"] = np.cos(2 * np.pi * X["year"])
        X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
        X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)
        X["day_sin"] = np.sin(2 * np.pi * X["day"] / 31)
        X["day_cos"] = np.cos(2 * np.pi * X["day"] / 31)

        today = pd.Timestamp.now()

        X["policy_age"] = (today - X[self.date_column]).dt.days
        X["policy_age_month"] = X["policy_age"] // 30
        X["policy_age_years"] = X["policy_age"] // 365

        X["claim_rate"] = X["Previous Claims"] / X["policy_age"]
        X["combined_risk"] = X["Health Score"] + X["Credit Score"].fillna(0.0) + X["claim_rate"]

        X["n_nan"] = X.isna().sum(axis=1)

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


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        """
        FrequencyEncoder constructor.
        :param columns: List of columns to apply frequency encoding.
        """
        self.columns = columns
        self.frequency_maps = {}
        self._is_fitted = False

    def fit(self, X, y=None):
        """
        Learn the frequency encoding mapping.
        :param X: Input data
        :param y: Ignored.
        :return: self
        """
        X = pd.DataFrame(X)
        for col in self.columns:
            freq_map = X[col].value_counts(normalize=True)
            self.frequency_maps[col] = freq_map
        self._is_fitted = True
        return self

    def transform(self, X):
        """
        Apply the frequency encoding.
        :param X: Input DataFrame.
        :return: Transformed DataFrame.
        """
        check_is_fitted(self, attributes=["_is_fitted"])
        X = pd.DataFrame(_safe_indexing(X, None))  # Handle different types of inputs safely
        for col in self.columns:
            if col in self.frequency_maps:
                X[f"{col}__freq"] = X[col].map(self.frequency_maps[col])
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

    transfo = Pipeline(
        [
            ("transfo", DateTransformer(date_column="Policy Start Date")),
        ]
    )
    columns_engineering = ColumnTransformer(
        transformers=[
            (
                "date_transformer",
                transfo,
                ["Policy Start Date", "Previous Claims", "Health Score", "Credit Score"],
            ),
            ("cat_transfo", FrequencyEncoder(columns=feat_cols.categorical), feat_cols.categorical),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    num_transformer = Pipeline([("scaler", StandardScaler())])
    cat_imputer = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="unknown"))])
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                num_transformer,
                feat_cols.numeric
                + [
                    "year",
                    "month",
                    "day",
                    "dayofweek",
                    "quarter",
                    "policy_age",
                    "policy_age_month",
                    "policy_age_years",
                    "claim_rate",
                    "combined_risk",
                    "n_nan",
                ]
                + [f"{col}__freq" for col in feat_cols.categorical],
            ),
            ("cat", cat_imputer, feat_cols.categorical),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline(
        [
            ("date_preproc", columns_engineering),
            ("preprocessor", preprocessor),
        ]
    )
    pipeline.set_output(transform="pandas")
    return pipeline


def make_boosters_pipeline(feat_cols: Features | None = None) -> Pipeline:
    return make_pipeline(feat_cols=feat_cols, do_scale=False)


def get_folds(n_splits: int = 5):
    kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)
    return kf
