from dataclasses import dataclass, field, fields
from datetime import timedelta
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

        for c in X.columns:
            X[f"is_{c}_na"] = X[c].isna().astype(int)

        feat_cols = get_feat_columns()

        X[feat_cols.categorical] = X[feat_cols.categorical].fillna("None").astype("string")
        X[feat_cols.numeric] = X[feat_cols.numeric].fillna(-999).astype(float)

        X[self.date_column] = pd.to_datetime(X[self.date_column], format=self.date_format)
        X["year"] = X[self.date_column].dt.year
        X["day"] = X[self.date_column].dt.day
        X["month"] = X[self.date_column].dt.month
        X["day_of_year"] = X[self.date_column].dt.dayofyear
        X["day_of_week"] = X[self.date_column].dt.weekday
        X["sin_day_of_week"] = np.sin(2 * np.pi * X["day_of_week"] / 7)
        X["cos_day_of_week"] = np.cos(2 * np.pi * X["day_of_week"] / 7)

        X["seconds since 1970"] = X[self.date_column].astype("int64") // 10**9

        X["days_passed"] = (X[self.date_column].max() - X[self.date_column]).dt.days

        policy_starts_min = X[self.date_column].min()  # 2019-08-17
        year = policy_starts_min.year

        if policy_starts_min >= pd.Timestamp(f"{year}-01-01"):
            fiscal_year_start = pd.Timestamp(f"{year}-01-01")
        else:
            fiscal_year_start = pd.Timestamp(f"{year-1}-01-01")

        X["time_from_fiscal_year"] = (X[self.date_column] - fiscal_year_start).dt.days
        X["seconds_from_fiscal_year"] = (X[self.date_column] - fiscal_year_start).dt.total_seconds()

        new_date = policy_starts_min - timedelta(days=1)
        X["time_from_first_policy"] = (X[self.date_column] - new_date).dt.days

        X["time_from_first_policy_seconds"] = (X[self.date_column] - new_date).dt.total_seconds()

        X["Days Passed"] = (X[self.date_column].max() - X[self.date_column]).dt.days

        X["claims_vs_duration"] = X["Previous Claims"] / X["Insurance Duration"]
        X["days_from_2019_crisis"] = (X[self.date_column] - pd.Timestamp("2019-01-01")).dt.days
        X["revenue_per_dependent"] = X["Annual Income"] / (X["Number of Dependents"] + 1)
        X["ratio_of_doubts"] = (X["Previous Claims"] + 1) / X["Annual Income"]  # NEW

        X.drop(columns=["time_from_first_policy", self.date_column], inplace=True)

        for col in feat_cols.numeric:
            X[f"cat_{col}"] = X[col].astype("string")

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


def make_pipeline(feat_cols: Features | None = None) -> Pipeline:
    if feat_cols is None:
        feat_cols = get_feat_columns()

    transfo = Pipeline(
        [
            ("transfo", DateTransformer(date_column="Policy Start Date")),
        ]
    )
    columns_engineering = ColumnTransformer(
        transformers=[
            ("date_transformer", transfo, feat_cols.names),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline(
        [
            ("date_preproc", columns_engineering),
        ]
    )
    pipeline.set_output(transform="pandas")
    return pipeline


def get_folds(n_splits: int = 5):
    kf = KFold(n_splits=n_splits, random_state=42, shuffle=True)
    return kf
