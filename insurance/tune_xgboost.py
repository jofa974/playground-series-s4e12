import pickle
from pathlib import Path

import dvc.api
import numpy as np
import optuna
import pandas as pd
from dvclive.optuna import DVCLiveCallback
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor

from insurance.common import OUT_PATH, PREP_DATA_PATH

model_label = Path(__file__).stem.split("_")[-1]

DATA_PIPELINE_PATH = OUT_PATH / f"data_pipeline_tune_{model_label}.pkl"
MODEL_PATH = OUT_PATH / f"models/model_{model_label}.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

TUNE_PATH = OUT_PATH / f"tune_{model_label}"
TUNE_PATH.parent.mkdir(parents=True, exist_ok=True)


def main():
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

    data_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
        ]
    )
    pickle.dump(data_pipeline, (DATA_PIPELINE_PATH).open("wb"))

    def objective(trial):
        X_train, X_test, y_train, y_test = train_test_split(
            features[feat_cols], labels, test_size=0.25
        )
        X_train = data_pipeline.fit_transform(X_train)

        param = {
            "device": "cuda",
            "verbosity": 0,
            "objective": "reg:squarederror",
            # use exact for small dataset.
            "tree_method": "auto",
            # defines booster, gblinear for linear functions.
            "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            # L2 regularization weight.
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            # L1 regularization weight.
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            # sampling ratio for training data.
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            # sampling according to each tree.
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        }

        if param["booster"] in ["gbtree", "dart"]:
            # maximum depth of the tree, signifies complexity of the tree.
            param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
            # minimum child weight, larger the term more conservative the tree.
            param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
            param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            # defines how selective algorithm is.
            param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["grow_policy"] = trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"]
            )

        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical(
                "normalize_type", ["tree", "forest"]
            )
            param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

        model = XGBRegressor(**param)
        model.fit(X_train, np.log1p(y_train))

        y_pred = np.expm1(model.predict(data_pipeline.transform(X_test)))
        rmsle = root_mean_squared_log_error(y_test, y_pred)
        print(f"Root Mean Squared Logarithmic Error: {rmsle:.4f}")
        return rmsle

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, callbacks=[DVCLiveCallback(str(TUNE_PATH))])

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    main()
