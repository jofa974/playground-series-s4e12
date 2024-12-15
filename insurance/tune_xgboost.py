import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import KFold

from insurance.common import OUT_PATH, PREP_DATA_PATH
from insurance.data_pipeline import get_feat_columns, make_pipeline
from insurance.log import setup_logger
import typer

model_label = Path(__file__).stem.split("_")[-1]

DATA_PIPELINE_PATH = OUT_PATH / f"data_pipeline_tune_{model_label}"
BEST_PARAMS_PATH = OUT_PATH / "xgboost_model_params.yaml"


def main(prep_data_path: Path):
    log_file = datetime.now().strftime("xgboost_tune_log_%Y-%m-%d_%H-%M-%S.log")
    logger = setup_logger(log_file=log_file)

    target_column = "Premium Amount"

    df = pd.read_feather(prep_data_path)
    features = df.drop(columns=[target_column])
    labels = df[target_column]

    feat_cols = get_feat_columns()
    feat_names = feat_cols.names

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    train_folds = []
    test_folds = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(features[feat_names])):
        logger.info(f"Fold {fold + 1}")
        X_train, X_test = (
            features[feat_names].iloc[train_idx],
            features[feat_names].iloc[test_idx],
        )
        y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]

        # Fit the pipeline
        data_pipeline = make_pipeline()
        X_train = data_pipeline.fit_transform(X_train)
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

        train_folds.append(dtrain)
        test_folds.append(dtest)

        pickle.dump(data_pipeline, open(str(DATA_PIPELINE_PATH) + f"_fold_{fold}.pkl", "wb"))

    def objective(trial):
        param = {
            "device": "cuda",
            "verbosity": 0,
            "objective": "reg:squarederror",
            "random_state": 42,
            "eval_metric": "rmse",
            # use exact for small dataset.
            "tree_method": "auto",
            # defines booster, gblinear for linear functions.
            "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
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
            param["eta"] = trial.suggest_float("eta", 1e-5, 1.0, log=True)
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

        score = 0
        for fold, (dtrain, dtest) in enumerate(zip(train_folds, test_folds)):
            bst = xgb.train(param, dtrain)

            # Predict and evaluate
            y_pred = np.expm1(bst.predict(dtest))
            rmsle = root_mean_squared_log_error(y_test, y_pred)
            logger.info(f" !!! Fold {fold+1} !!! Root Mean Squared Logarithmic Error: {rmsle:.4f}")
            score += rmsle

        avg = score / n_splits
        logger.info(f"Average RMSLE across folds: {avg}")
        return avg

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    logger.info(f"Number of finished trials: {len(study.trials)}")
    logger.info("Best trial:")
    trial = study.best_trial

    logger.info("  Value: {}".format(trial.value))
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info("    {}: {}".format(key, value))
    with open(BEST_PARAMS_PATH, "w") as f:
        yaml.dump(trial.params, f)


if __name__ == "__main__":
    typer.run(main)
