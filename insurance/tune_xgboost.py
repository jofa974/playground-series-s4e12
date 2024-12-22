import copy
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
import random
from insurance.common import OUT_PATH
from insurance.data_pipeline import get_feat_columns, make_xgboost_pipeline
from insurance.logger import setup_logger
import typer

model_label = Path(__file__).stem.split("_")[-1]

DATA_PIPELINE_PATH = OUT_PATH / f"data_pipeline_tune_{model_label}"
BEST_PARAMS_PATH = OUT_PATH / "xgboost_model_params.yaml"


def main(prep_data_path: Path):
    log_file = datetime.now().strftime("xgboost_tune_log_%Y-%m-%d_%H-%M-%S.log")
    logger = setup_logger(log_file=log_file)

    target_column = "Premium Amount"

    df = pd.read_feather(prep_data_path)

    feat_cols = get_feat_columns()
    feat_names = feat_cols.names

    features = df.drop(columns=[target_column])
    features = features[feat_names]
    logger.info(f"features shape: {features.shape}")

    labels = df[target_column]

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=21)

    train_folds = []
    valid_folds = []
    idx = []
    for fold, (train_idx, valid_idx) in enumerate(
        random.sample(list(kf.split(features[feat_names])), n_splits)
    ):
        logger.info(f"Fold {fold + 1}")
        X_train, X_valid = (
            features[feat_names].iloc[train_idx],
            features[feat_names].iloc[valid_idx],
        )
        y_train, y_valid = labels.iloc[train_idx], labels.iloc[valid_idx]

        # Fit the pipeline
        data_pipeline = make_xgboost_pipeline()
        X_train = data_pipeline.fit_transform(X_train)
        for col in feat_cols.categorical:
            X_train[col] = X_train[col].astype("category")
        dtrain = xgb.DMatrix(
            X_train,
            label=np.log1p(y_train),
            enable_categorical=True,
            feature_names=X_train.columns.to_list(),
        )

        X_valid = data_pipeline.transform(X_valid)
        for col in feat_cols.categorical:
            X_valid[col] = X_valid[col].astype("category")
        dvalid = xgb.DMatrix(
            X_valid,
            label=np.log1p(y_valid),
            enable_categorical=True,
            feature_names=X_valid.columns.to_list(),
        )

        train_folds.append(dtrain)
        valid_folds.append(dvalid)
        idx.append(valid_idx)

        pickle.dump(data_pipeline, open(str(DATA_PIPELINE_PATH) + f"_fold_{fold}.pkl", "wb"))

    base_param = {
        "device": "cuda",
        "verbosity": 0,
        "objective": "reg:squarederror",
        "random_state": 42,
        "eval_metric": "rmse",
        # use exact for small dataset.
        "tree_method": "auto",
    }

    def objective(trial):
        param = copy.deepcopy(base_param)
        param.update(
            {
                # # defines booster, gblinear for linear functions.
                # "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
                "booster": "gbtree",
                # L2 regularization weight.
                "lambda": trial.suggest_float("lambda", 0.1, 100, log=True),
                # L1 regularization weight.
                "alpha": trial.suggest_float("alpha", 1e-3, 0.2, log=True),
                # sampling ratio for training data.
                # "subsample": trial.suggest_float("subsample", 0.2, 1.0),
                # # sampling according to each tree.
                # "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            }
        )

        if param["booster"] in ["gbtree", "dart"]:
            # maximum depth of the tree, signifies complexity of the tree.
            param["max_depth"] = trial.suggest_int("max_depth", 2, 12, step=1)
        #     # minimum child weight, larger the term more conservative the tree.
        #     param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        #     param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        #     # defines how selective algorithm is.
        #     param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        #     param["grow_policy"] = trial.suggest_categorical(
        #         "grow_policy", ["depthwise", "lossguide"]
        #     )

        # if param["booster"] == "dart":
        #     param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        #     param["normalize_type"] = trial.suggest_categorical(
        #         "normalize_type", ["tree", "forest"]
        #     )
        #     param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        #     param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

        num_boost_round = trial.suggest_int("num_boost_round", 10, 400)
        rmsle_scores = np.zeros(n_splits)
        oof_preds = np.zeros(labels.shape[0])
        for fold, (dtrain, dvalid, val_idx) in enumerate(zip(train_folds, valid_folds, idx)):
            bst = xgb.train(param, dtrain, num_boost_round=num_boost_round)

            # Predict and evaluate
            y_pred = np.expm1(bst.predict(dvalid))
            oof_preds[val_idx] = y_pred
            rmsle = root_mean_squared_log_error(labels.iloc[val_idx], y_pred)
            logger.info(f" !!! Fold {fold+1} !!! Root Mean Squared Logarithmic Error: {rmsle:.4f}")
            rmsle_scores[fold] = rmsle

        avg = np.average(rmsle_scores)
        std = np.std(rmsle_scores)
        logger.info(f"Average +-std RMSLE across folds: {avg:.4f} +-{std:.4f}")

        rmsle_oof = root_mean_squared_log_error(labels, oof_preds)
        logger.info(f"Out-of-fold RMSLE: {rmsle_oof:.4f}")

        return rmsle_oof

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    logger.info(f"Number of finished trials: {len(study.trials)}")
    logger.info("Best trial:")
    trial = study.best_trial

    logger.info("  Value: {}".format(trial.value))
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info("    {}: {}".format(key, value))
        base_param[key] = value
    with open(BEST_PARAMS_PATH, "w") as f:
        yaml.dump(base_param, f)


if __name__ == "__main__":
    typer.run(main)
