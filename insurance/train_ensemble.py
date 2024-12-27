import copy
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import typer
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
)

from dvclive import Live
from insurance.common import OUT_PATH
from insurance.data_pipeline import get_feat_columns, get_folds
from insurance.logger import setup_logger
from insurance.train_catboost import get_oof_preds as catboost_oof_preds
from insurance.train_xgboost import get_oof_preds as xgboost_oof_preds

MODEL_PATH = OUT_PATH / "models/ensemble_model.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

DATA_PIPELINE_PATH = OUT_PATH / "data_pipeline_train_ensemble.pkl"

log_file = datetime.now().strftime("ensemble_train_log_%Y-%m-%d_%H-%M-%S.log")
logger = setup_logger(log_file=log_file, name="ensemble trainer")

PREV_LAYER_OOF = {"xgboost": xgboost_oof_preds, "catboost": catboost_oof_preds}

xgb_params = {
    "device": "cuda",
    "verbosity": 0,
    "objective": "reg:squarederror",
    "random_state": 42,
    "eval_metric": "rmse",
    "tree_method": "auto",
    "alpha": 0.012278360049275504,
    "booster": "gbtree",
    "gamma": 3e-06,
    "grow_policy": "depthwise",
    "eta": 0.2,
    "lambda": 189.5759434037735,
    "subsample": 0.9756296886302929,
    "colsample_bytree": 0.9555993163831182,
    "max_depth": 10,
    "min_child_weight": 8,
}


def custom_learning_rate(current_iter):
    base_learning_rate = 0.3
    lr = base_learning_rate * np.power(0.95, current_iter)
    return lr if lr > 1e-3 else 1e-3


def plot_train_test(history: pd.DataFrame):
    # Plotting
    fig, ax = plt.subplots()

    ax.plot(history["booster"], history["train-rmse-mean"], label="Train RMSE Mean")
    ax.plot(history["booster"], history["test-rmse-mean"], label="Test RMSE Mean")

    # Adding labels, title, and legend
    ax.set_xlabel("Booster")
    ax.set_ylabel("RMSE Mean")
    ax.set_title("Train and Test RMSE Mean vs Booster")
    plt.legend()
    plt.grid(True)

    last_booster = history["booster"].iloc[-1]
    train_rmse_last = history["train-rmse-mean"].iloc[-1]
    test_rmse_last = history["test-rmse-mean"].iloc[-1]

    ax.text(
        last_booster,
        train_rmse_last,
        f"{train_rmse_last:.6f}",
        fontsize=10,
        ha="left",
        va="bottom",
        color="blue",
        bbox=dict(facecolor="white", alpha=0.8),
    )
    ax.text(
        last_booster,
        test_rmse_last,
        f"{test_rmse_last:.6f}",
        fontsize=10,
        ha="left",
        va="bottom",
        color="orange",
        bbox=dict(facecolor="white", alpha=0.8),
    )
    plt.draw()

    fig_path = OUT_PATH / "ensemble_training.png"
    fig.savefig(fig_path, dpi=300)
    logger.info(f"Training loss saved at {fig_path}")


class SaveBestModel(xgb.callback.TrainingCallback):
    def __init__(self, cvboosters):
        self._cvboosters = cvboosters

    def after_training(self, model):
        self._cvboosters[:] = [cvpack.bst for cvpack in model.cvfolds]
        return model


def tune_ensemble(dtrain: xgb.DMatrix):
    base_param = {
        "device": "cuda",
        "verbosity": 0,
        "objective": "reg:squarederror",
        "random_state": 42,
        "eval_metric": "rmse",
        # use exact for small dataset.
        "tree_method": "auto",
        "alpha": 0.1,
        "booster": "gbtree",
        "gamma": 3e-6,
        "grow_policy": "depthwise",
        "eta": 0.2,
    }

    def objective(trial):
        param = copy.deepcopy(base_param)
        param.update(
            {
                # # defines booster, gblinear for linear functions.
                # "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
                # L2 regularization weight.
                "lambda": trial.suggest_float("lambda", 10, 200),
                # L1 regularization weight.
                "alpha": trial.suggest_float("alpha", 1e-3, 0.2, log=True),
                # sampling ratio for training data.
                "subsample": trial.suggest_float("subsample", 0.2, 1.0),
                # # sampling according to each tree.
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            }
        )

        if param["booster"] in ["gbtree", "dart"]:
            # maximum depth of the tree, signifies complexity of the tree.
            param["max_depth"] = trial.suggest_int("max_depth", 2, 10, step=1)
            # minimum child weight, larger the term more conservative the tree.
            param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
            # param["eta"] = trial.suggest_float("eta", 0.15, 0.25)
            # defines how selective algorithm is.
            # param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

        # if param["booster"] == "dart":
        #     param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        #     param["normalize_type"] = trial.suggest_categorical(
        #         "normalize_type", ["tree", "forest"]
        #     )
        #     param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        #     param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

        # num_boost_round = trial.suggest_int("num_boost_round", 10, 40)
        num_boost_round = 40

        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-rmse")
        history = xgb.cv(
            param, dtrain, num_boost_round=num_boost_round, callbacks=[pruning_callback]
        )
        mean_rmse = history["test-rmse-mean"].values[-1]

        print(f"Out-of-fold RMSLE: {mean_rmse:.4f}")
        return mean_rmse

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        # pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=50)

    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    base_param.update(trial.params)
    for key, value in base_param.items():
        print("    {}: {}".format(key, value))


def make_ensemble_pipeline() -> Pipeline:
    num_transformer = Pipeline([("scaler", StandardScaler())])
    pipeline = Pipeline(
        [
            (
                "num_scaler",
                ColumnTransformer(
                    transformers=[
                        ("num", num_transformer, ["xgboost_preds", "catboost_preds"]),
                    ],
                    remainder="drop",
                    verbose_feature_names_out=False,
                ),
            ),
        ]
    )
    pipeline.set_output(transform="pandas")
    return pipeline


def main(prep_data_path: Path):
    target_column = "Premium Amount"

    df = pd.read_feather(prep_data_path)

    tune = False

    X_train = df.drop(columns=[target_column])
    y_train = df[target_column]
    y_train = np.log1p(y_train)

    columns = X_train.columns
    for model, oof_func in PREV_LAYER_OOF.items():
        logger.info(f"{model} OOF predictions...")
        # Ensure that OOF predictions of a model do not use previous model OOF preds.
        X_train[f"{model}_preds"] = oof_func(X_train=X_train[columns])

    data_pipeline = make_ensemble_pipeline()
    X_train = data_pipeline.fit_transform(X_train)

    logger.info(f"Train shape: {X_train.shape=}")
    logger.info(f"Columns: {X_train.columns}")
    dtrain = xgb.DMatrix(
        X_train,
        label=y_train,
        enable_categorical=True,
        feature_names=X_train.columns.to_list(),
    )

    folds = get_folds(df_train=X_train, n_splits=5)

    if tune:
        tune_ensemble(dtrain=dtrain)
    else:
        cv_ensemble_boosters = []
        history = xgb.cv(
            xgb_params,
            dtrain,
            num_boost_round=40,
            callbacks=[SaveBestModel(cv_ensemble_boosters)],
            folds=folds,
            verbose_eval=2,
        )

        history = history.reset_index()
        history["index"] = history["index"] + 1
        history = history.rename(columns={"index": "booster"})

        plot_train_test(history=history)

        live_dir = Path("dvclive/ensemble/")
        live_dir.mkdir(parents=True, exist_ok=True)
        with Live(dir=str(live_dir)) as live:
            live.log_plot(
                "ensemble CV Loss",
                history,
                x="booster",
                y=["train-rmse-mean", "test-rmse-mean"],
                template="linear",
                y_label="Booster",
                x_label="RMSLE",
            )

            live.log_metric("ensemble/train-cv-loss", history["train-rmse-mean"].iloc[-1])
            live.log_metric("ensemble/test-cv-loss", history["test-rmse-mean"].iloc[-1])
        pickle.dump(data_pipeline, open(DATA_PIPELINE_PATH, "wb"))
        logger.info(f"Data pipeline saved at {DATA_PIPELINE_PATH}")
        pickle.dump(cv_ensemble_boosters, open(MODEL_PATH, "wb"))
        logger.info(f"Model saved at {MODEL_PATH}")


if __name__ == "__main__":
    typer.run(main)
