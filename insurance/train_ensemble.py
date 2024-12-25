import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
import xgboost as xgb

from dvclive import Live
from insurance.common import OUT_PATH
from insurance.data_pipeline import get_feat_columns, get_folds, make_boosters_pipeline
from insurance.logger import setup_logger
from insurance.train_catboost import get_oof_preds as catboost_oof_preds
from insurance.train_xgboost import get_oof_preds as xgboost_oof_preds

MODEL_PATH = OUT_PATH / "models/ensemble_model.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

DATA_PIPELINE_PATH = OUT_PATH / "data_pipeline_train_ensemble.pkl"

log_file = datetime.now().strftime("xgboost_train_log_%Y-%m-%d_%H-%M-%S.log")
logger = setup_logger(log_file=log_file, name="ensemble trainer")

PREV_LAYER_OOF = {"xgboost": xgboost_oof_preds, "catboost": catboost_oof_preds}

xgb_params = {
    "lambda": 151.89367354249936,
    "max_depth": 8,
    "min_child_weight": 2,
    "device": "cuda",
    "verbosity": 0,
    "objective": "reg:squarederror",
    "random_state": 42,
    "eval_metric": "rmse",
    "tree_method": "auto",
    "alpha": 0.1,
    "booster": "gbtree",
    "gamma": 3e-06,
    "grow_policy": "depthwise",
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


def main(prep_data_path: Path):
    target_column = "Premium Amount"

    df = pd.read_feather(prep_data_path)

    X_train = df.drop(columns=[target_column])
    y_train = df[target_column]
    y_train = np.log1p(y_train)

    for model, oof_func in PREV_LAYER_OOF.items():
        logger.info(f"{model} OOF predictions...")
        X_train[f"{model}_oof_preds"] = oof_func(X_train=X_train)

    feat_cols = get_feat_columns()

    data_pipeline = make_boosters_pipeline()
    X_train = data_pipeline.fit_transform(X_train)
    for col in feat_cols.categorical:
        X_train[col] = X_train[col].astype("category")

    logger.info(f"Train shape: {X_train.shape=}")
    logger.info(f"Columns: {X_train.columns}")
    dtrain = xgb.DMatrix(
        X_train,
        label=y_train,
        enable_categorical=True,
        feature_names=X_train.columns.to_list(),
    )

    folds = get_folds(df_train=X_train, n_splits=5)
    folds = [(train, val) for (val, train) in folds]

    lr_scheduler = xgb.callback.LearningRateScheduler(custom_learning_rate)
    cv_ensemble_boosters = []
    history = xgb.fit(
        xgb_params,
        dtrain,
        num_boost_round=100,
        callbacks=[lr_scheduler, SaveBestModel(cv_ensemble_boosters)],
        folds=folds,
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
