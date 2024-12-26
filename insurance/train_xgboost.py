import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
import xgboost as xgb

from dvclive import Live
from insurance.common import OUT_PATH
from insurance.data_pipeline import get_feat_columns, make_boosters_pipeline, get_folds
from insurance.logger import setup_logger
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error, root_mean_squared_error

MODEL_PATH = OUT_PATH / "models/xgboost_model.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

DATA_PIPELINE_PATH = OUT_PATH / "data_pipeline_train_xgboost.pkl"

log_file = datetime.now().strftime("xgboost_train_log_%Y-%m-%d_%H-%M-%S.log")
logger = setup_logger(log_file=log_file, name="xgboost trainer")


xgb_params = {
    "n_estimators": 1000,
    "learning_rate": 0.01,
    "max_depth": 10,
    "reg_lambda": 1.17,
    "reg_alpha": 0.1,
    "random_state": 42,
    "num_leaves": None,
    "min_child_weight": 1,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "device": "cuda",
    "tree_method": "hist",
    "verbosity": 0,
    "enable_categorical": True,
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

    fig_path = OUT_PATH / "xgboost_training.png"
    fig.savefig(fig_path, dpi=300)
    logger.info(f"Training loss saved at {fig_path}")


class SaveBestModel(xgb.callback.TrainingCallback):
    def __init__(self, cvboosters):
        self._cvboosters = cvboosters

    def after_training(self, model):
        self._cvboosters[:] = [cvpack.bst for cvpack in model.cvfolds]
        return model


def get_oof_preds(X_train: pd.DataFrame) -> np.ndarray[np.float64]:
    X_train = X_train.copy()
    models = pickle.load(MODEL_PATH.open("rb"))

    data_pipeline = pickle.load(DATA_PIPELINE_PATH.open("rb"))
    X_train = data_pipeline.transform(X_train)
    feat_cols = get_feat_columns()
    for col in feat_cols.categorical:
        X_train[col] = X_train[col].astype("category")

    oof_preds = np.zeros(len(X_train))
    folds = get_folds(df_train=X_train)
    splits = folds.split(X_train)
    for i, ((_, test_index), model) in enumerate(zip(splits, models)):
        logger.info(f"Predicting OOF -- {i+1}/{len(models)}")
        data = xgb.DMatrix(
            data=X_train.loc[test_index, :],
            enable_categorical=True,
            feature_names=X_train.columns.to_list(),
        )
        oof_preds[test_index] = model.predict(data=data)
    return oof_preds


def get_avg_preds(X: pd.DataFrame) -> np.ndarray[np.float64]:
    X = X.copy()
    models = pickle.load(MODEL_PATH.open("rb"))

    data_pipeline = pickle.load(DATA_PIPELINE_PATH.open("rb"))
    X = data_pipeline.transform(X)
    feat_cols = get_feat_columns()
    for col in feat_cols.categorical:
        X[col] = X[col].astype("category")

    preds = np.zeros(len(X))
    for i, model in enumerate(models):
        logger.info(f"Predicting on Test Data -- {i+1}/{len(models)}")
        data = xgb.DMatrix(
            data=X,
            enable_categorical=True,
            feature_names=X.columns.to_list(),
        )
        preds += model.predict(data=data)
    preds = preds / len(models)
    return preds


def main(prep_data_path: Path):
    target_column = "Premium Amount"

    df = pd.read_feather(prep_data_path)

    X_train = df.drop(columns=[target_column])
    logger.info(f"Train shape: {X_train.shape=}")
    X_train = X_train.loc[
        pd.to_datetime(X_train["Policy Start Date"], format="%Y%m%d").dt.year >= 2020
    ]
    logger.info(f"Train shape: {X_train.shape=}")
    y_train = df.loc[X_train.index, target_column]
    y_train = np.log1p(y_train)

    feat_cols = get_feat_columns()

    data_pipeline = make_boosters_pipeline()
    X_train = data_pipeline.fit_transform(X_train)
    for col in feat_cols.categorical:
        X_train[col] = X_train[col].astype("category")

    logger.info(f"Train shape: {X_train.shape=}")

    dtrain = xgb.DMatrix(
        X_train,
        label=y_train,
        enable_categorical=True,
        feature_names=X_train.columns.to_list(),
    )

    folds = get_folds(df_train=X_train, n_splits=5)

    lr_scheduler = xgb.callback.LearningRateScheduler(custom_learning_rate)
    cv_boosters = []
    history = xgb.cv(
        xgb_params,
        dtrain,
        num_boost_round=1000,
        early_stopping_rounds=10,
        # callbacks=[lr_scheduler, SaveBestModel(cv_boosters)],
        callbacks=[SaveBestModel(cv_boosters)],
        folds=folds,
        verbose_eval=True,
    )

    history = history.reset_index()
    history["index"] = history["index"] + 1
    history = history.rename(columns={"index": "booster"})

    plot_train_test(history=history)

    live_dir = Path("dvclive/xgboost/")
    live_dir.mkdir(parents=True, exist_ok=True)
    with Live(dir=str(live_dir)) as live:
        live.log_plot(
            "XGBoost CV Loss",
            history,
            x="booster",
            y=["train-rmse-mean", "test-rmse-mean"],
            template="linear",
            y_label="Booster",
            x_label="RMSLE",
        )

        live.log_metric("xgboost/train-cv-loss", history["train-rmse-mean"].iloc[-1])
        live.log_metric("xgboost/test-cv-loss", history["test-rmse-mean"].iloc[-1])
    pickle.dump(data_pipeline, open(DATA_PIPELINE_PATH, "wb"))
    logger.info(f"Data pipeline saved at {DATA_PIPELINE_PATH}")
    pickle.dump(cv_boosters, open(MODEL_PATH, "wb"))
    logger.info(f"Model saved at {MODEL_PATH}")


if __name__ == "__main__":
    typer.run(main)
