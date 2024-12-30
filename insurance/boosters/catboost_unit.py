import pickle
from pathlib import Path

import catboost as cb
import numpy as np
import optuna
import pandas as pd
import typer

from dvclive import Live
from insurance.common import OUT_PATH, TARGET_COLUMN, RAW_DATA_PATH
from insurance.data_pipeline import get_feat_columns, get_folds, make_pipeline
from insurance.logger import setup_logger

logger = setup_logger(name="catboost")


def get_oof_preds(X_train: pd.DataFrame, model_path: Path) -> np.ndarray[np.float64]:
    """Assumes data is transformed."""
    models = pickle.load(model_path.open("rb"))

    oof_preds = np.zeros(len(X_train))
    folds = get_folds()
    splits = folds.split(X_train)
    for i, ((_, test_index), model) in enumerate(zip(splits, models)):
        logger.info(f"Predicting OOF -- {i+1}/{len(models)}")
        oof_preds[test_index] = model.predict(data=X_train.loc[test_index, :])
    return oof_preds


def get_avg_preds(X: pd.DataFrame, model_path: Path) -> np.ndarray[np.float64]:
    """Assumes data is transformed."""
    models = pickle.load(model_path.open("rb"))

    preds = np.zeros(len(X))
    for i, model in enumerate(models):
        logger.info(f"Predicting on Test Data -- {i+1}/{len(models)}")
        preds += model.predict(
            data=X,
        )
    preds = preds / len(models)
    return preds


def tune_catboost():
    feat_cols = get_feat_columns()

    df_train = pd.read_csv(RAW_DATA_PATH / "train.csv")
    target = df_train[TARGET_COLUMN]
    df_train = df_train.drop(columns=["id", TARGET_COLUMN])
    df_train["Policy Start Date"] = (
        pd.to_datetime(df_train["Policy Start Date"]).dt.strftime("%Y%m%d").astype(np.int64)
    )
    logger.info("Transforming training data...")
    feat_cols = get_feat_columns()
    data_pipeline = make_pipeline()
    df_train = data_pipeline.fit_transform(df_train)
    for col in feat_cols.categorical:
        df_train[col] = df_train[col].astype("category")

    df_train[TARGET_COLUMN] = np.log1p(target)
    logger.info(f"{TARGET_COLUMN} transformed with log1p")

    X_train = df_train.drop(columns=[TARGET_COLUMN])
    y_train = df_train[TARGET_COLUMN]

    logger.info(f"Train shape: {X_train.shape=}")
    logger.info(f"Columns: {X_train.columns=}")

    cat_features = X_train.select_dtypes(exclude="number").columns.tolist()
    print(cat_features)
    train_pool = cb.Pool(data=X_train, label=y_train, cat_features=cat_features)

    folds = get_folds(n_splits=5)

    def objective(trial):
        param = {
            "iterations": trial.suggest_int("iterations", 400, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.4, log=True),
            "depth": trial.suggest_int("depth", 4, 9),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 1, log=True),
            "loss_function": trial.suggest_categorical("loss_function", ["RMSE"]),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 1, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 1e-2, 1, log=True),
            "task_type": "GPU",
        }
        history = cb.cv(
            pool=train_pool,
            params=param,
            folds=folds,
            verbose=0,
            return_models=False,
        )
        mean_rmse = history["test-RMSE-mean"].values[-1]

        print(f"Out-of-fold RMSLE: {mean_rmse:.4f}")
        return mean_rmse

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=50)

    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("catboost_params = {")
    for key, value in trial.params.items():
        if isinstance(value, str):
            print('    "{}": "{}",'.format(key, value))
        else:
            print('    "{}": {},'.format(key, value))
    print("}")


def train(
    params: dict, model_name: str, layer: int, train_data: pd.DataFrame, test_data: pd.DataFrame
) -> tuple[np.ndarray]:
    X_train = train_data.drop(columns=[TARGET_COLUMN])
    y_train = train_data[TARGET_COLUMN]

    logger.info(f"Train shape: {X_train.shape=}")
    logger.info(f"Columns: {X_train.columns=}")

    cat_features = X_train.select_dtypes(exclude="number").columns.tolist()
    train_pool = cb.Pool(data=X_train, label=y_train, cat_features=cat_features, has_header=True)

    folds = get_folds(n_splits=3)

    logger.info(f"{params=}")
    history, cv_boosters = cb.cv(
        pool=train_pool,
        params=params,
        folds=folds,
        as_pandas=True,
        verbose=True,
        return_models=True,
    )

    live_dir = Path(f"dvclive/{model_name}_layer_{layer}/")
    live_dir.mkdir(parents=True, exist_ok=True)
    with Live(dir=str(live_dir)) as live:
        live.log_plot(
            f"Catboost CV Loss {layer}",
            history,
            x="booster",
            y=["train-RMSE-mean", "test-RMSE-mean"],
            template="linear",
            y_label="RMSLE",
            x_label="Booster",
        )

        live.log_metric(f"{model_name}/train-cv-loss", history["train-RMSE-mean"].iloc[-1])
        live.log_metric(f"{model_name}/test-cv-loss", history["test-RMSE-mean"].iloc[-1])

    model_path = OUT_PATH / f"models/{model_name}_model_layer_{layer}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(cv_boosters, open(model_path, "wb"))
    logger.info(f"Model saved at {model_path}")

    oof_preds = get_oof_preds(X_train=X_train, model_path=model_path)
    avg_preds = get_avg_preds(X=test_data, model_path=model_path)

    return oof_preds, avg_preds


if __name__ == "__main__":
    typer.run(tune_catboost)
