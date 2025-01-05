import pickle
from pathlib import Path

import dvc.api
import numpy as np
import optuna
import pandas as pd
import typer
from typing import Annotated, Optional
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import (
    StandardScaler,
)

from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error

from dvclive import Live
from insurance.common import (
    OUT_PATH,
    TARGET_COLUMN,
    OOF_PREDS_PATH,
    PREDS_PATH,
    RAW_DATA_PATH,
    ModelType,
)
from insurance.data_pipeline import get_folds
from insurance.logger import setup_logger

logger = setup_logger(name="ensemble")


def tune_ensemble(X_train: pd.DataFrame, y_train: pd.Series):
    def objective(trial):
        param = {
            "alpha": trial.suggest_float("alpha", 1e-1, 100, log=True),
            "solver": trial.suggest_categorical(
                "solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"]
            ),
        }

        if param["solver"] in ["sag", "saga"]:
            param["random_state"] = 42
        if param["solver"] == "lbfgs":
            param["positive"] = True

        n_splits = 5
        folds = get_folds(n_splits=5)
        mean_rmse = 0
        for train_idx, val_idx in folds.split(X_train):
            model = Ridge(**param)
            X, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X=X, y=y)
            test_preds = model.predict(X=X_val)
            test_rmse = root_mean_squared_error(y_true=y_val, y_pred=test_preds)
            mean_rmse += test_rmse / n_splits

        print(f"Out-of-fold RMSLE: {mean_rmse:.4f}")
        return mean_rmse

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=100)

    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("PARAMS = {")
    for key, value in trial.params.items():
        if isinstance(value, str):
            print('    "{}": "{}",'.format(key, value))
        else:
            print('    "{}": {},'.format(key, value))
    print("}")


def make_ensemble_pipeline(pred_columns: list[str]) -> Pipeline:
    pipeline = make_pipeline(
        make_column_transformer(
            (StandardScaler(), pred_columns),
            remainder="drop",
            verbose_feature_names_out=False,
        )
    )
    pipeline.set_output(transform="pandas")
    return pipeline


def main(
    layer: Annotated[int, typer.Option(help="Layer number")],
    ensemble_name: Annotated[
        str, typer.Option(help="Name of ensemble model. Must be an entry in params.yaml:ensemble")
    ],
    additional_data: Annotated[
        Optional[list[Path]],
        typer.Option(help="Path to additional predictions or features."),
    ] = None,
):
    params = dvc.api.params_show()
    try:
        params = params["ensemble"][ensemble_name]
    except KeyError as err:
        msg = f"{ensemble_name} must be defined in params.yaml:ensemble"
        raise KeyError(msg) from err

    raw_train_data = pd.read_csv(RAW_DATA_PATH / "train.csv")
    y_train = raw_train_data[TARGET_COLUMN]
    y_train = np.log1p(y_train)

    train_data_path = OOF_PREDS_PATH / f"layer_{layer}_concatenated.feather"
    train_data = pd.read_feather(train_data_path)
    logger.info(f"Read train data {train_data_path}")

    test_data_path = PREDS_PATH / f"layer_{layer}_concatenated.feather"
    test_data = pd.read_feather(test_data_path)
    logger.info(f"Read train data {test_data_path}")

    X_train = train_data.drop(columns=[TARGET_COLUMN], errors="ignore")

    # Select only predictions of models from last layer
    pred_columns = [col for col in X_train if "_preds" in col]

    data_pipeline = make_ensemble_pipeline(pred_columns=pred_columns)
    X_train = np.log1p(X_train)
    X_train = data_pipeline.fit_transform(X_train)
    test_data = test_data[pred_columns]
    test_data = np.log1p(test_data)
    X_test = data_pipeline.transform(test_data)

    logger.info(f"Train shape: {X_train.shape=}")
    logger.info(f"Columns: {X_train.columns}")

    n_splits = 5
    folds = get_folds(n_splits=n_splits)

    tune = False
    if tune:
        tune_ensemble(X_train=X_train, y_train=y_train)
        return
    ensemble_regressors = []
    metrics = {"train-rmse-mean": 0.0, "test-rmse-mean": 0.0}

    if params["type"] == ModelType.SKLEARN.value:
        cap_class = params["class"].capitalize()
        mod = __import__("sklearn.linear_model", fromlist=[cap_class])
        sklearn_model = getattr(mod, cap_class)
        logger.info(f"Loaded {cap_class} from sklearn")

        for train_idx, val_idx in folds.split(X_train):
            model = sklearn_model(**params["params"])
            X, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X=X, y=y)
            train_preds = model.predict(X=X)
            train_rmse = root_mean_squared_error(y_true=y, y_pred=train_preds)
            val_preds = model.predict(X=X_val)
            val_rmse = root_mean_squared_error(y_true=y_val, y_pred=val_preds)
            metrics["train-rmse-mean"] += train_rmse / n_splits
            metrics["test-rmse-mean"] += val_rmse / n_splits
            ensemble_regressors.append(model)
        live_dir = Path(f"dvclive/ensemble_{ensemble_name}_layer_{layer}/")
        live_dir.mkdir(parents=True, exist_ok=True)
        with Live(dir=str(live_dir)) as live:
            live.log_metric(
                f"ensemble_{ensemble_name}_layer_{layer}/train-cv-loss",
                metrics["train-rmse-mean"],
            )
            live.log_metric(
                f"ensemble_{ensemble_name}_layer_{layer}/test-cv-loss",
                metrics["test-rmse-mean"],
            )

        model_path = OUT_PATH / f"models/ensemble_model_{ensemble_name}_layer_{layer}.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(ensemble_regressors, open(model_path, "wb"))
        logger.info(f"Model saved at {model_path}")

        preds = np.expm1(model.predict(X=X_test))

    elif params["class"] == "hill-climbing":
        logger.info("Calling Hill Climbing")
        raise NotImplementedError

    output = pd.read_csv(RAW_DATA_PATH / "sample_submission.csv")
    output["Premium Amount"] = preds
    predictions_path = OUT_PATH / f"layer_{layer}_preds_{ensemble_name}.csv"
    output.to_csv(predictions_path, index=False)
    logger.info(f"Final prediction on test set saved at {predictions_path}")


if __name__ == "__main__":
    typer.run(main)
