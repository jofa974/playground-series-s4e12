from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

from insurance.logger import setup_logger
from insurance.train_catboost import get_oof_preds as catboost_oof_preds
from insurance.train_lgbm import get_oof_preds as lgbm_oof_preds
from insurance.train_xgboost import get_oof_preds as xgboost_oof_preds
from insurance.train_ensemble import make_ensemble_pipeline
from tqdm import tqdm

PREV_LAYER_OOF = {
    "xgboost": xgboost_oof_preds,
    "catboost": catboost_oof_preds,
    "lgbm": lgbm_oof_preds,
}

log_file = datetime.now().strftime("hillclimbing_log_%Y-%m-%d_%H-%M-%S.log")
logger = setup_logger(log_file=log_file, name="hill climbing")


# Combine predictions using hill climbing
def hill_climbing(models, true_labels, steps=2000, learning_rate=10):
    n_models = len(models)
    weights = np.ones(n_models) / n_models  # Start with equal weights
    best_loss = float("inf")
    best_weights = weights.copy()

    for step in tqdm(range(steps)):
        # Generate neighbor weights
        new_weights = weights + np.random.uniform(-learning_rate, learning_rate, n_models)
        new_weights = np.clip(new_weights, -10, 10)  # Keep weights between 0 and 1
        new_weights /= new_weights.sum()  # Normalize to sum to 1

        # Combine predictions
        combined_preds = sum(w * p for w, p in zip(new_weights, models))
        loss = root_mean_squared_error(true_labels, combined_preds)

        # Update if better
        if loss < best_loss:
            best_step = step
            best_loss = loss
            best_weights = new_weights

        if step % 100 == 0:
            print(f" Best loss {best_loss:.8f}")
            print(f" Best step {best_step}")

    return best_weights, best_loss


def main(prep_data_path: Path = Path("data/prepared/prepared_data.feather")):
    target_column = "Premium Amount"

    df = pd.read_feather(prep_data_path)

    X_train = df.drop(columns=[target_column])
    y_train = df[target_column]
    y_train = np.log1p(y_train)

    columns = X_train.columns
    for model, oof_func in PREV_LAYER_OOF.items():
        logger.info(f"{model} OOF predictions...")
        # Ensure that OOF predictions of a model do not use previous model OOF preds.
        X_train[f"{model}_preds"] = oof_func(X_train=X_train[columns])

    X_train = X_train[[col for col in X_train.columns if "_preds" in col]]

    logger.info(f"Train shape: {X_train.shape=}")
    logger.info(f"Columns: {X_train.columns}")

    models = X_train.to_numpy().T
    weights, loss = hill_climbing(models=models, true_labels=y_train)
    print(f"Optimal Weights: {weights}, Loss: {loss}")


if __name__ == "__main__":
    main()
