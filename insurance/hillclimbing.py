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
def hill_climbing(models, true_targets, max_iters=2000, step_size=0.01):
    n_models = len(models)
    # Initialize weights randomly (normalized to sum to 1)
    weights = np.random.dirichlet(np.ones(n_models), size=1)[0]
    best_loss = float("inf")
    best_weights = weights.copy()

    for iteration in range(max_iters):
        # Combine predictions using current weights
        combined_preds = sum(w * model for w, model in zip(weights, models))
        loss = root_mean_squared_error(true_targets, combined_preds)

        # Check if the current solution is better
        if loss < best_loss:
            best_loss = loss
            best_weights = weights.copy()

        # Generate new weights (small random perturbations)
        new_weights = weights + np.random.uniform(-step_size, step_size, n_models)
        new_weights = np.clip(new_weights, 0, 1)  # Ensure weights are between 0 and 1
        new_weights /= new_weights.sum()  # Normalize weights to sum to 1

        # Combine predictions using new weights
        new_combined_preds = sum(w * model for w, model in zip(new_weights, models))
        new_loss = root_mean_squared_error(true_targets, new_combined_preds)

        # Move to the new weights if they improve the loss
        if new_loss < best_loss:
            weights = new_weights.copy()

        if iteration % 100 == 0:
            print(f"{best_loss=}")

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
    weights, loss = hill_climbing(models=models, true_targets=y_train)
    print(f"Optimal Weights: {weights}, Loss: {loss}")


if __name__ == "__main__":
    main()
