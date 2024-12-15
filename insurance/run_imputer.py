import pickle
from pathlib import Path

import pandas as pd
import torch
import typer
from tqdm import tqdm

from insurance.common import OUT_PATH, PREP_DATA_PATH
from insurance.data_pipeline import Features
from insurance.train_imputer import FeedforwardNN, TabularDataset, DataLoader, features_columns
from insurance.logger import setup_logger

MODEL_PATH = OUT_PATH / "models/torch_imputer.pt"
DATA_PIPELINE_PATH = OUT_PATH / "models/torch_imputer_data_pipeline.pkl"
INPUT_PATH = PREP_DATA_PATH / "prepared_data.feather"
OUTPUT_PATH = PREP_DATA_PATH / "prepared_imputed_data.feather"

app = typer.Typer()

logger = setup_logger()


def run_inference():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Running torch on {device}")

    target_column = "Previous Claims"
    df_init = pd.read_feather(INPUT_PATH)

    X = df_init[df_init[target_column].isna()]
    X = X.drop(columns=[target_column])
    X = X[features_columns.names]

    logger.info("Initialized data")

    data_pipeline = pickle.load(DATA_PIPELINE_PATH.open("rb"))
    X_transformed = data_pipeline.transform(X).to_numpy()

    logger.info("Transformed trained data")

    # Create PyTorch Datasets and DataLoaders
    inference_dataset = TabularDataset(X_transformed)

    inference_loader = DataLoader(inference_dataset, batch_size=1024)

    logger.info("Pytorch data structures")

    # Initialize the model, loss function, and optimizer
    logger.info(f"{X_transformed.shape=}")
    input_dim = X_transformed.shape[1]
    output_dim = 6
    logger.info(f"Number of output classes: {output_dim}")
    model = FeedforwardNN(input_dim=input_dim, output_dim=output_dim).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    with torch.no_grad():
        preds = []
        for features, _ in tqdm(inference_loader, desc="Inference"):
            features = features.to(device)
            outputs = model(features).squeeze().to("cpu")
            _, predicted = torch.max(outputs, 1)  # Get the class with highest score
            preds.append(predicted)
        preds = torch.cat(preds)
        X[target_column] = preds.numpy()

    df_init.loc[df_init[target_column].isna(), target_column] = X[target_column].values
    df_init.to_feather(OUTPUT_PATH)
    logger.info(f"Inference saved to {OUTPUT_PATH}")
    return df_init


if __name__ == "__main__":
    run_inference()
