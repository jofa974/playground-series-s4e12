import pickle
from pathlib import Path

import dvc.api
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import typer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dvclive import Live
from insurance.common import OUT_PATH, PREP_DATA_PATH
from insurance.data_pipeline import Features, make_pipeline
from insurance.logger import setup_logger

MODEL_PATH = OUT_PATH / "models/torch_imputer.pt"
DATA_PIPELINE_PATH = OUT_PATH / "models/torch_imputer_data_pipeline.pkl"
INPUT_PATH = PREP_DATA_PATH / "prepared_data.feather"
app = typer.Typer()


features_columns = Features(
    numeric=[
        "Age",
        "Health Score",
        "Credit Score",
        "Insurance Duration",
        "Number of Dependents",
        "Vehicle Age",
    ],
    numeric_log=["Annual Income"],
    categorical=[],
    ordinal=[],
)


# Custom Dataset class
class TabularDataset(Dataset):
    def __init__(self, features, targets=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.long)
        else:
            self.targets = torch.tensor([torch.nan] * len(features))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeedforwardNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
        )

    def forward(self, x):
        return self.model(x)


def prepare_data_pipeline(
    input_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, Pipeline]:
    target_column = "Previous Claims"

    input_df = input_df[~input_df[target_column].isna()]
    features = input_df.drop(columns=[target_column])
    target = input_df[target_column]

    data_pipeline = make_pipeline(feat_cols=features_columns)
    return features, target, data_pipeline


def train():
    logger = setup_logger()

    params = dvc.api.params_show()["torch_imputer"]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Running torch on {device}")
    df = pd.read_feather(INPUT_PATH)

    features, target, data_pipeline = prepare_data_pipeline(input_df=df)

    target = target.clip(0, 5)

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    X_train, X_test = (
        X_train[features_columns.names],
        X_test[features_columns.names],
    )
    X_train = data_pipeline.fit_transform(X_train).to_numpy()
    X_test = data_pipeline.transform(X_test).to_numpy()

    logger.info("Transformed trained data")
    pickle.dump(data_pipeline, DATA_PIPELINE_PATH.open("wb"))
    logger.info(f"Saved data pipeline at {DATA_PIPELINE_PATH}")

    # Create PyTorch Datasets and DataLoaders
    train_dataset = TabularDataset(X_train, y_train.values)
    test_dataset = TabularDataset(X_test, y_test.values)

    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False)

    logger.info("Pytorch data structures")

    # Initialize the model, loss function, and optimizer
    logger.info(f"{X_train.shape=}")
    input_dim = X_train.shape[1]
    output_dim = y_train.nunique()
    logger.info(f"Number of output classes: {output_dim}")
    model = FeedforwardNN(input_dim=input_dim, output_dim=output_dim).to(device)
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])

    logger.info("NN + optim")

    with Live() as live:
        live.log_params(params)

        model.train()
        plots = []
        for epoch in range(params["epochs"]):
            epoch_loss = 0
            total = 0
            correct = 0
            for features, targets in tqdm(
                train_loader, desc=f"Training Epoch {epoch + 1}/{params['epochs']}"
            ):
                features, targets = features.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(features).squeeze()
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                _, predicted = torch.max(outputs, 1)  # Get the class with highest score
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            train_loss = epoch_loss / len(train_loader)
            train_acc = correct / total

            logger.info(
                f"Epoch {epoch + 1}/{params['epochs']}, Loss: {epoch_loss / len(train_loader):.4f}, Accuracy: {correct / total:.4f} "
            )

            eval_loss, eval_accuracy = evaluate_model(
                model=model, test_loader=test_loader, device=device
            )
            logger.info(f"Eval Loss: {eval_loss:.4f}")
            logger.info(f"Eval Accuracy: {eval_accuracy:.4f}")

            plots.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                    "train_accuracy": train_acc,
                    "eval_accuracy": eval_accuracy,
                }
            )
            live.log_plot(name="loss", datapoints=plots, x="epoch", y=["train_loss", "eval_loss"])
            live.log_plot(
                name="accuracy", datapoints=plots, x="epoch", y=["train_accuracy", "eval_accuracy"]
            )
            live.next_step()

        torch.save(model.state_dict(), MODEL_PATH)
        live.log_artifact(MODEL_PATH, type="model", name="pytorch-model")


def evaluate_model(model, test_loader, device) -> tuple[np.float64, np.float64]:
    model.eval()
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        total_loss = 0
        correct = 0
        total = 0
        for features, targets in tqdm(test_loader, desc="Evaluating"):
            features, targets = features.to(device), targets.to(device)
            outputs = model(features).squeeze()
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)  # Get the class with highest score
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return total_loss / len(test_loader), correct / total


if __name__ == "__main__":
    train()
