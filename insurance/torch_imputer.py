import pickle
from dataclasses import dataclass, field
from pathlib import Path

import dvc.api
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import typer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dvclive import Live
from insurance.common import OUT_PATH, PREP_DATA_PATH
from insurance.data_pipeline import Features

MODEL_PATH = OUT_PATH / "models/torch_imputer.pt"
DATA_PIPELINE_PATH = OUT_PATH / "models/torch_imputer_data_pipeline.pkl"

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
    # categorical = [""],
    # ordinal= ["Previous Claims"]
)


def make_pipeline(features: Features) -> Pipeline:
    # Preprocessing pipeline
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    log_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, validate=True)),
            ("scaler", StandardScaler()),
        ]
    )

    oh_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )

    ord_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    transformers = []
    if features.numeric:
        transformers.append(("num", numeric_transformer, features.numeric))
    if features.numeric_log:
        transformers.append(("num_log", log_transformer, features.numeric_log))
    if features.categorical:
        transformers.append(("oh", oh_transformer, features.categorical))
    if features.ordinal:
        transformers.append(("ord", ord_transformer, features.ordinal))

    preprocessor = ColumnTransformer(
        transformers=transformers, remainder="drop", verbose_feature_names_out=False
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
        ]
    )
    # pipeline.set_output(transform="pandas")
    return pipeline


# Custom Dataset class
class TabularDataset(Dataset):
    def __init__(self, features, targets=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        if targets:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = torch.tensor([torch.nan] * len(features))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# Define a simple feedforward neural network
class FeedforwardNN(nn.Module):
    def __init__(self, input_dim):
        super(FeedforwardNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)


@app.command()
def make_data_pipeline(prepared_data_path: Path):
    target_column = "Previous Claims"
    df = pd.read_feather(prepared_data_path)

    df = df[~df[target_column].isna()]
    features = df.drop(columns=[target_column])
    target = df[target_column]

    print("Initialized data")

    X_train, _, _, _ = train_test_split(features, target, test_size=0.2, random_state=42)

    data_pipeline = make_pipeline(features=features_columns)
    _ = data_pipeline.fit(X_train)
    pickle.dump(data_pipeline, DATA_PIPELINE_PATH.open("wb"))
    print(f"Saved data pipeline at {DATA_PIPELINE_PATH}")


@app.command()
def train(prepared_data_path: Path):
    params = dvc.api.params_show()["torch_imputer"]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Running torch on {device}")

    target_column = "Previous Claims"
    df = pd.read_feather(prepared_data_path)

    df = df[~df[target_column].isna()]
    features = df.drop(columns=[target_column])
    target = df[target_column]

    print("Initialized data")

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    data_pipeline = make_pipeline(features=features_columns)
    X_train = data_pipeline.fit_transform(X_train)
    X_test = data_pipeline.transform(X_test)

    print("Transformed trained data")
    pickle.dump(data_pipeline, DATA_PIPELINE_PATH.open("wb"))
    print(f"Saved data pipeline at {DATA_PIPELINE_PATH}")

    # Create PyTorch Datasets and DataLoaders
    train_dataset = TabularDataset(X_train, y_train.values)
    test_dataset = TabularDataset(X_test, y_test.values)

    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False)

    print("Pytorch data structures")

    # Initialize the model, loss function, and optimizer
    print(f"{X_train.shape=}")
    input_dim = X_train.shape[1]
    model = FeedforwardNN(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])

    print("NN + optim")

    with Live() as live:
        live.log_params(params)

        model.train()
        for epoch in range(params["epochs"]):
            epoch_loss = 0
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
            live.log_metric("loss/train", epoch_loss / len(train_loader))
            print(
                f"Epoch {epoch + 1}/{params['epochs']}, Loss: {epoch_loss / len(train_loader):.4f}"
            )

            eval_loss = evaluate_model(model=model, test_loader=test_loader, device=device)
            live.log_metric("loss/eval", eval_loss)

            live.next_step()

        torch.save(model.state_dict(), MODEL_PATH)
        live.log_artifact(MODEL_PATH, type="model", name="pytorch-model")


def evaluate_model(model, test_loader, device):
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        total_loss = 0
        for features, targets in tqdm(test_loader, desc="Evaluating"):
            features, targets = features.to(device), targets.to(device)
            outputs = model(features).squeeze()
            loss = criterion(outputs, targets)
            total_loss += loss.item()
        print(f"Test Loss: {total_loss / len(test_loader):.4f}")
    return total_loss / len(test_loader)


@app.command()
def run_inference(prepared_data_path: Path, out_file_name: str):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Running torch on {device}")

    target_column = "Previous Claims"
    df_init = pd.read_feather(prepared_data_path)

    X = df_init[df_init[target_column].isna()]
    X = X.drop(columns=[target_column])

    print("Initialized data")

    data_pipeline = pickle.load(DATA_PIPELINE_PATH.open("rb"))
    X_transformed = data_pipeline.transform(X)

    print("Transformed trained data")

    # Create PyTorch Datasets and DataLoaders
    inference_dataset = TabularDataset(X_transformed)

    inference_loader = DataLoader(inference_dataset, batch_size=1024)

    print("Pytorch data structures")

    # Initialize the model, loss function, and optimizer
    print(f"{X_transformed.shape=}")
    input_dim = X_transformed.shape[1]
    model = FeedforwardNN(input_dim).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    with torch.no_grad():
        preds = []
        for features, _ in tqdm(inference_loader, desc="Inference"):
            features = features.to(device)
            outputs = model(features).squeeze().to("cpu")
            preds.append(outputs)
        preds = torch.cat(preds)
        X[target_column] = preds.numpy()

    df_init.loc[df_init[target_column].isna(), target_column] = X[target_column].values
    out_file = PREP_DATA_PATH / out_file_name
    df_init.to_feather(out_file)
    print(f"Inference saved to {out_file}")
    return df_init


if __name__ == "__main__":
    app()
