import dvc.api
import numpy as np
import pandas as pd
from dvclive import Live
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
import pickle

from insurance.common import OUT_PATH, PREP_DATA_PATH


def main():
    params = dvc.api.params_show()["train"]

    model_path = OUT_PATH / "models/model_basic.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    target_column = "Premium Amount"

    df = pd.read_csv(PREP_DATA_PATH / "prepared_data.csv")
    features = df.drop(columns=[target_column])
    labels = df[target_column]

    numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = features.select_dtypes(include=["object", "category"]).columns.tolist()
    ordinal_features = ["Policy Start Date"]

    # Preprocessing pipeline
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    oh_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    ord_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("oh", oh_transformer, categorical_features),
            ("ord", ord_transformer, ordinal_features),
        ]
    )

    model = LinearRegression()
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # Stratified K-Fold Cross-Validation
    n_splits = params["n_splits"]
    skf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    with Live(OUT_PATH) as live:
        rmsle_scores = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
            print(f"Fold {fold + 1}")
            X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
            y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]

            # Fit the pipeline
            pipeline.fit(X_train, y_train)

            # Predict and evaluate
            predictions = pipeline.predict(X_test)
            rmsle = root_mean_squared_log_error(y_test, predictions)
            rmsle_scores.append(rmsle)
            print(f"Root Mean Squared Logarithmic Error: {rmsle:.4f}")
            live.log_metric(f"rmsle/{fold}", rmsle)

        # Overall performance
        average_rmsle = np.mean(rmsle_scores)
        live.log_metric("rmsle/average", average_rmsle)
        print(
            f"Average Mean Squared Logarithmic Error across {n_splits} folds: {average_rmsle:.4f}"
        )
        model_path = OUT_PATH / "models"
        model_path.mkdir(parents=True, exist_ok=True)
        pickle.dump(
            pipeline,
            (model_path / "model_basic.pkl").open("wb"),
        )
        live.log_artifact(OUT_PATH / "models/model_basic.pkl")

    # # Inference on a new dataset
    # inference_data = pd.read_csv(inference_data_path)

    # # Make predictions
    # predictions = pipeline.predict(inference_data)

    # # Prepare submission file
    # submission = pd.DataFrame(
    #     {
    #         "Id": inference_data.index,  # Replace 'Id' with appropriate identifier column if needed
    #         "Prediction": predictions,
    #     }
    # )
    # submission.to_csv(submission_file_path, index=False)
    # print(f"Submission file saved to {submission_file_path}")


if __name__ == "__main__":
    main()
