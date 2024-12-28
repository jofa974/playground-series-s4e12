from typing import Annotated
import typer
import pandas as pd
from insurance.common import (
    OOF_PREDS_PATH,
    TARGET_COLUMN,
    PREP_DATA_PATH,
    OUT_PATH,
    RAW_DATA_PATH,
    PREDS_PATH,
)
import pickle
from insurance.logger import setup_logger
from insurance.data_pipeline import get_feat_columns, make_pipeline
from insurance.prepare_basic import prepare
import numpy as np
import dvc.api

logger = setup_logger(name="Data for boosters")


def get_booster_training_data(
    layer: Annotated[int, typer.Option(help="Stack layer number")],
    model_name: Annotated[str, typer.Option(help="Model name")],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = dvc.api.params_show()

    if layer == 0:
        df = pd.read_feather(PREP_DATA_PATH / "prepared_data.feather")
        logger.info("Transforming training data...")
        feat_cols = get_feat_columns()
        data_pipeline = make_pipeline()
        df = data_pipeline.fit_transform(df)
        for col in feat_cols.categorical:
            df[col] = df[col].astype("category")

        data_pipeline_path = OUT_PATH / f"data_pipeline_train_{model_name}.pkl"
        pickle.dump(data_pipeline, open(data_pipeline_path, "wb"))
        logger.info(f"Data pipeline saved at {data_pipeline_path}")

        df_test = pd.read_csv(RAW_DATA_PATH / "test.csv")
        logger.info("Transforming test data...")
        df_test = prepare(df=df_test)
        df_test.to_feather(PREP_DATA_PATH / "test_prepared.feather")
        df_test = data_pipeline.transform(df_test)

    else:
        prev_layer_models = list(params[f"layer_{layer-1}"].keys())
        df = pd.read_feather(OOF_PREDS_PATH / f"{prev_layer_models[0]}_layer_{layer-1}.feather")
        prev_oofs = [df]
        for i in range(1, len(prev_layer_models)):
            prev_oofs.append(
                pd.read_feather(OOF_PREDS_PATH / f"{prev_layer_models[i]}_layer_{layer-1}.feather")[
                    f"{prev_layer_models[i]}_layer_{layer-1}"
                ],
            )
        df = pd.concat(
            prev_oofs,
            axis=1,
        )

        df_test = pd.read_feather(PREDS_PATH / f"{prev_layer_models[0]}_layer_{layer-1}.feather")

    X_train = df.drop(columns=[TARGET_COLUMN])
    logger.info(f"Train shape: {X_train.shape=}")
    y_train = df[TARGET_COLUMN]
    y_train = np.log1p(y_train)

    logger.info(f"Train shape: {X_train.shape=}")
    logger.info(f"Columns: {X_train.columns=}")
    return X_train, y_train


def get_booster_test_data(layer: int) -> pd.DataFrame:
    df_test = pd.read_csv(RAW_DATA_PATH / "test.csv")
    ids = df_test["id"].values
    df_test = prepare(df=df_test)
    df_test.to_feather(PREP_DATA_PATH / "test_prepared.feather")


if __name__ == "__main__":
    typer.run(get_booster_training_data)
