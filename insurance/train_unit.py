from typing import Annotated

import dvc.api
import pandas as pd
import typer
import numpy as np

from insurance.common import OOF_PREDS_PATH, PREDS_PATH, PREP_DATA_PATH, ModelType
from insurance.logger import setup_logger
from insurance.boosters.xgboost_unit import train as xgboost_train
from insurance.boosters.catboost_unit import train as catboost_train
from insurance.boosters.lgbm_unit import train as lgbm_train


def main(
    layer: Annotated[int, typer.Option(help="Stack layer number")],
    model_name: Annotated[str, typer.Option(help="Model name")],
    model_type: Annotated[ModelType, typer.Option(help="Model type")],
):
    logger = setup_logger(name=f"Layer {layer}")

    params = dvc.api.params_show()
    params = params[f"layer_{layer}"][model_name]["params"]

    train_data = pd.read_feather(PREP_DATA_PATH / "train_data.feather")
    test_data = pd.read_feather(PREP_DATA_PATH / "test_data.feather")

    if layer != 0:
        train_data = pd.concat(
            [train_data, pd.read_feather(OOF_PREDS_PATH / f"layer_{layer-1}.feather")], axis=1
        )
        test_data = pd.concat(
            [test_data, pd.read_feather(PREDS_PATH / f"layer_{layer-1}.feather")], axis=1
        )
        for col in train_data.columns:
            if "_preds" in col:
                train_data[col] = np.log1p(train_data[col])
        for col in test_data.columns:
            if "_preds" in col:
                test_data[col] = np.log1p(test_data[col])

    if model_type == ModelType.XGBOOST:
        _oof_preds, _avg_preds = xgboost_train(
            params=params,
            model_name=model_name,
            layer=layer,
            train_data=train_data,
            test_data=test_data,
        )
    elif model_type == ModelType.CATBOOST:
        _oof_preds, _avg_preds = catboost_train(
            params=params,
            model_name=model_name,
            layer=layer,
            train_data=train_data,
            test_data=test_data,
        )
    elif model_type == ModelType.LGBM:
        _oof_preds, _avg_preds = lgbm_train(
            params=params,
            model_name=model_name,
            layer=layer,
            train_data=train_data,
            test_data=test_data,
        )
    else:
        ValueError(f"Unknown model {model_type.value}")

    layer_oof_dir = OOF_PREDS_PATH / f"layer_{layer}/"
    layer_oof_dir.mkdir(parents=True, exist_ok=True)
    layer_oof_path = layer_oof_dir / f"model_{model_name}.feather"
    pd.DataFrame({f"{model_name}_preds": _oof_preds}).to_feather(layer_oof_path)
    logger.info(f"OOF for layer {layer} and model {model_name} saved at {layer_oof_path}")

    layer_pred_dir = PREDS_PATH / f"layer_{layer}/"
    layer_pred_dir.mkdir(parents=True, exist_ok=True)
    layer_pred_path = layer_pred_dir / f"model_{model_name}.feather"
    pd.DataFrame({f"{model_name}_preds": _avg_preds}).to_feather(layer_pred_path)
    logger.info(
        f"Average models predictions on test data for layer {layer} and model {model_name} saved at {layer_pred_path}"
    )


if __name__ == "__main__":
    typer.run(main)
