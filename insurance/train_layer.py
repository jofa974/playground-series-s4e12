from typing import Annotated

import dvc.api
import pandas as pd
import typer


from insurance.common import OOF_PREDS_PATH, PREDS_PATH, PREP_DATA_PATH
from insurance.logger import setup_logger
from insurance.boosters.xgboost import train as xgboost_train
from insurance.boosters.catboost import train as catboost_train
from insurance.boosters.lgbm import train as lgbm_train


def main(
    layer: Annotated[int, typer.Option(help="Stack layer number")],
):
    logger = setup_logger(name=f"Layer {layer}")

    params = dvc.api.params_show()
    params = params[f"layer_{layer}"]

    if layer == 0:
        train_data = pd.read_feather(PREP_DATA_PATH / "train_data.feather")
        test_data = pd.read_feather(PREP_DATA_PATH / "test_data.feather")
    else:
        train_data = pd.read_feather(OOF_PREDS_PATH / f"layer_{layer-1}.feather")
        test_data = pd.read_feather(PREDS_PATH / f"layer_{layer-1}.feather")

    next_train = train_data.copy()
    next_test = test_data.copy()
    for model_name, model_def in params.items():
        if model_def["type"] == "xgboost":
            _oof_preds, _avg_preds = xgboost_train(
                params=model_def["params"],
                model_name=model_name,
                layer=layer,
                train_data=train_data,
                test_data=test_data,
            )
        elif model_def["type"] == "catboost":
            _oof_preds, _avg_preds = catboost_train(
                params=model_def["params"],
                model_name=model_name,
                layer=layer,
                train_data=train_data,
                test_data=test_data,
            )
        elif model_def["type"] == "lgbm":
            _oof_preds, _avg_preds = lgbm_train(
                params=model_def["params"],
                model_name=model_name,
                layer=layer,
                train_data=train_data,
                test_data=test_data,
            )
        else:
            ValueError(f"Unknown model {model_def['type']}")

        next_train[f"{model_name}_preds"] = _oof_preds
        next_test[f"{model_name}_preds"] = _avg_preds

    OOF_PREDS_PATH.mkdir(parents=True, exist_ok=True)
    oof_path = OOF_PREDS_PATH / f"layer_{layer}.feather"
    next_train.to_feather(oof_path)
    logger.info(f"OOF for layer {layer} saved at {oof_path}")

    PREDS_PATH.mkdir(parents=True, exist_ok=True)
    layer_pred_path = PREDS_PATH / f"layer_{layer}.feather"
    next_test.to_feather(layer_pred_path)
    logger.info(
        f"Average models predictions on test data for layer {layer} saved at {layer_pred_path}"
    )


if __name__ == "__main__":
    typer.run(main)
