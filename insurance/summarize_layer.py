import typer
from typing import Annotated
from insurance.common import OOF_PREDS_PATH, PREDS_PATH
from insurance.logger import setup_logger
import pandas as pd


def main(
    layer: Annotated[int, typer.Option(help="Stack layer number")],
):
    logger = setup_logger(name=f"Summarize Layer {layer}")

    layer_oof_dir = OOF_PREDS_PATH / f"layer_{layer}/"
    preds_dir = PREDS_PATH / f"layer_{layer}/"

    for dir in [layer_oof_dir, preds_dir]:
        res = []
        for results in dir.glob("*.feather"):
            res.append(pd.read_feather(results))
        res = pd.concat(res, axis=1)
        res_file = dir.parent / f"layer_{layer}_concatenated.feather"
        res.to_feather(res_file)
        logger.info(f"{res_file} saved")


if __name__ == "__main__":
    typer.run(main)
