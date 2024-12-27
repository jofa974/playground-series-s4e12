from pathlib import Path

TARGET_COLUMN = "Premium Amount"

RAW_DATA_PATH = Path(__file__).parent.parent / "data/raw"
PREP_DATA_PATH = Path(__file__).parent.parent / "data/prepared"
FEATURIZER_PATH = Path(__file__).parent.parent / "data/featurizers"

OOF_PREDS_PATH = Path(__file__).parent.parent / "data/oof/"

OUT_PATH = Path(__file__).parent.parent / "out/"
PREDS_PATH = OUT_PATH / "preds"
