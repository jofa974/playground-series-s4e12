from pathlib import Path

RAW_DATA_PATH = Path(__file__).parent.parent / "data/raw"
PREP_DATA_PATH = Path(__file__).parent.parent / "data/prepared"
FEATURIZER_PATH = Path(__file__).parent.parent / "data/featurizers"

OUT_PATH = Path(__file__).parent.parent / "out/"
PREDS_PATH = OUT_PATH / "preds"
