import os

from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent
ROOT = PACKAGE.parent

STATIC = PACKAGE / "static"

# local folder, mounted bucket, or fsspec URL (e.g. hf://buckets/<user>/<bucket>)
DATA = os.environ.get("PULSE_DATA_DIR") or str(ROOT / "data")
