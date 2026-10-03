from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent
ROOT = PACKAGE.parent

STATIC = PACKAGE / "static"

DATA = ROOT / "data"
TASKS = DATA / "tasks"
RESULTS = DATA / "results"

DOCS = DATA / "docs"
COMPLETIONS = DOCS / "completions"
PERSONAS = DOCS / "personas"
