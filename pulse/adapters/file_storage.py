"""`Storage` on files: a local folder, a mounted bucket, or any fsspec URL (e.g. hf://buckets/<user>/<bucket>).

Layout under the root:
    tasks/<poll>.yaml
    docs/personas/<name>.json|csv
    docs/completions/<name>.json|csv
    results/<poll>-<model>.json
"""

import re
import json
import logging
import posixpath

from copy import copy
from typing import Final
from pathlib import Path
from dataclasses import asdict

import yaml
import pandas as pd

from fsspec.core import url_to_fs

from pulse.ports import Storage
from pulse.domain.poll import Status, TableKind, PulseConfig, PulseResults, check_table

_LOGGER: Final = logging.getLogger(__name__)

POLLS_DIR: Final = "tasks"
RESULTS_DIR: Final = "results"
TABLE_DIRS: Final[dict[TableKind, str]] = {
    "personas": "docs/personas",
    "completions": "docs/completions",
}


class FileStorage(Storage):
    """Reads everything once at start-up and writes changes through to the files."""

    def __init__(self, root: str | Path):
        self.fs, self.root = url_to_fs(str(root))

        self._polls: dict[str, PulseConfig] = {}
        for path in self._glob(POLLS_DIR, "*.yaml"):
            poll = PulseConfig(**yaml.safe_load(self._read(path)))
            self._polls[poll.name] = poll

        self._tables: dict[TableKind, dict[str, tuple[str, pd.DataFrame]]] = {kind: {} for kind in TABLE_DIRS}
        for kind, folder in TABLE_DIRS.items():
            for path in self._glob(folder, "*.json") + self._glob(folder, "*.csv"):
                df = self._read_table(path)
                if (status := check_table(kind=kind, df=df)) != Status.OK:
                    _LOGGER.warning("Skipping %s: %s", path, status)
                    continue
                self._tables[kind][_stem(path)] = (path, df)

        self._results: dict[str, PulseResults] = {}
        for path in self._glob(RESULTS_DIR, "*.json"):
            data = json.loads(self._read(path))
            self._results[_stem(path)] = PulseResults(
                task=data["task"],
                model=data["model"],
                metrics=data["metrics"],
                dataset={"docs": data.get("docs", {}), "completions": data.get("completions", {})},
            )

    # polls
    def poll_names(self) -> list[str]:
        return list(self._polls)

    def get_poll(self, name: str) -> PulseConfig:
        return copy(self._polls[name])

    def save_poll(self, poll: PulseConfig) -> Status:
        if any(poll == saved for saved in self._polls.values()):
            return Status.DUPLICATE_POLL

        text = yaml.safe_dump(asdict(poll), sort_keys=False, allow_unicode=True)
        self._write(self._path(POLLS_DIR, f"{poll.name}.yaml"), text)
        self._polls[poll.name] = copy(poll)
        return Status.OK

    def delete_poll(self, name: str) -> Status:
        if name not in self._polls:
            return Status.NOT_FOUND

        self.fs.rm(self._path(POLLS_DIR, f"{name}.yaml"))
        del self._polls[name]
        return Status.OK

    # personas and completions tables
    def table_names(self, kind: TableKind) -> list[str]:
        return list(self._tables[kind])

    def get_table(self, kind: TableKind, name: str) -> pd.DataFrame:
        _, df = self._tables[kind][name]
        return df

    def add_table(self, kind: TableKind, name: str, df: pd.DataFrame) -> Status:
        if name in self._tables[kind]:
            return Status.DUPLICATE
        if (status := check_table(kind=kind, df=df)) != Status.OK:
            return status

        path = self._path(TABLE_DIRS[kind], f"{name}.json")
        self._write_table(path, df)
        self._tables[kind][name] = (path, df)
        return Status.OK

    def update_table(self, kind: TableKind, name: str, df: pd.DataFrame) -> Status:
        if name not in self._tables[kind]:
            return Status.NOT_FOUND
        if (status := check_table(kind=kind, df=df)) != Status.OK:
            return status

        path, _ = self._tables[kind][name]
        self._write_table(path, df)
        self._tables[kind][name] = (path, df)
        return Status.OK

    def delete_table(self, kind: TableKind, name: str) -> Status:
        if name not in self._tables[kind]:
            return Status.NOT_FOUND

        path, _ = self._tables[kind].pop(name)
        self.fs.rm(path)
        return Status.OK

    # results
    def results(self) -> list[PulseResults]:
        return list(self._results.values())

    def save_results(self, results: PulseResults) -> None:
        model = re.sub(r"[\"<>:/|\\?*\[\]]+", "__", results.model)  # same file names as lm-eval's sanitizer
        key = f"{results.task}-{model}"

        payload = {
            "task": results.task,
            "model": results.model,
            "docs": results.docs,
            "completions": results.completions,
            "metrics": results.metrics,
        }
        self._write(self._path(RESULTS_DIR, f"{key}.json"), json.dumps(payload, indent=2, ensure_ascii=False))
        self._results[key] = results

    # files
    def _path(self, folder: str, filename: str) -> str:
        return posixpath.join(self.root, folder, filename)

    def _glob(self, folder: str, pattern: str) -> list[str]:
        return sorted(self.fs.glob(self._path(folder, pattern)))

    def _read(self, path: str) -> str:
        with self.fs.open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _write(self, path: str, text: str) -> None:
        self.fs.makedirs(posixpath.dirname(path), exist_ok=True)
        with self.fs.open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def _read_table(self, path: str) -> pd.DataFrame:
        with self.fs.open(path, "r", encoding="utf-8") as f:
            return pd.read_csv(f) if path.endswith(".csv") else pd.read_json(f)

    def _write_table(self, path: str, df: pd.DataFrame) -> None:
        if path.endswith(".csv"):
            self._write(path, df.to_csv(index=False))
        else:
            self._write(path, df.to_json(orient="records", indent=4))


def _stem(path: str) -> str:
    return posixpath.splitext(posixpath.basename(path))[0]
