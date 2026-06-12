from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any


def configure_logging(name: str = "ufc_ml", log_dir: str | Path = "logs") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(Path(log_dir) / "scraper.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def append_error_row(error_path: str | Path, row: dict[str, Any]) -> None:
    path = Path(error_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["stage", "entity_id", "url", "message", "scraped_at"]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})