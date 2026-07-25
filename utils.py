"""
Utility functions including logging setup and schema validation.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd

from config import LOG_DIR


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a configured logger that logs to both console and file.
    """

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name if name else __name__)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console logger
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File logger
    file_handler = logging.FileHandler(
        LOG_DIR / "app.log",
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def validate_recommendation_schema(
    items: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Clean recommendation objects so they satisfy the FastAPI response model.
    """

    validated = []

    for item in items:

        item = item.copy()

        # -----------------------------
        # URL
        # -----------------------------
        item["url"] = str(item.get("url", ""))

        # -----------------------------
        # Name
        # -----------------------------
        item["name"] = str(item.get("name", ""))

        # -----------------------------
        # Description
        # -----------------------------
        item["description"] = str(item.get("description", ""))

        # -----------------------------
        # Adaptive Support
        # -----------------------------
        item["adaptive_support"] = str(
            item.get("adaptive_support", "No")
        )

        # -----------------------------
        # Remote Support
        # -----------------------------
        item["remote_support"] = str(
            item.get("remote_support", "No")
        )

        # -----------------------------
        # Duration
        # -----------------------------
        duration = item.get("duration")

        if pd.isna(duration):
            item["duration"] = None
        else:
            try:
                item["duration"] = int(float(duration))
            except Exception:
                item["duration"] = None

        # -----------------------------
        # Test Type
        # -----------------------------
        test_type = item.get("test_type", [])

        if pd.isna(test_type):
            test_type = []

        elif isinstance(test_type, str):

            try:
                parsed = ast.literal_eval(test_type)

                if isinstance(parsed, list):
                    test_type = parsed
                else:
                    test_type = [str(parsed)]

            except Exception:
                test_type = [test_type]

        elif not isinstance(test_type, list):
            test_type = [str(test_type)]

        item["test_type"] = test_type

        validated.append(item)

    return validated