import ast
import json
import re
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import torch


def ensure_dirs(*paths: Path) -> None:
    # দরকারি folder না থাকলে তৈরি করা হচ্ছে
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def normalize_item_name(text: str) -> str:
    # Product name cleaning, যাতে একই ধরনের spacing সমস্যা কমে
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_basket_items(value) -> List[str]:
    # Basket column JSON list বা Python list string হলে সেটি item list এ convert করা হচ্ছে
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if isinstance(value, list):
        return [normalize_item_name(x) for x in value if normalize_item_name(x)]

    value = str(value).strip()
    if not value:
        return []

    try:
        parsed = json.loads(value)
    except Exception:
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            parsed = []

    if isinstance(parsed, list):
        return [normalize_item_name(x) for x in parsed if normalize_item_name(x)]

    return []


def get_device() -> str:
    # CUDA থাকলে GPU ব্যবহার হবে, না থাকলে CPU ব্যবহার হবে
    return "cuda" if torch.cuda.is_available() else "cpu"


def print_device_info() -> None:
    device = get_device()
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(0)}")


def build_product_text(item_name: str, category: str = "") -> str:
    # Text embedding এর জন্য product information sentence আকারে তৈরি করা হচ্ছে
    item_name = normalize_item_name(item_name)
    category = normalize_item_name(category)

    if category:
        return f"Product name: {item_name}. Category: {category}."
    return f"Product name: {item_name}."


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    # Cosine similarity সহজ করার জন্য vector normalize করা হচ্ছে
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
