import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import (
    BATCH_SIZE,
    TEXT_EMBEDDING_MATRIX_PATH,
    TEXT_EMBEDDING_METADATA_PATH,
    ITEM_CATALOG_PATH,
    MODEL_DIR,
    MODEL_NAME,
    NORMALIZE_EMBEDDINGS,
    OUTPUT_DIR,
)

from utils import ensure_dirs, get_device, print_device_info


def generate_embeddings() -> np.ndarray:
    ensure_dirs(OUTPUT_DIR, MODEL_DIR)

    if TEXT_EMBEDDING_MATRIX_PATH.exists() and TEXT_EMBEDDING_METADATA_PATH.exists():
        print("MiniLM embedding already exists. Skipping embedding generation.")
        embeddings = np.load(TEXT_EMBEDDING_MATRIX_PATH).astype(np.float32)
        print(f"Loaded existing embedding shape: {embeddings.shape}")
        return embeddings

    if not ITEM_CATALOG_PATH.exists():
        raise FileNotFoundError(
            "Item catalog not found. Run step_01_create_item_catalog.py first."
        )

    catalog = pd.read_csv(ITEM_CATALOG_PATH)

    if "product_text" not in catalog.columns:
        raise ValueError("product_text column not found in item catalog.")

    texts = catalog["product_text"].fillna("").astype(str).tolist()

    print_device_info()

    model = SentenceTransformer(
        MODEL_NAME,
        device=get_device()
    )

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=NORMALIZE_EMBEDDINGS,
    )

    np.save(TEXT_EMBEDDING_MATRIX_PATH, embeddings)

    metadata = catalog[
        [
            "item_id",
            "item_name",
            "purchase_count",
            "product_text",
        ]
    ].copy()

    metadata["embedding_model"] = MODEL_NAME
    metadata["embedding_dim"] = embeddings.shape[1]

    metadata.to_csv(TEXT_EMBEDDING_METADATA_PATH, index=False)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Saved embedding matrix: {TEXT_EMBEDDING_MATRIX_PATH}")
    print(f"Saved metadata: {TEXT_EMBEDDING_METADATA_PATH}")

    return embeddings


if __name__ == "__main__":
    generate_embeddings()