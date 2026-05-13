from collections import Counter

import pandas as pd

from config import BASKET_DATA_PATH, ITEMS_COLUMN, ITEM_CATALOG_PATH, OUTPUT_DIR
from utils import ensure_dirs, parse_basket_items, build_product_text


def create_item_catalog() -> pd.DataFrame:
    ensure_dirs(OUTPUT_DIR)

    item_counter = Counter()
    basket_count = 0

    for chunk in pd.read_csv(BASKET_DATA_PATH, chunksize=20000):
        if ITEMS_COLUMN not in chunk.columns:
            raise ValueError(f"Column not found: {ITEMS_COLUMN}")

        for raw_items in chunk[ITEMS_COLUMN].dropna():
            items = parse_basket_items(raw_items)

            if not items:
                continue

            basket_count += 1
            item_counter.update(items)

    catalog = pd.DataFrame(
        [
            {
                "item_id": idx + 1,
                "item_name": item_name,
                "purchase_count": count,
                "product_text": build_product_text(item_name),
            }
            for idx, (item_name, count) in enumerate(item_counter.most_common())
        ]
    )

    catalog.to_csv(ITEM_CATALOG_PATH, index=False)

    print(f"Total valid baskets: {basket_count}")
    print(f"Total unique items: {len(catalog)}")
    print(f"Saved item catalog: {ITEM_CATALOG_PATH}")

    return catalog


if __name__ == "__main__":
    create_item_catalog()