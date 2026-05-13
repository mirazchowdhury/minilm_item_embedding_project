import pandas as pd

from config import TOPK_RECOMMENDATION_PATH


def recommend_by_name(search_text, top_k=5):
    if not TOPK_RECOMMENDATION_PATH.exists():
        raise FileNotFoundError(
            f"Recommendation file not found: {TOPK_RECOMMENDATION_PATH}"
        )

    df = pd.read_csv(TOPK_RECOMMENDATION_PATH)

    search_text = str(search_text).lower().strip()

    matched = df[
        df["source_item_name"]
        .astype(str)
        .str.lower()
        .str.contains(search_text, na=False)
    ]

    if len(matched) == 0:
        print("No matching item found")
        return

    source_name = matched.iloc[0]["source_item_name"]

    result = df[df["source_item_name"] == source_name].copy()
    result = result.sort_values("rank").head(top_k)

    print()
    print("Input item:")
    print(source_name)

    print()
    print("Top recommendations:")

    for _, row in result.iterrows():
        print(
            f"{int(row['rank'])}. "
            f"{row['recommended_item_name']} "
            f"| Final score: {round(row['final_score'], 4)} "
            f"| Rule score: {round(row.get('rule_seed_score', 0), 4)} "
            f"| Pair count: {int(row['pair_count'])} "
            f"| Type: {row.get('recommendation_type', '')}"
        )


if __name__ == "__main__":
    item_name = input("Enter item name: ")
    recommend_by_name(item_name, top_k=5)