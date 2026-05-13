import math
from difflib import get_close_matches

import pandas as pd

from item_intelligence import normalize_text, tokenize, detect_category, detect_family
from basket_intent_engine import detect_basket_intent, get_intent_profile


def load_catalog(catalog_path):
    df = pd.read_csv(catalog_path)

    required_columns = {
        "item_name",
        "normalized_item_name",
        "category",
        "family",
        "item_index"
    }

    missing_columns = required_columns.difference(set(df.columns))

    if missing_columns:
        raise ValueError(f"Missing columns in catalog: {missing_columns}")

    return df


def fuzzy_match_items(input_items, catalog, cutoff=0.72):
    norm_to_name = dict(
        zip(catalog["normalized_item_name"], catalog["item_name"])
    )

    normalized_names = list(norm_to_name.keys())

    matched_items = []
    unmatched_items = []

    for item in input_items:
        norm = normalize_text(item)

        if norm in norm_to_name:
            matched_items.append(norm_to_name[norm])
            continue

        match = get_close_matches(
            norm,
            normalized_names,
            n=1,
            cutoff=cutoff
        )

        if match:
            matched_items.append(norm_to_name[match[0]])
        else:
            unmatched_items.append(item)

    return list(dict.fromkeys(matched_items)), unmatched_items


def family_need_score(intent, candidate_family, cart_families):
    profile = get_intent_profile(intent)

    required_families = profile["required_families"]

    if candidate_family in cart_families:
        return 0.0

    if candidate_family in required_families:
        return 1.0

    return 0.0


def category_score(intent, candidate_category):
    profile = get_intent_profile(intent)

    if candidate_category in profile["bad_categories"]:
        return -1.0

    if candidate_category in profile["good_categories"]:
        return 0.7

    return 0.0


def token_relevance_score(intent, candidate_name):
    tokens = tokenize(candidate_name)

    intent_tokens = {
        "beef_cooking": {
            "beef", "onion", "garlic", "ginger", "chili", "masala",
            "garam", "biryani", "rice", "oil", "salt", "potato",
            "lemon", "borhani", "coriander", "yogurt"
        },

        "chicken_cooking": {
            "chicken", "onion", "garlic", "ginger", "chili", "masala",
            "roast", "garam", "rice", "oil", "salt", "potato",
            "egg", "lemon", "borhani", "yogurt"
        },

        "fish_cooking": {
            "fish", "onion", "garlic", "ginger", "chili", "tomato",
            "coriander", "masala", "turmeric", "mustard", "oil",
            "salt", "lemon", "rice"
        },

        "biryani_cooking": {
            "biryani", "kachchi", "rice", "chinigura", "onion",
            "garlic", "ginger", "chili", "garam", "masala",
            "yogurt", "ghee", "raisin", "potato", "borhani",
            "cucumber", "salt", "oil"
        },

        "daily_meal_cooking": {
            "rice", "onion", "garlic", "ginger", "chili", "oil",
            "salt", "fish", "chicken", "beef", "pulse", "dal",
            "potato", "tomato", "masala", "turmeric", "coriander"
        },

        "dessert": {
            "sugar", "milk", "powder", "semai", "lacha", "vermicelli",
            "suji", "sagu", "ghee", "raisin", "cardamom", "cinnamon",
            "custard", "firni", "kheer", "falooda", "dates", "honey"
        },

        "breakfast": {
            "bread", "egg", "milk", "butter", "cheese", "jam",
            "tea", "coffee", "banana", "corn", "flakes", "oats",
            "sausage", "paratha"
        },

        "snacks_party": {
            "coke", "cola", "sprite", "pepsi", "fanta", "chips",
            "kurkure", "nuggets", "sausage", "fries", "sauce",
            "ketchup", "burger", "bread", "chocolate", "ice", "cream"
        },

        "cleaning": {
            "vim", "dishwash", "sponge", "tissue", "towel",
            "harpic", "detergent", "soap", "handwash", "floor",
            "cleaner", "toilet", "air", "freshener"
        },
    }

    target_tokens = intent_tokens.get(intent, intent_tokens["daily_meal_cooking"])
    overlap = len(tokens.intersection(target_tokens))

    if overlap == 0:
        return 0.0

    return min(1.0, overlap / 3.0)


def brand_priority_score(candidate_name):
    text = normalize_text(candidate_name)

    strong_brands = {
        "radhuni", "aci", "teer", "fresh", "pusti", "rupchanda",
        "aarong", "diploma", "prince", "bashundhara", "maggi",
        "knorr", "kazi", "gh", "cp", "pran"
    }

    tokens = set(text.split())

    if tokens.intersection(strong_brands):
        return 0.2

    return 0.0


def popularity_proxy_score(candidate_name):
    text = normalize_text(candidate_name)

    common_words = {
        "onion", "garlic", "ginger", "chili", "rice", "oil",
        "salt", "sugar", "milk", "bread", "egg", "potato",
        "fish", "chicken", "beef", "masala", "tea"
    }

    tokens = set(text.split())

    if tokens.intersection(common_words):
        return 0.3

    return 0.0


def recommend_from_catalog(input_items, catalog_path, top_k=10):
    catalog = load_catalog(catalog_path)

    matched_items, unmatched_items = fuzzy_match_items(input_items, catalog)

    if not matched_items:
        return {
            "matched_items": [],
            "unmatched_items": unmatched_items,
            "intent": "unknown",
            "recommendations": []
        }

    intent = detect_basket_intent(matched_items)

    cart_families = set(detect_family(item) for item in matched_items)
    cart_normalized = set(normalize_text(item) for item in matched_items)

    scored_rows = []

    for _, row in catalog.iterrows():
        candidate_name = row["item_name"]
        candidate_norm = row["normalized_item_name"]
        candidate_category = row["category"]
        candidate_family = row["family"]

        if candidate_norm in cart_normalized:
            continue

        cat_score = category_score(intent, candidate_category)

        if cat_score < 0:
            continue

        fam_score = family_need_score(intent, candidate_family, cart_families)
        tok_score = token_relevance_score(intent, candidate_name)
        brand_score = brand_priority_score(candidate_name)
        pop_score = popularity_proxy_score(candidate_name)

        final_score = (
            0.40 * fam_score
            + 0.30 * tok_score
            + 0.15 * cat_score
            + 0.08 * pop_score
            + 0.07 * brand_score
        )

        if final_score <= 0:
            continue

        scored_rows.append({
            "recommended_item": candidate_name,
            "category": candidate_category,
            "family": candidate_family,
            "family_need_score": round(fam_score, 4),
            "token_relevance_score": round(tok_score, 4),
            "category_score": round(cat_score, 4),
            "brand_score": round(brand_score, 4),
            "popularity_proxy_score": round(pop_score, 4),
            "final_score": round(final_score, 4),
        })

    result_df = pd.DataFrame(scored_rows)

    if result_df.empty:
        return {
            "matched_items": matched_items,
            "unmatched_items": unmatched_items,
            "intent": intent,
            "recommendations": []
        }

    result_df = result_df.sort_values(
        by=[
            "final_score",
            "family_need_score",
            "token_relevance_score",
            "popularity_proxy_score"
        ],
        ascending=[False, False, False, False]
    )

    result_df = result_df.drop_duplicates(subset=["family"], keep="first")
    result_df = result_df.head(top_k).reset_index(drop=True)
    result_df["rank"] = result_df.index + 1

    return {
        "matched_items": matched_items,
        "unmatched_items": unmatched_items,
        "intent": intent,
        "recommendations": result_df.to_dict(orient="records")
    }


if __name__ == "__main__":
    catalog_path = "data/output/enriched_item_catalog.csv"

    raw_text = input("Enter cart items separated by comma: ")

    input_items = [
        item.strip()
        for item in raw_text.split(",")
        if item.strip()
    ]

    output = recommend_from_catalog(
        input_items=input_items,
        catalog_path=catalog_path,
        top_k=10
    )

    print()
    print("Matched items:")
    for item in output["matched_items"]:
        print(item)

    print()
    print("Unmatched items:")
    for item in output["unmatched_items"]:
        print(item)

    print()
    print("Detected intent:")
    print(output["intent"])

    print()
    print("Top recommendations:")

    for row in output["recommendations"]:
        print(
            f"{row['rank']}. {row['recommended_item']} "
            f"| score: {row['final_score']} "
            f"| category: {row['category']} "
            f"| family: {row['family']}"
        )