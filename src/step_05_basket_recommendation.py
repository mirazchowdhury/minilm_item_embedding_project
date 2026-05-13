import math
import re
from collections import Counter
from difflib import get_close_matches
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from config import (
    FEATURE_ITEM_INDEX_PATH, COOCCURRENCE_MATRIX_PATH,
    ITEM_USER_SVD_EMBEDDING_PATH, COOCCURRENCE_SVD_EMBEDDING_PATH,
    TEXT_EMBEDDING_MATRIX_PATH, ITEM_POPULARITY_PATH,
)

from business_rule_seed import (
    infer_product_intent, rule_candidate_score,
    normalize_name, tokenize_name, detect_basket_intent, get_rule_profile
)

TOP_K = 5


def safe_divide(a, b):
    return 0.0 if b == 0 else float(a) / float(b)


def get_product_family(item_name):
    tokens = tokenize_name(item_name)
    family_rules = [
        ("beef", {"beef", "cow", "bone", "meat"}),
        ("chicken", {"chicken", "cock", "hen"}),
        ("fish", {"fish", "rui", "hilsha", "shrimp", "pabda"}),
        ("rice", {"rice", "chinigura", "miniket", "polao"}),
        ("atta_maida", {"atta", "maida", "flour"}),
        ("milk", {"milk", "dano", "diploma", "arla", "nido"}),
        ("sugar", {"sugar", "misri"}),
        ("oil", {"oil", "soyabean", "mustard"}),
        ("salt", {"salt"}),
        ("spice_masala", {"masala", "garam", "cumin", "turmeric", "chilli", "coriander"}),
        ("cleaning", {"vim", "harpic", "detergent", "cleaner", "soap", "tissue"}),
    ]
    for family, keywords in family_rules:
        if len(tokens.intersection(keywords)) > 0:
            return family
    return "other"


def hard_block_candidate(cart_intent, candidate_name):
    tokens = tokenize_name(candidate_name)

    # 1. Cleaning & Toiletries Blocking
    cleaning_tokens = {"vim", "harpic", "detergent", "soap", "shampoo", "tissue", "cleaner"}
    food_intents = {"beef_cooking", "chicken_cooking", "haleem_cooking", "ruti_meal", "noodles_cooking", "dessert",
                    "milk_tea_snacks"}
    if cart_intent in food_intents and len(tokens.intersection(cleaning_tokens)) > 0:
        return True

    # 2. Raw Meat/Fish Blocking (The "Beef Bone" Fix)
    raw_meat_fish = {"beef", "bone", "mutton", "chicken", "fish", "shrimp", "prawn"}
    if cart_intent in {"dessert", "milk_tea_snacks", "general", "cleaning", "breakfast"}:
        if len(tokens.intersection(raw_meat_fish)) > 0:
            return True

    # 3. Sweet/Dessert Blocking
    sweet_tokens = {"sugar", "semai", "suji", "biscuit", "cake", "pitha", "chocolate"}
    if cart_intent in {"beef_cooking", "chicken_cooking", "haleem_cooking", "fish_cooking"}:
        if len(tokens.intersection(sweet_tokens)) > 0:
            return True

    return False


def l2_normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def load_runtime_assets():
    catalog = pd.read_csv(FEATURE_ITEM_INDEX_PATH)
    cooccurrence_matrix = load_npz(COOCCURRENCE_MATRIX_PATH)
    item_user_embedding = np.load(ITEM_USER_SVD_EMBEDDING_PATH).astype(np.float32)
    cooccurrence_embedding = np.load(COOCCURRENCE_SVD_EMBEDDING_PATH).astype(np.float32)
    text_embedding = np.load(TEXT_EMBEDDING_MATRIX_PATH).astype(np.float32)

    item_user_embedding = l2_normalize(item_user_embedding)
    cooccurrence_embedding = l2_normalize(cooccurrence_embedding)
    text_embedding = l2_normalize(text_embedding)

    popularity_df = pd.read_csv(ITEM_POPULARITY_PATH)
    popularity_map = dict(zip(popularity_df["item_index"].astype(int), popularity_df["popularity_score"].astype(float)))

    return {
        "catalog": catalog, "cooccurrence_matrix": cooccurrence_matrix,
        "item_user_embedding": item_user_embedding, "cooccurrence_embedding": cooccurrence_embedding,
        "text_embedding": text_embedding, "popularity_map": popularity_map,
    }


def find_item_indexes(input_items, catalog):
    normalized_to_index = dict(zip(catalog["normalized_item_name"].astype(str), catalog["item_index"].astype(int)))
    normalized_names = list(normalized_to_index.keys())
    matched_indexes = []
    unmatched_items = []

    for item_name in input_items:
        item_norm = normalize_name(item_name)
        if item_norm in normalized_to_index:
            matched_indexes.append(normalized_to_index[item_norm])
            continue

        close_matches = get_close_matches(item_norm, normalized_names, n=1, cutoff=0.65)
        if close_matches:
            matched_indexes.append(normalized_to_index[close_matches[0]])
        else:
            unmatched_items.append(item_name)
    return sorted(set(matched_indexes)), unmatched_items


def cart_rule_score(cart_intent, candidate_name):
    profile = get_rule_profile(cart_intent)
    candidate_tokens = tokenize_name(candidate_name)

    if len(candidate_tokens.intersection(profile["bad"])) > 0:
        return -1.0

    must_match = len(candidate_tokens.intersection(profile["must_have"]))
    soft_match = len(candidate_tokens.intersection(profile["soft_good"]))

    score = 0.0
    if must_match > 0: score += 0.50 + min(must_match, 4) * 0.10
    if soft_match > 0: score += 0.20 + min(soft_match, 2) * 0.05
    return score


def get_pair_score_for_cart(input_indexes, candidate_index, cooccurrence_matrix):
    pair_counts = [float(cooccurrence_matrix[idx, candidate_index]) for idx in input_indexes]
    if not pair_counts: return 0.0, 0.0
    return sum(pair_counts), max(pair_counts)


def recommend_for_cart(input_items, top_k=TOP_K):
    assets = load_runtime_assets()
    catalog = assets["catalog"]
    cooccurrence_matrix = assets["cooccurrence_matrix"]
    text_embedding = assets["text_embedding"]
    popularity_map = assets["popularity_map"]

    input_indexes, unmatched_items = find_item_indexes(input_items, catalog)

    if len(input_indexes) == 0:
        print("No input items matched with catalog")
        return pd.DataFrame()

    matched_input_names = catalog.loc[input_indexes, "item_name"].tolist()
    cart_intent = detect_basket_intent(matched_input_names)
    cart_families = set(get_product_family(name) for name in matched_input_names)
    input_index_set = set(input_indexes)

    # Simplified embedding vectors for speed & accuracy
    text_cart_vector = np.mean([text_embedding[idx] for idx in input_indexes], axis=0)
    if np.linalg.norm(text_cart_vector) > 0:
        text_cart_vector = text_cart_vector / np.linalg.norm(text_cart_vector)

    candidate_set = set()
    for input_index in input_indexes:
        row = cooccurrence_matrix.getrow(input_index)
        if row.nnz > 0:
            top_order = np.argsort(row.data)[::-1][:300]
            candidate_set.update(row.indices[top_order].tolist())

    for candidate_index in range(len(catalog)):
        candidate_name = catalog.loc[candidate_index, "item_name"]
        score = cart_rule_score(cart_intent, candidate_name)
        if score > 0:
            candidate_set.add(candidate_index)

    scored_rows = []
    for candidate_index in candidate_set:
        if candidate_index in input_index_set: continue
        candidate_name = catalog.loc[candidate_index, "item_name"]

        # STRICT BLOCKING
        if hard_block_candidate(cart_intent, candidate_name):
            continue

        candidate_family = get_product_family(candidate_name)
        rule_score = cart_rule_score(cart_intent, candidate_name)

        # Drop totally irrelevant items
        if rule_score < 0:
            continue

        if candidate_family in cart_families:
            rule_score = max(0.0, rule_score - 0.25)

        total_pair_count, max_pair_count = get_pair_score_for_cart(input_indexes, candidate_index, cooccurrence_matrix)

        text_similarity = float(
            np.dot(text_cart_vector, text_embedding[candidate_index])) if text_cart_vector is not None else 0.0
        text_similarity = max(0.0, text_similarity)

        popularity_score = popularity_map.get(candidate_index, 0.0)
        max_pair_score = math.log1p(max_pair_count) / 8.0

        # Highly penalized historical data if rule score is 0
        if rule_score == 0 and total_pair_count < 5:
            continue

        final_score = (
                0.65 * rule_score
                + 0.20 * text_similarity
                + 0.10 * popularity_score
                + 0.05 * max_pair_score
        )

        scored_rows.append({
            "cart_intent": cart_intent,
            "input_items": ", ".join(matched_input_names),
            "recommended_item_name": candidate_name,
            "recommended_family": candidate_family,
            "rule_score": rule_score,
            "total_pair_count": total_pair_count,
            "final_score": final_score,
        })

    if len(scored_rows) == 0: return pd.DataFrame()

    result_df = pd.DataFrame(scored_rows)
    result_df = result_df.sort_values(by=["final_score", "rule_score", "total_pair_count"],
                                      ascending=[False, False, False])

    diverse_rows = []
    used_families = set()

    for _, row in result_df.iterrows():
        family = row.get("recommended_family", "other")
        if family != "other" and family in used_families: continue
        diverse_rows.append(row)
        if family != "other": used_families.add(family)
        if len(diverse_rows) >= top_k: break

    result_df = pd.DataFrame(diverse_rows).copy()
    result_df["rank"] = range(1, len(result_df) + 1)
    return result_df


if __name__ == "__main__":
    raw_input_text = input("Enter cart items separated by comma: ")
    input_items = [item.strip() for item in raw_input_text.split(",") if item.strip()]

    result = recommend_for_cart(input_items, top_k=5)

    if len(result) == 0:
        print("No recommendation found")
    else:
        print(f"\nDetected cart intent: {result.iloc[0]['cart_intent']}")
        print(f"Input items: {result.iloc[0]['input_items']}\n")
        print("Top recommendations:")
        for _, row in result.iterrows():
            print(
                f"{int(row['rank'])}. {row['recommended_item_name']} | Final score: {round(row['final_score'], 4)} | Rule score: {round(row['rule_score'], 4)} | Pair count: {int(row['total_pair_count'])}")