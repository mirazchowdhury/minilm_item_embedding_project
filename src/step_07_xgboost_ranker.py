import numpy as np
import pandas as pd
import math
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.sparse import load_npz

from config import (
    FEATURE_ITEM_INDEX_PATH, COOCCURRENCE_MATRIX_PATH,
    TEXT_EMBEDDING_MATRIX_PATH, ITEM_POPULARITY_PATH, MODEL_DIR
)
from business_rule_seed import rule_candidate_score, detect_basket_intent
from item_intelligence import get_product_family


def l2_normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def generate_training_data():
    print("Loading assets for XGBoost Training...")
    catalog = pd.read_csv(FEATURE_ITEM_INDEX_PATH)
    cooccurrence_matrix = load_npz(COOCCURRENCE_MATRIX_PATH)
    text_embedding = l2_normalize(np.load(TEXT_EMBEDDING_MATRIX_PATH).astype(np.float32))

    popularity_df = pd.read_csv(ITEM_POPULARITY_PATH)
    popularity_map = dict(zip(popularity_df["item_index"], popularity_df["popularity_score"]))

    features = []
    labels = []

    num_items = len(catalog)
    # We will sample interactions to build training data
    print("Generating positive and negative pairs...")

    for source_idx in range(num_items):
        source_name = catalog.loc[source_idx, "item_name"]
        source_intent = detect_basket_intent([source_name])
        source_family = get_product_family(source_name)

        # Get co-purchased items (Positive Samples)
        row = cooccurrence_matrix.getrow(source_idx)
        if row.nnz == 0:
            continue

        positive_indices = row.indices
        pair_counts = row.data

        for p_idx, p_count in zip(positive_indices, pair_counts):
            if source_idx == p_idx: continue

            cand_name = catalog.loc[p_idx, "item_name"]

            # Features
            rule_score = rule_candidate_score(source_name, cand_name)
            text_sim = float(np.dot(text_embedding[source_idx], text_embedding[p_idx]))
            pop_score = popularity_map.get(p_idx, 0.0)
            same_family = 1 if source_family == get_product_family(cand_name) else 0

            features.append([rule_score, text_sim, pop_score, same_family])
            labels.append(1)  # Positive label (They were bought together)

        # Generate Negative Samples (Random items not bought together)
        negative_samples = np.random.choice(num_items, size=min(10, row.nnz), replace=False)
        for n_idx in negative_samples:
            if n_idx in positive_indices or n_idx == source_idx: continue

            cand_name = catalog.loc[n_idx, "item_name"]

            # Features
            rule_score = rule_candidate_score(source_name, cand_name)
            text_sim = float(np.dot(text_embedding[source_idx], text_embedding[n_idx]))
            pop_score = popularity_map.get(n_idx, 0.0)
            same_family = 1 if source_family == get_product_family(cand_name) else 0

            features.append([rule_score, text_sim, pop_score, same_family])
            labels.append(0)  # Negative label

    return np.array(features), np.array(labels)


def train_ranker():
    X, y = generate_training_data()
    print(f"Total Training Samples Generated: {X.shape[0]}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost Classifier (Using as a Ranker)
    print("Training XGBoost Model...")
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="auc",
        use_label_encoder=False
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=10)

    # Evaluate
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    print(f"\nModel Evaluation AUC Score: {auc:.4f}")

    # Feature Importance
    print("\nFeature Importances:")
    feature_names = ["Rule_Score", "Text_Similarity", "Popularity", "Same_Family_Penalty"]
    for name, imp in zip(feature_names, model.feature_importances_):
        print(f"{name}: {imp:.4f}")

    # Save Model
    model_path = MODEL_DIR / "xgboost_ranker_model.json"
    model.save_model(model_path)
    print(f"\nModel saved successfully at: {model_path}")


if __name__ == "__main__":
    train_ranker()