import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from src.api_recommendation_rules import (
    get_product_family,
    infer_basket_intent,
    rule_boost_for_basket,
    hard_block_candidate,
    same_type_penalty
)


# Change class name from RecommenderService to BasketRecommenderService
class BasketRecommenderService:
    def __init__(self, project_root):
        self.project_root = Path(project_root)

        self.data_dir = self.project_root / "data"
        self.output_dir = self.data_dir / "output"
        self.model_dir = self.project_root / "models"

        self.catalog_file = self.output_dir / "item_catalog.csv"
        self.item_index_file = self.model_dir / "item_index.csv"

        self.catalog_df = None
        self.item_index_df = None

        self.item_id_to_name = {}
        self.item_id_to_category = {}
        self.item_name_to_ = {}
        self.item_name_to_category = {}

        self.items = []
        self.item_to_idx = {}
        self.idx_to_item = {}

        self.co_matrix = None
        self.main_item_embedding = None
        self.backup_item_embedding = None
        self.text_embedding = None

        self.load_assets()


    def load_assets(self):
        if not self.catalog_file.exists():
            raise FileNotFoundError(f"item_catalog.csv not found here: {self.catalog_file}")

        if not self.item_index_file.exists():
            raise FileNotFoundError(f"item_index.csv not found here: {self.item_index_file}")

        # Load the catalog and index files
        self.catalog_df = pd.read_csv(self.catalog_file)
        self.item_index_df = pd.read_csv(self.item_index_file)

        # Log columns to confirm structure
        print("\nItem Catalog Columns:")
        print(self.catalog_df.columns)

        # Ensure 'item_id' column exists
        if 'item_id' not in self.catalog_df.columns:
            raise ValueError(f"Missing 'item_id' column in item_catalog.csv")

        # Ensure 'item_name' column exists
        if 'item_name' not in self.catalog_df.columns:
            raise ValueError(f"Missing 'item_name' column in item_catalog.csv")

        # If 'category' column does not exist, create it
        if "category" not in self.catalog_df.columns:
            self.catalog_df["category"] = "Unknown"

        # Clean and convert columns
        self.catalog_df["item_id"] = pd.to_numeric(
            self.catalog_df["item_id"],
            errors="coerce"
        ).fillna(0).astype(int)

        self.catalog_df["item_name"] = self.catalog_df["item_name"].astype(str).str.strip()
        self.catalog_df["category"] = self.catalog_df["category"].astype(str).str.strip()

        # Remove duplicates based on item_id and item_name
        self.catalog_df = self.catalog_df.drop_duplicates(subset=["item_id"])
        self.catalog_df = self.catalog_df.drop_duplicates(subset=["item_name"])

        # Load item index and process
        self.item_index_df["item_index"] = pd.to_numeric(
            self.item_index_df["item_index"],
            errors="coerce"
        ).fillna(0).astype(int)

        self.item_index_df["item_name"] = self.item_index_df["item_name"].astype(str).str.strip()
        self.item_index_df = self.item_index_df[self.item_index_df["item_name"] != ""]
        self.item_index_df = self.item_index_df.drop_duplicates(subset=["item_index"])
        self.item_index_df = self.item_index_df.sort_values("item_index")

        # Populate item mapping dictionaries
        self.items = self.item_index_df["item_name"].tolist()

        self.item_to_idx = {
            item_name: idx
            for idx, item_name in enumerate(self.items)
        }

        self.idx_to_item = {
            idx: item_name
            for item_name, idx in self.item_to_idx.items()
        }

        self.item_id_to_name = dict(
            zip(self.catalog_df["item_id"], self.catalog_df["item_name"])
        )

        self.item_id_to_category = dict(
            zip(self.catalog_df["item_id"], self.catalog_df["category"])
        )

        self.item_name_to_item_id = dict(
            zip(self.catalog_df["item_name"], self.catalog_df["item_id"])
        )

        self.item_name_to_category = dict(
            zip(self.catalog_df["item_name"], self.catalog_df["category"])
        )

        # Load matrices and embeddings
        co_path = self.model_dir / "debug_co_occurrence_matrix.npz"
        main_emb_path = self.model_dir / "debug_main_item_user_svd_embedding.npy"
        backup_emb_path = self.model_dir / "debug_backup_co_occurrence_svd_embedding.npy"
        text_emb_path = self.model_dir / "debug_minilm_text_embedding.npy"

        if not co_path.exists():
            raise FileNotFoundError(f"Co occurrence matrix not found: {co_path}")

        if not main_emb_path.exists():
            raise FileNotFoundError(f"Main item embedding not found: {main_emb_path}")

        if not backup_emb_path.exists():
            raise FileNotFoundError(f"Backup item embedding not found: {backup_emb_path}")

        if not text_emb_path.exists():
            raise FileNotFoundError(f"MiniLM text embedding not found: {text_emb_path}")

        self.co_matrix = sparse.load_npz(co_path).tocsr()
        self.main_item_embedding = np.load(main_emb_path)
        self.backup_item_embedding = np.load(backup_emb_path)
        self.text_embedding = np.load(text_emb_path)

        self.validate_asset_shapes()

    def validate_asset_shapes(self):
        item_count = len(self.items)

        errors = []

        if self.co_matrix.shape[0] != item_count or self.co_matrix.shape[1] != item_count:
            errors.append(
                f"co_matrix shape {self.co_matrix.shape} does not match item count {item_count}"
            )

        if self.main_item_embedding.shape[0] != item_count:
            errors.append(
                f"main embedding row count {self.main_item_embedding.shape[0]} does not match item count {item_count}"
            )

        if self.backup_item_embedding.shape[0] != item_count:
            errors.append(
                f"backup embedding row count {self.backup_item_embedding.shape[0]} does not match item count {item_count}"
            )

        if self.text_embedding.shape[0] != item_count:
            errors.append(
                f"text embedding row count {self.text_embedding.shape[0]} does not match item count {item_count}"
            )

        if errors:
            error_text = "\n".join(errors)
            raise ValueError(error_text)

    def safe_cosine_vector(self, matrix, query_vector):
        scores = cosine_similarity(query_vector.reshape(1, -1), matrix)[0]
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        return scores

    def get_input_items_from_payload(self, payload):
        raw_items = payload.get("items", [])

        input_item_names = []
        missing_item_ids = []

        for item in raw_items:
            item_id = item.get("item_id")

            try:
                item_id = int(item_id)
            except Exception:
                missing_item_ids.append(item_id)
                continue

            item_name = self.item_id_to_name.get(item_id)

            if item_name is None:
                missing_item_ids.append(item_id)
                continue

            if item_name not in self.item_to_idx:
                missing_item_ids.append(item_id)
                continue

            input_item_names.append(item_name)

        input_item_names = list(dict.fromkeys(input_item_names))

        return input_item_names, missing_item_ids

    def build_basket_profile(self, input_item_names):
        input_indices = []

        for name in input_item_names:
            if name in self.item_to_idx:
                input_indices.append(self.item_to_idx[name])

        if len(input_indices) == 0:
            return None

        main_vector = self.main_item_embedding[input_indices].mean(axis=0)
        backup_vector = self.backup_item_embedding[input_indices].mean(axis=0)
        text_vector = self.text_embedding[input_indices].mean(axis=0)

        return {
            "input_indices": input_indices,
            "main_vector": main_vector,
            "backup_vector": backup_vector,
            "text_vector": text_vector
        }

    def normalize_scores(self, values):
        values = np.asarray(values, dtype=float)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        min_value = float(values.min())
        max_value = float(values.max())

        if max_value == min_value:
            return np.zeros_like(values)

        return (values - min_value) / (max_value - min_value)

    def recommend(self, payload, top_n=10):
        input_item_names, missing_item_ids = self.get_input_items_from_payload(payload)

        if len(input_item_names) == 0:
            return {
                "input_item_names": [],
                "missing_item_ids": missing_item_ids,
                "detected_cart_intent": "unknown",
                "recommendations": []
            }

        basket_profile = self.build_basket_profile(input_item_names)

        if basket_profile is None:
            return {
                "input_item_names": input_item_names,
                "missing_item_ids": missing_item_ids,
                "detected_cart_intent": "unknown",
                "recommendations": []
            }

        input_indices = basket_profile["input_indices"]
        input_index_set = set(input_indices)

        cart_intent = infer_basket_intent(input_item_names)
        input_families = {get_product_family(name) for name in input_item_names}

        main_scores = self.safe_cosine_vector(
            self.main_item_embedding,
            basket_profile["main_vector"]
        )

        backup_scores = self.safe_cosine_vector(
            self.backup_item_embedding,
            basket_profile["backup_vector"]
        )

        text_scores = self.safe_cosine_vector(
            self.text_embedding,
            basket_profile["text_vector"]
        )

        pair_counts = np.asarray(self.co_matrix[input_indices].sum(axis=0)).ravel()
        max_pair_count = max(float(pair_counts.max()), 1.0)
        pair_scores = np.log1p(pair_counts) / np.log1p(max_pair_count)

        main_scores = self.normalize_scores(main_scores)
        backup_scores = self.normalize_scores(backup_scores)
        text_scores = self.normalize_scores(text_scores)
        pair_scores = self.normalize_scores(pair_scores)

        rows = []

        for idx, candidate_name in enumerate(self.items):
            if idx in input_index_set:
                continue

            pair_count = float(pair_counts[idx])

            model_score = (
                0.30 * float(main_scores[idx])
                + 0.25 * float(backup_scores[idx])
                + 0.15 * float(text_scores[idx])
                + 0.30 * float(pair_scores[idx])
            )

            rule_score = rule_boost_for_basket(cart_intent, candidate_name)
            penalty = same_type_penalty(input_families, candidate_name)
            blocked = hard_block_candidate(cart_intent, candidate_name)

            final_score = model_score + rule_score
            final_score = final_score * (1.0 - penalty)

            if blocked:
                final_score = 0.0

            candidate_item_id = int(self.item_name_to_item_id.get(candidate_name, 0))
            candidate_category = self.item_name_to_category.get(candidate_name, "Unknown")
            candidate_family = get_product_family(candidate_name)

            rows.append({
                "item_name": candidate_name,
                "item_id": candidate_item_id,
                "category": candidate_category,
                "score": round(float(final_score), 6),
                "model_score": round(float(model_score), 6),
                "rule_score": round(float(rule_score), 6),
                "pair_count": int(pair_count),
                "item_user_sim": round(float(main_scores[idx]), 6),
                "co_sim": round(float(backup_scores[idx]), 6),
                "text_sim": round(float(text_scores[idx]), 6),
                "family": candidate_family,
                "blocked": bool(blocked)
            })

        result_df = pd.DataFrame(rows)

        if len(result_df) == 0:
            recommendations = []
        else:
            result_df = result_df[result_df["score"] > 0]
            result_df = result_df.sort_values(
                by=["score", "rule_score", "pair_count"],
                ascending=[False, False, False]
            )

            recommendations = result_df.head(top_n)[
                ["category", "item_name", "item_id", "score"]
            ].to_dict(orient="records")

        return {
            "input_item_names": input_item_names,
            "missing_item_ids": missing_item_ids,
            "detected_cart_intent": cart_intent,
            "recommendations": recommendations
        }