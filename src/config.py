from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
MODEL_DIR = PROJECT_ROOT / "models"

BASKET_DATA_PATH = DATA_INPUT_DIR / "customer_purchase_with_basket_list.csv"
MAIN_DATA_PATH = DATA_INPUT_DIR / "form_paste_clean_v2 .csv"

ITEM_CATALOG_PATH = OUTPUT_DIR / "item_catalog_from_basket.csv"

TEXT_EMBEDDING_MATRIX_PATH = MODEL_DIR / "product_text_embeddings_minilm.npy"
TEXT_EMBEDDING_METADATA_PATH = OUTPUT_DIR / "product_text_embedding_metadata.csv"

BASKET_ONE_HOT_MATRIX_PATH = MODEL_DIR / "basket_one_hot_matrix.npz"
ITEM_USER_MATRIX_PATH = MODEL_DIR / "item_user_matrix.npz"
COOCCURRENCE_MATRIX_PATH = MODEL_DIR / "cooccurrence_matrix.npz"

ITEM_USER_SVD_EMBEDDING_PATH = MODEL_DIR / "item_user_svd_embeddings.npy"
COOCCURRENCE_SVD_EMBEDDING_PATH = MODEL_DIR / "cooccurrence_svd_embeddings.npy"

FEATURE_ITEM_INDEX_PATH = OUTPUT_DIR / "feature_item_index.csv"
ITEM_POPULARITY_PATH = OUTPUT_DIR / "item_popularity.csv"
TOPK_RECOMMENDATION_PATH = OUTPUT_DIR / "topk_hybrid_recommendations.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 128
NORMALIZE_EMBEDDINGS = True

ITEMS_COLUMN = "items"
CUSTOMER_COLUMN = "customerId"
ITEM_NAME_COLUMN = "itemName"

SVD_DIM = 64
TOP_K = 5
CANDIDATE_POOL_SIZE = 120